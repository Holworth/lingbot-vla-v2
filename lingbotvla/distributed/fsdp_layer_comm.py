"""Scheduling helpers for the per-layer FSDP2 units of the joint VLM / action-expert stack.

I. no_reshard_after_backward: keep each layer's unsharded parameters through the whole backward so the second
   (qkv) call's backward does not re-all-gather them. The caller must call after_backward() after every
   loss.backward(), before clip_grad_norm_ / optimizer.step(): it drains the reduce-scatter queue (when
   ReduceScatterInflight is installed) and reshards the layers.
H. forward prefetch depth k: at the start of a step (before the teacher phase, when the NCCL queue is idle) all-gather
   the first k layers of both towers; inside the layer loop FSDP keeps a lookahead of k via
   set_modules_to_forward_prefetch, so the all-gather chain runs ~k*14 ms ahead of the compute.
   With prefix_units (the root and the units the forward calls before the layers, in call order) the prologue is meant
   to run right before the forward instead: it all-gathers those units first, then the first k layers, so the layers'
   all-gathers run under the ViT / prefix compute and none of the prefix units queues behind them (the NCCL
   communicator runs collectives in issue order). This does not need the teacher phase.
J. reduce-scatters in flight: let up to D of FSDP2's backward reduce-scatters be in flight instead of one
   (ReduceScatterInflight below); after_backward() drains the queue.
"""
import collections
import inspect
import os
from typing import List, Optional

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _pg


class ReduceScatterInflight:
    """Let up to `depth` FSDP2 reduce-scatters be in flight instead of one.

    FSDP2's FSDPParamGroup.post_backward makes the compute stream wait, before issuing a reduce-scatter, for
    (a) the previous reduce-scatter (any group) so that its input buffer can be freed, and
    (b) this group's previous post-reduce event.
    Both are stream-ordering safety waits, not data dependencies: the RS inputs are only read on the RS stream, and
    the per-group gradient accumulation happens on the RS stream in issue order.  We take the two objects out of
    FSDP's hands before the original post_backward runs (so it finds nothing to wait for), keep them in a FIFO, and
    only when more than `depth - 1` are pending do we make the compute stream wait for the oldest and drop it
    (freeing its input).  `drain()` must run after loss.backward() and before anything reads gradients.
    depth=1 reproduces FSDP2's behaviour exactly (every entry is waited for at the next post_backward).
    """

    def __init__(self, depth: int):
        assert depth >= 1, depth
        self.depth = int(depth)
        self.pending: collections.deque = collections.deque()   # [(ReduceScatterState | None, post_reduce_event | None)]
        self._orig = None
        self.max_pending = 0
        self.stress_ticks = 0   # LINGBOT_RS_STRESS_MS (test only): GPU sleep put on the RS stream before every reduce-scatter

    @staticmethod
    def check(groups) -> None:
        """The configuration the deferred waits rely on: no HSDP all-reduce, no all-reduce hook, RS buffers from the
        caching allocator, no CPU offload (so the post-reduce runs on the RS stream and nothing else uses the events)."""
        for g in groups:                                # FSDPParamGroup objects
            assert not g._is_hsdp and g._all_reduce_hook is None and not g.allocate_memory_from_process_group
            assert all(not getattr(p, "offload_to_cpu", False) for p in g.fsdp_params)

    def install(self, groups_to_check) -> None:
        src = inspect.getsource(_pg.FSDPParamGroup.post_backward)
        assert "reduce_scatter_state" in src and "_wait_for_post_backward" in src, "FSDP2 post_backward changed: ReduceScatterInflight was written against torch 2.8's FSDPParamGroup.post_backward"
        assert not getattr(_pg.FSDPParamGroup, "_lingbot_rs_inflight_installed", False), "ReduceScatterInflight installed twice"
        self.check(groups_to_check)
        orig = _pg.FSDPParamGroup.post_backward          # may already be wrapped: wrap whatever is installed
        self._orig = orig
        inflight = self
        stress_ms = float(os.environ.get("LINGBOT_RS_STRESS_MS", "0") or 0)
        if stress_ms > 0:
            # delay every reduce-scatter by stress_ms on its own stream: it widens the window in which a gradient or an RS
            # input could be used before the RS has finished, so bit-exact results then show that the waits suffice
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            torch.cuda._sleep(10**7)
            end.record()
            end.synchronize()
            self.stress_ticks = int(stress_ms * 1e7 / start.elapsed_time(end))

        def post_backward(group, *unused):
            ctx = group.comm_ctx
            prev_state, ctx.reduce_scatter_state = ctx.reduce_scatter_state, None     # (a)
            prev_evt, group._post_reduce_event = group._post_reduce_event, None       # (b)
            if prev_state is not None or prev_evt is not None:
                inflight.pending.append((prev_state, prev_evt))
                inflight.max_pending = max(inflight.max_pending, len(inflight.pending))
            while len(inflight.pending) > inflight.depth - 1:
                inflight._wait_oldest()
            if inflight.stress_ticks:
                with torch.cuda.stream(ctx.reduce_scatter_stream):
                    torch.cuda._sleep(inflight.stress_ticks)
            return orig(group, *unused)

        _pg.FSDPParamGroup.post_backward = post_backward
        _pg.FSDPParamGroup._lingbot_rs_inflight_installed = True

    def _wait_oldest(self) -> None:
        state, evt = self.pending.popleft()
        cur = torch.cuda.current_stream()
        if state is not None and state.event is not None:
            cur.wait_event(state.event)
        if evt is not None:
            cur.wait_event(evt)
        # dropping `state` here frees the reduce-scatter input, ordered after the RS on the compute stream

    def drain(self) -> None:
        """After loss.backward(), before clip / optimizer: wait for everything still pending."""
        while self.pending:
            self._wait_oldest()


class LayerCommScheduler:
    def __init__(self, vlm_layers: List[torch.nn.Module], expert_layers: List[torch.nn.Module],
                 prefetch_depth: int = 0, no_reshard_after_backward: bool = False,
                 prefix_units: Optional[List[torch.nn.Module]] = None, rs_inflight_depth: int = 1):
        assert len(vlm_layers) == len(expert_layers), (len(vlm_layers), len(expert_layers))
        for m in list(vlm_layers) + list(expert_layers) + list(prefix_units or []):
            assert isinstance(m, FSDPModule), f"{type(m).__name__} is not an FSDP2 module"
        self.vlm, self.exp = list(vlm_layers), list(expert_layers)
        self.n = len(self.vlm)
        self.depth = int(prefetch_depth)
        self.no_reshard = bool(no_reshard_after_backward)
        self.prefix = list(prefix_units or [])
        self.rs_inflight_depth = int(rs_inflight_depth)
        assert 0 <= self.depth < self.n and self.rs_inflight_depth >= 1

    # ---- one-time configuration, right after build_parallelize_model
    def configure(self) -> None:
        if self.no_reshard:
            for m in self.vlm + self.exp:
                m.set_reshard_after_backward(False, recurse=False)
        if self.depth > 0:
            for i in range(self.n - self.depth):
                # issued in layer i's pre-forward (its first call, compute_kqv=True), on FSDP's own all-gather streams
                self.vlm[i].set_modules_to_forward_prefetch([self.vlm[i + self.depth], self.exp[i + self.depth]])
        self.rs_inflight = None
        # LINGBOT_RS_INFLIGHT_WRAP_AT_1=1 (test only): install the post_backward wrapper even at depth 1, where it must
        # reproduce FSDP2's own behaviour; that run checks that the wrapper itself changes nothing
        if self.rs_inflight_depth > 1 or os.environ.get("LINGBOT_RS_INFLIGHT_WRAP_AT_1") == "1":
            groups = [m._get_fsdp_state()._fsdp_param_group for m in self.vlm + self.exp]
            self.rs_inflight = ReduceScatterInflight(self.rs_inflight_depth)
            self.rs_inflight.install(groups)

    # ---- H: start of every forward (micro-batch), before the teacher phase; with prefix_units right before the forward,
    #      after the teacher. The callers skip global step 1 (not required: after a resume the first call runs before
    #      FSDP's lazy init, which FSDPModule.unshard supports)
    def prologue(self) -> None:
        if self.depth == 0:
            return
        for m in self.prefix:
            m.unshard(async_op=True)
        for i in range(self.depth):
            self.vlm[i].unshard(async_op=True)
            self.exp[i].unshard(async_op=True)

    # ---- I: right after loss.backward(), before clip / optimizer.step()
    def after_backward(self) -> None:
        if self.rs_inflight is not None:
            self.rs_inflight.drain()
        if self.no_reshard:
            for m in self.vlm + self.exp:
                m.reshard()
