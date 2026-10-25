"""Is the FSDP2 root unit's AllGather slow because its per-rank size is not a multiple of 16 bytes?

    python -m torch.distributed.run --standalone --nproc-per-node 8 \
        profile/nsys/analysis/nccl_align_test.py --reps 10 --warmup 3
    # dry run of the script itself on the CPU (gloo, no GPU); its times mean nothing
    CUDA_VISIBLE_DEVICES= python -m torch.distributed.run --standalone \
        --nproc-per-node 8 profile/nsys/analysis/nccl_align_test.py --backend gloo --device cpu --counts 7 8 16 24

Why: the root FSDP2 unit of the policy (ViT blocks, deepstack mergers, merger, embed_tokens, align heads, ...)
holds 955,727,159 parameters + 769 FSDP padding = 955,727,928 elements, n = 119,465,991 per rank on 8 GPUs.
Its bf16 AllGather moves 238,931,982 B per rank (% 16 = 14), its fp32 ReduceScatter 477,863,964 B per rank
(% 16 = 12). In the rank-0 nsys capture that AllGather took 395 ms (NCCL time there includes
waiting for the other ranks), while coll_bw.py's 25.8 GB/s busbw predicts about 65 ms for its 1.9 GB.
Hypothesis: rank r's slice starts r * n elements into the gathered buffer, so with n * 2 % 16 = 14 the slices
of ranks 1-7 are not 16-byte aligned and NCCL's copy loops lose their 16-byte vector loads and stores.
nccl-tests cannot show this: it rounds the per-rank count down to a multiple of 16 bytes
(nccl-tests src/all_gather.cu:11, src/reduce_scatter.cu:11).

For each per-rank element count n (--counts; default 7 counts around the root's: per-rank bf16 bytes % 16 of
0, 8, 12, 14, 0, 8, 0) every rank runs, with buffers allocated once per count and freed before the next:
  bf16 AllGather as FSDP2 issues it, in place: out = empty(n * world), inp = out.narrow(0, rank * n, n),
     dist.all_gather_into_tensor(out, inp). In torch 2.8, torch/distributed/fsdp/_fully_shard/
     _fsdp_collectives.py:103-111 (all_gather_copy_in_cuda) allocates the output and takes the input as that
     narrow of it, and :204 makes this call. --ag-input separate gives the input its own buffer instead.
  fp32 ReduceScatter as FSDP2 issues it: dist.reduce_scatter_tensor(out[n], inp[n * world], op=ReduceOp.AVG),
     both buffers allocated separately (:437, :453, call at :461-466). For an fp32 reduce_dtype FSDP2 uses
     NCCL's AVG with no pre- or post-divide (_get_gradient_divide_factors, :624 overflow_risk False, :631
     factor = world size, :637 `return None, None, ReduceOp.AVG, ReduceOp.AVG`), given the defaults
     gradient_divide_factor = None and force_sum_reduction_for_comms = False (_fsdp_param_group.py:182, :185);
     the repo sets a divide factor only on EP expert modules (lingbotvla/distributed/torch_parallelize.py).
     --rs-op sum uses ReduceOp.SUM instead.
After the timed reps each result is checked (AllGather: slice r holds rank r's value; ReduceScatter: rank r's
output holds the expected sum or average); a wrong result shows in the "check" column.

Timing: before every rep (--warmup untimed ones, then --reps timed ones) each rank runs cuda.synchronize,
barrier, cuda.synchronize; the rep is timed with two CUDA events on the current stream around the call (a
synchronous collective: the current stream does not go on before it ends). Each rep's time is the max over
ranks (one all_reduce MAX after the reps), so it includes waiting for the last rank to start, which the
barrier keeps small. Per count and collective the table gives the per-rank bytes and their % 16,
median / min / max over the reps in ms, and the bus bandwidth of the median in GB/s, as nccl-tests defines it
for both collectives: busbw = (world - 1) / world * total bytes / time, total bytes = per-rank bytes * world.

Decision rule, T = median bf16 AllGather time: the hypothesis holds iff
T(119465991) >= 200 ms and max(T(119465984), T(119465992), T(119466000)) <= 0.5 * T(119465991).
It is evaluated only when --counts contains these four counts.

NCCL settings: train.sh exports no NCCL_* variable (only TOKENIZERS_PARALLELISM, HF_*_OFFLINE,
TRANSFORMERS_OFFLINE and, depending on its options, MEM_PROBE_DIR, NSYS_STEPS/NSYS_OUT and
LINGBOT_SKIP_FINAL_CKPT: experiment/single_node_repro/train.sh), and the trainer calls
torch.cuda.set_device and dist.init_process_group(backend="nccl") without options
(tasks/vla/train_lingbotvla.py, main()), as this script does. Both therefore run with NCCL's defaults plus
whatever NCCL_* the launching shell exports (none on this machine, and no /etc/nccl.conf, see
experiment/single_node_repro/README.md), when launched the same way (torch.distributed.run --standalone, 8
processes; torchrun itself sets TORCH_NCCL_ASYNC_ERROR_HANDLING=1 for every worker, training included:
torch/distributed/elastic/agent/server/local_elastic_agent.py:321). The header lists the NCCL_* and
TORCH_NCCL_* variables and NCCL config files rank 0 sees. To test a setting, prefix it to the command, e.g.
NCCL_MIN_CTAS=8 or NCCL_IB_DISABLE=1; NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING shows the channels,
transports, algorithm and protocol NCCL picks. At the default counts each GPU needs about 6.2 GB (1.9 GB
AllGather output, 3.8 + 0.5 GB ReduceScatter input and output), so run it on idle GPUs.

CPU dry run (--backend gloo --device cpu): times are CPU wall times. In torch 2.8 gloo has
all_gather_into_tensor (in place too) and reduce_scatter_tensor, but its reduce_scatter_tensor returns the sum
for ReduceOp.AVG, so on gloo AVG is emulated as SUM followed by a division by the world size.
"""

import argparse
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist

COUNTS = [119_465_984, 119_465_988, 119_465_990, 119_465_991, 119_465_992, 119_465_996, 119_466_000]
ROOT = 119_465_991  # the root unit's elements per rank: 955,727,928 / 8
ALIGNED = (119_465_984, 119_465_992, 119_466_000)  # per-rank bf16 bytes % 16 == 0
SLOW_MS, RATIO = 200.0, 0.5  # decision rule thresholds


def busbw(per_rank_bytes, world, ms):
    """nccl-tests bus bandwidth in GB/s: (world - 1) / world * (per-rank bytes * world) / time."""
    return per_rank_bytes * (world - 1) / (ms * 1e-3) / 1e9


def fill_value(rank):
    return rank % 64 + 1  # exact in bf16


def timed_reps(fn, ctx):
    """This rank's times in ms of ctx.reps calls of fn, after ctx.warmup untimed ones."""
    cuda = ctx.device.type == "cuda"
    if cuda:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for i in range(ctx.warmup + ctx.reps):
        if cuda:
            torch.cuda.synchronize()
        ctx.barrier()
        if cuda:
            torch.cuda.synchronize()
            start.record()
            fn()
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            fn()
            ms = (time.perf_counter() - t0) * 1e3
        if i >= ctx.warmup:
            times.append(ms)
    return times


def all_gather(n, ctx):
    out = torch.empty(n * ctx.world, dtype=torch.bfloat16, device=ctx.device)
    if ctx.ag_input == "inplace":  # FSDP2's layout, _fsdp_collectives.py:103-111
        inp = out.narrow(0, ctx.rank * n, n)
    else:
        inp = torch.empty(n, dtype=torch.bfloat16, device=ctx.device)
    inp.fill_(fill_value(ctx.rank))
    times = timed_reps(lambda: dist.all_gather_into_tensor(out, inp), ctx)
    rows = out.view(ctx.world, n)
    expect = torch.tensor([fill_value(r) for r in range(ctx.world)], dtype=torch.bfloat16, device=ctx.device)
    ok = torch.equal(rows.amin(dim=1), expect) and torch.equal(rows.amax(dim=1), expect)
    return times, ok


def reduce_scatter(n, ctx):
    inp = torch.empty(n * ctx.world, dtype=torch.float32, device=ctx.device)
    for j, chunk in enumerate(inp.view(ctx.world, n)):  # the chunk for rank j holds (rank + 1) * (j + 1)
        chunk.fill_((ctx.rank + 1) * (j + 1))
    out = torch.empty(n, dtype=torch.float32, device=ctx.device)
    if ctx.rs_op == "avg" and ctx.backend == "gloo":  # gloo returns the sum for AVG: emulate (dry run only)
        def fn():
            dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
            out.div_(ctx.world)
    else:
        op = dist.ReduceOp.AVG if ctx.rs_op == "avg" else dist.ReduceOp.SUM

        def fn():
            dist.reduce_scatter_tensor(out, inp, op=op)
    times = timed_reps(fn, ctx)
    expect = (ctx.rank + 1) * ctx.world * (ctx.world + 1) / 2
    if ctx.rs_op == "avg":
        expect /= ctx.world
    lo, hi = (v.item() for v in out.aminmax())
    ok = abs(lo - expect) <= 1e-6 * expect and abs(hi - expect) <= 1e-6 * expect
    return times, ok


def max_over_ranks(values, ctx):
    t = torch.tensor(values, dtype=torch.float64, device=ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def nccl_settings():
    """Lines describing the NCCL configuration this process sees."""
    lines = []
    for prefix in ("NCCL_", "TORCH_NCCL_"):
        found = sorted(f"{k}={v}" for k, v in os.environ.items() if k.startswith(prefix))
        lines.append(f"{prefix}* in effect: {' '.join(found) if found else 'none'}")
    # NCCL also reads NCCL_CONF_FILE (default ~/.nccl.conf) and /etc/nccl.conf; the environment wins.
    files = [os.environ.get("NCCL_CONF_FILE") or os.path.expanduser("~/.nccl.conf"), "/etc/nccl.conf"]
    for path in files:
        if os.path.isfile(path):
            with open(path) as f:
                body = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
            lines.append(f"NCCL config file {path}: {' '.join(body) or '(empty)'}")
    if not any(os.path.isfile(p) for p in files):
        lines.append(f"NCCL config files: none ({', '.join(files)})")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        lines.append(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    return lines


def decision(ag_median):
    """The decision rule on {n: median bf16 AllGather ms}: (True/False, or None if counts are missing; text)."""
    missing = [n for n in (ROOT, *ALIGNED) if n not in ag_median]
    if missing:
        return None, f"not evaluated, --counts lacks {', '.join(map(str, missing))}"
    t, t_al = ag_median[ROOT], max(ag_median[n] for n in ALIGNED)
    slow, fast = t >= SLOW_MS, t_al <= RATIO * t
    return slow and fast, (
        f"T({ROOT}) = {t:.1f} ms {'>=' if slow else '<'} {SLOW_MS:.0f} ms; "
        f"max(T({ALIGNED[0]}), T({ALIGNED[1]}), T({ALIGNED[2]})) = {t_al:.1f} ms "
        f"{'<=' if fast else '>'} {RATIO} * T({ROOT}) = {RATIO * t:.1f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--counts", type=int, nargs="+", default=COUNTS, help="per-rank element counts")
    ap.add_argument("--reps", type=int, default=10, help="timed reps per count and collective")
    ap.add_argument("--warmup", type=int, default=3, help="untimed reps before them")
    ap.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--ag-input", choices=("inplace", "separate"), default="inplace",
                    help="AllGather input: a narrow of the output at rank * n as FSDP2 (default), or its own buffer")
    ap.add_argument("--rs-op", choices=("avg", "sum"), default="avg",
                    help="ReduceScatter op: avg as FSDP2 for an fp32 reduce_dtype (default), or sum")
    args = ap.parse_args()
    if (args.backend, args.device) not in (("nccl", "cuda"), ("gloo", "cpu")):
        ap.error("use --backend nccl --device cuda, or --backend gloo --device cpu for a dry run")
    if args.reps < 1 or args.warmup < 0 or min(args.counts) < 1:
        ap.error("need --reps >= 1, --warmup >= 0 and counts >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("no CUDA device; use --backend gloo --device cpu for a dry run")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)  # as tasks/vla/train_lingbotvla.py does
        args.device = torch.device("cuda", local_rank)
        args.barrier = lambda: dist.barrier(device_ids=[local_rank])
    else:
        args.device = torch.device("cpu")
        args.barrier = dist.barrier
    dist.init_process_group(backend=args.backend)
    args.rank, args.world = dist.get_rank(), dist.get_world_size()
    args.barrier()  # creates the communicator before anything is timed
    rank0 = args.rank == 0

    if rank0:
        try:
            v = torch.cuda.nccl.version()
            nccl = ".".join(map(str, v)) if isinstance(v, tuple) else str(v)
        except Exception:  # torch built without NCCL
            nccl = "n/a"
        where = torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else "CPU"
        print(f"nccl_align_test on {socket.gethostname()}: torch {torch.__version__}, NCCL {nccl}, "
              f"backend {args.backend}, world size {args.world}, {where}", flush=True)
        for line in nccl_settings():
            print("  " + line)
        ag = ("in place, input = output.narrow(0, rank * n, n) as FSDP2" if args.ag_input == "inplace"
              else "input in its own buffer")
        rs = f"ReduceOp.{args.rs_op.upper()}" + (" as FSDP2" if args.rs_op == "avg" else "")
        if args.rs_op == "avg" and args.backend == "gloo":
            rs += ", emulated as SUM / world size on gloo"
        timer = "CUDA events on the current stream" if args.device.type == "cuda" else "CPU wall time"
        print(f"  bf16 AllGather {ag}; fp32 ReduceScatter {rs}")
        print(f"  reps {args.reps} (after {args.warmup} warm-up); time of a rep = max over ranks ({timer}), ms; "
              f"busbw = (world - 1) / world * total bytes / median time, GB/s\n")
        cols = f"{'B/rank':>13} {'%16':>3} {'median':>8} {'min':>8} {'max':>8} {'busbw':>7}"
        print(f"{'':>13} | {'bf16 AllGather':^52} | {'fp32 ReduceScatter':^52} |")
        print(f"{'n/rank':>13} | {cols} | {cols} | check", flush=True)

    ag_median, wrong = {}, []
    for n in args.counts:
        ag_times, ag_ok = all_gather(n, args)
        rs_times, rs_ok = reduce_scatter(n, args)
        if args.device.type == "cuda":
            torch.cuda.empty_cache()  # the buffers went out of scope with the two calls
        m = max_over_ranks(ag_times + rs_times + [float(not ag_ok), float(not rs_ok)], args)
        ag_times, rs_times, bad = m[:args.reps], m[args.reps:2 * args.reps], m[2 * args.reps:]
        ag_median[n] = statistics.median(ag_times)
        check = " + ".join(name for name, b in zip(("AG", "RS"), bad) if b) or "ok"
        if check != "ok":
            check += " WRONG"
            wrong.append(n)
        if rank0:
            cells = []
            for times, size in ((ag_times, 2), (rs_times, 4)):
                med, nbytes = statistics.median(times), n * size
                cells.append(f"{nbytes:>13,} {nbytes % 16:>3} {med:>8.2f} {min(times):>8.2f} {max(times):>8.2f} "
                             f"{busbw(nbytes, args.world, med):>7.2f}")
            print(f"{n:>13,} | {cells[0]} | {cells[1]} | {check}{'  <- root unit' if n == ROOT else ''}", flush=True)

    if rank0:
        holds, text = decision(ag_median)
        verdict = "" if holds is None else f" -> hypothesis {'holds' if holds else 'does not hold'}"
        print(f"\ndecision rule (T = median bf16 AllGather time): {text}{verdict}")
        if wrong:
            print(f"WARNING: wrong collective results for n = {', '.join(map(str, wrong))}; do not trust these times")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
