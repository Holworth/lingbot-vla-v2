"""Run a training script with NVTX step markers and an Nsight Systems capture window.

    NSYS_STEPS=31-35 torchrun ... experiment/single_node_repro/nsys_probe.py tasks/vla/train_lingbotvla.py <args>

train.sh sets this up when NSYS_STEPS is given (rank 0, or every rank with NSYS_ALL_RANKS=1, runs under
`nsys profile --capture-range=cudaProfilerApi`, see nsys_rank0.sh / nsys_allranks.sh). Steps are counted with global
optimizer-step hooks (plus a wrapper around CombinedOptimizer.step, which bypasses
Optimizer.__init__ and so the hooks), so the trainer runs unchanged:
  - cudaProfilerStart right after optimizer step FIRST-1, cudaProfilerStop right after
    optimizer step LAST, so the capture holds steps FIRST..LAST; after a barrier the process then
    exits with code 0, which lets each nsys finish its report before torchrun tears the job down
    (so the trainer never logs step LAST, and nsys closes the ranges still open in it);
  - an NVTX range "step N" from the end of optimizer step N-1 to the end of step N
    (it therefore includes logging and the wait for step N's batch);
  - an NVTX range per optimizer.step call, named after the optimizer class (nested when
    a combined optimizer steps its children).
"""

import os
import runpy
import sys

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

from lingbotvla.optim.optimizer import CombinedOptimizer


def main():
    script, sys.argv = sys.argv[1], sys.argv[1:]
    first, last = map(int, os.environ["NSYS_STEPS"].split("-"))
    if not 2 <= first <= last:
        sys.exit(f"NSYS_STEPS={os.environ['NSYS_STEPS']}: need 2 <= FIRST <= LAST (the capture starts after optimizer "
                 "step FIRST-1)")
    state = {"depth": 0, "done": 0}  # optimizer.step nesting depth, finished outer steps

    def pre_hook(optimizer, args, kwargs):
        state["depth"] += 1
        torch.cuda.nvtx.range_push(f"{type(optimizer).__name__}.step")

    def post_hook(optimizer, args, kwargs):
        torch.cuda.nvtx.range_pop()
        state["depth"] -= 1
        if state["depth"]:
            return
        state["done"] += 1
        step = state["done"]
        if first <= step <= last:
            torch.cuda.nvtx.range_pop()  # "step <step>"
        if step == last:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            # All ranks reach this point at the same step; once all have, exit cleanly instead of finishing it.
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        if step == first - 1:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
        if first <= step + 1 <= last:
            torch.cuda.nvtx.range_push(f"step {step + 1}")

    register_optimizer_step_pre_hook(pre_hook)
    register_optimizer_step_post_hook(post_hook)
    combined_step = CombinedOptimizer.step

    def step(self, *args, **kwargs):
        pre_hook(self, args, kwargs)
        out = combined_step(self, *args, **kwargs)
        post_hook(self, args, kwargs)
        return out

    CombinedOptimizer.step = step
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
