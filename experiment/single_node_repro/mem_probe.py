"""Run a training script while recording this rank's PyTorch memory peaks once per second.

    torchrun ... experiment/single_node_repro/mem_probe.py tasks/vla/train_lingbotvla.py <args>

The trainer computes max_memory_allocated/reserved but never logs them, so this wrapper
samples them from a background thread and resets the peak after each sample. Each row of
$MEM_PROBE_DIR/mem_rank<R>.csv is "unix_time,allocated_peak_bytes,reserved_peak_bytes" for
the preceding second. The trainer itself runs unchanged (it never reads these peaks for
anything but an unlogged metric).
"""

import os
import runpy
import sys
import threading
import time

import torch


def _sample(path, device):
    while not torch.cuda.is_initialized():
        time.sleep(1)
    with open(path, "w", buffering=1) as f:
        while True:
            time.sleep(1)
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            f.write(f"{time.time():.1f},{allocated},{reserved}\n")


def main():
    script, sys.argv = sys.argv[1], sys.argv[1:]
    out_dir = os.environ["MEM_PROBE_DIR"]
    rank, local_rank = os.environ.get("RANK", "0"), int(os.environ.get("LOCAL_RANK", "0"))
    thread = threading.Thread(
        target=_sample, args=(os.path.join(out_dir, f"mem_rank{rank}.csv"), local_rank), daemon=True
    )
    thread.start()
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
