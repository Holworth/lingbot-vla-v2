"""Measure all-gather / reduce-scatter bus bandwidth across all local GPUs.

    torchrun --nproc_per_node 8 experiment/single_node_repro/coll_bw.py

Bus bandwidth follows the nccl-tests definition: busbw = size / time * (n - 1) / n, where
size is the full (gathered) buffer. algbw (size / time) is busbw * n / (n - 1), i.e. 8/7 of it
on 8 GPUs, so check which one a quoted number is before comparing. NCCL settings come from
the environment, e.g. prefix the command with NCCL_MIN_CTAS=8 to compare configurations, or
with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,TUNING to see the channels, transports and
algorithm/protocol NCCL picks.
"""

import os
import time

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
for dtype in (torch.bfloat16, torch.float32):
    for mb in (64, 256, 1024, 2048):
        numel = mb * 2**20 // dtype.itemsize  # full (gathered) buffer size
        full = torch.empty(numel, dtype=dtype, device="cuda")
        shard = torch.ones(numel // world, dtype=dtype, device="cuda")
        res = {}
        for name, fn in (
            ("allgather", lambda: dist.all_gather_into_tensor(full, shard)),
            ("reducescatter", lambda: dist.reduce_scatter_tensor(shard, full)),
        ):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            dist.barrier()
            iters = 20
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - start) / iters
            res[name] = (dt, numel * dtype.itemsize / dt * (world - 1) / world / 1e9)
        if rank == 0:
            print(
                f"{str(dtype).split('.')[1]:9s} {mb:5d} MB  "
                f"AG {res['allgather'][0] * 1e3:7.1f} ms busbw {res['allgather'][1]:5.1f} GB/s   "
                f"RS {res['reducescatter'][0] * 1e3:7.1f} ms busbw {res['reducescatter'][1]:5.1f} GB/s",
                flush=True,
            )
dist.destroy_process_group()
