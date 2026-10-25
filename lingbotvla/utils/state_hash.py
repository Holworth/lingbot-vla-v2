"""Debug hooks for bit-exact comparisons of training runs (all off by default).

LINGBOT_STATE_HASH_STEPS=1,2,5    per rank, md5 of every parameter's gradient (after clip_grad_norm_, before
                                  optimizer.step()) and of every parameter (after optimizer.step()), written to
                                  <output_dir>/state_hash_{grad,param}_rank{r}_step{s}.txt with a TOTAL line
LINGBOT_PARAMS_DUMP_STEPS=1,130   per rank, every parameter's local shard as fp32 after optimizer.step(), saved to
                                  <output_dir>/params_rank{r}_step{s}.pt (about 2 GB per rank and step)
"""
import hashlib
import os

import torch
from torch.distributed.tensor import DTensor


def steps_from_env(name):
    return {int(s) for s in os.environ.get(name, "").split(",") if s.strip()}


def _md5(t):
    if t is None: return "none"
    t = t._local_tensor if isinstance(t, DTensor) else t
    t = t.detach().contiguous().reshape(-1)                 # reshape(-1) so 0-dim tensors can be viewed as bytes
    return hashlib.md5(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def dump_state_hash(model, step, rank, out_dir, what):      # what in ("grad", "param")
    lines = []
    for name, p in model.named_parameters():
        lines.append(f"{name}  {_md5(p.grad) if what == 'grad' else _md5(p.data)}")
    body = "\n".join(lines)
    total = hashlib.md5(body.encode()).hexdigest()
    with open(os.path.join(out_dir, f"state_hash_{what}_rank{rank}_step{step}.txt"), "w") as f:
        f.write(body + f"\nTOTAL {total}\n")
    return total


def dump_params(model, step, rank, out_dir):
    """Every parameter's local shard (DTensor._local_tensor) as fp32 on the CPU, keyed by parameter name."""
    state = {}
    for name, p in model.named_parameters():
        t = p.data._local_tensor if isinstance(p.data, DTensor) else p.data
        state[name] = t.detach().float().cpu().clone()
    path = os.path.join(out_dir, f"params_rank{rank}_step{step}.pt")
    torch.save(state, path)
    return path
