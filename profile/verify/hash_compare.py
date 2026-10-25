"""Compare the per-rank gradient / parameter md5 dumps of several runs at one step (LINGBOT_STATE_HASH_STEPS, see
lingbotvla/utils/state_hash.py), e.g. to show that a scheduling change leaves the training bitwise unchanged.

    python profile/verify/hash_compare.py <step> [--noise <run dir>] <ref run dir> <run dir> ...

Prints, per run: step-s loss / GradNorm (TensorBoard float32), and for each of the 8 ranks the TOTAL md5 of the gradient
and parameter files, plus the number of (parameter, rank) entries whose md5 differs from the reference run.
--noise: a rerun of the reference configuration; the parameters whose md5 differs between it and the reference on any
rank are nondeterministic without any change, and differences are then counted separately inside / outside that set."""
import os
import sys

from tb_scalars import read_scalars  # profile/verify/tb_scalars.py, next to this script

args = sys.argv[1:]
step = int(args.pop(0))
noise_run = None
if args and args[0] == "--noise":
    noise_run = args[1]; args = args[2:]
runs = args

def read(run, what, rank):
    p = os.path.join(run, f"state_hash_{what}_rank{rank}_step{step}.txt")
    if not os.path.exists(p): return None, {}
    per, total = {}, None
    for line in open(p):
        if line.startswith("TOTAL "): total = line.split()[1]
        elif line.strip():
            name, h = line.rsplit(None, 1); per[name.strip()] = h
    return total, per

ranks = range(8)
ref = runs[0]
refd = {(w, r): read(ref, w, r) for w in ("grad", "param") for r in ranks}
noise = set()
if noise_run:
    for w in ("grad", "param"):
        for r in ranks:
            per = read(noise_run, w, r)[1]
            noise |= {k for k, h in refd[(w, r)][1].items() if per.get(k) != h}
print(f"step {step}; reference {os.path.basename(ref)}" + (f"; noise set from {os.path.basename(noise_run)}: {len(noise)} parameters "
      f"({', '.join(sorted(k.replace('_orig_mod.', '') for k in noise))})" if noise_run else ""))
for run in runs:
    tb = read_scalars(os.path.join(run, "runs"), ["training/loss", "training/grad_norm"])
    l, g = tb.get("training/loss", {}).get(step), tb.get("training/grad_norm", {}).get(step)
    print(f"== {os.path.basename(run):18s} loss {l!r}  GradNorm {g!r}")
    for w in ("grad", "param"):
        cells, ndiff, nmiss, nout = [], 0, 0, 0
        for r in ranks:
            tot, per = read(run, w, r)
            rtot, rper = refd[(w, r)]
            if tot is None: cells.append("missing"); nmiss += 1; continue
            dk = [k for k in rper if per.get(k) != rper[k]] + [k for k in per if k not in rper]
            ndiff += len(dk); nout += sum(1 for k in dk if k not in noise)
            cells.append(tot[:8] + ("" if tot == rtot else "*"))
        same = nmiss == 0 and ndiff == 0
        verdict = "IDENTICAL" if same else ("IDENTICAL outside the noise set" if noise_run and nmiss == 0 and nout == 0 else "DIFFERENT")
        print(f"   {w:5s} TOTAL md5 per rank: {' '.join(cells)}   (parameter, rank) entries differing: {ndiff}"
              f"{f' ({nout} outside the noise set)' if noise_run else ''}{'' if nmiss == 0 else f', {nmiss} ranks missing'} -> {verdict}")
