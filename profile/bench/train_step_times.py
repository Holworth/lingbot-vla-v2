#!/usr/bin/env python
"""Per-step times of train.sh runs, side by side: StepTime (the trainer's timer) and the wall gap between step lines.

    python profile/bench/train_step_times.py runs/rmsnorm_ab_base runs/rmsnorm_compiled_130 --names base compiled \
        --csv profile/bench/results/train_step_times.csv

Prints window statistics (step 1, warm-up 2-8, 9-30, 31-130, 9-last) and a 10-steps-per-row table of every step.
"""

import argparse
import csv
import re
import statistics
from datetime import datetime
from pathlib import Path

LINE = re.compile(r"(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d) - INFO - __main__ - Step (\d+)/\d+, .*?StepTime ([\d.]+)s")


def read(run):
    steps = {}
    for line in (Path(run) / "train.log").read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if m:
            steps[int(m.group(2))] = (datetime.strptime(m.group(1), "%m/%d/%Y %H:%M:%S"), float(m.group(3)))
    return steps


def pct(vals, q):
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--csv")
    ap.add_argument("--last", type=int, default=130)
    args = ap.parse_args()
    names = args.names or [Path(r).name for r in args.runs]
    data = [read(r) for r in args.runs]
    last = min(args.last, *(max(d) for d in data))
    windows = [("step 1", 1, 1), ("2-8", 2, 8), ("9-30", 9, 30), ("31-%d" % last, 31, last), ("9-%d" % last, 9, last)]
    print(f"StepTime per window (s): mean / p50 / p95 / min / max; steps 1-{last}")
    print(f"{'window':<10}" + "".join(f"{n:>46}" for n in names) + ("    diff of means" if len(names) == 2 else ""))
    for label, a, b in windows:
        cells, means = [], []
        for d in data:
            v = [d[s][1] for s in range(a, b + 1) if s in d]
            means.append(statistics.fmean(v))
            cells.append(f"{statistics.fmean(v):.3f} / {statistics.median(v):.3f} / {pct(v, .95):.3f} / {min(v):.3f} / {max(v):.3f}")
        diff = f"    {means[1] - means[0]:+.3f} ({(means[1] / means[0] - 1) * 100:+.1f}%)" if len(names) == 2 else ""
        print(f"{label:<10}" + "".join(f"{c:>46}" for c in cells) + diff)
    for n, d in zip(names, data):
        wall = [(d[s][0] - d[s - 1][0]).total_seconds() for s in range(32, last + 1) if s in d and s - 1 in d]
        print(f"{n}: wall clock between step lines, steps 32-{last}: mean {statistics.fmean(wall):.3f} s "
              f"(1 s timestamp resolution); steps 1-{last} took {(d[last][0] - d[1][0]).total_seconds() / 60:.1f} min after step 1")
    for n, d in zip(names, data):
        print(f"\n{n}: StepTime of every step (s), 10 per row")
        print("steps    " + "".join(f"{'+%d' % k:>7}" for k in range(10)))
        for row in range(0, last, 10):
            print(f"{row + 1:>3}-{min(row + 10, last):<4} " + "".join(
                f"{d[s][1]:>7.3f}" if s in d else f"{'-':>7}" for s in range(row + 1, min(row + 10, last) + 1)))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step"] + [f"{n}_steptime_s" for n in names] + [f"{n}_log_time" for n in names])
            for s in range(1, last + 1):
                w.writerow([s] + [f"{d[s][1]:.3f}" if s in d else "" for d in data]
                           + [d[s][0].strftime("%H:%M:%S") if s in d else "" for d in data])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
