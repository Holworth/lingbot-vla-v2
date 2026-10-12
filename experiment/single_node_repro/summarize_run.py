"""Summarize step time, wall-clock time, loss and memory of a train.sh run.

    python experiment/single_node_repro/summarize_run.py <run_dir> [--first 31] [--last 130]

Steps before --first are warm-up (step 1 includes torch.compile, steps 2-4 recompile)
and are left out. StepTime is the trainer's own timer: it excludes waiting for the
next batch. "Wall / step" is the gap between log timestamps, so it also includes data
waits and logging; the two should agree within a few milliseconds when loading is hidden.
"""

import argparse
import csv
import re
import statistics
from datetime import datetime
from pathlib import Path

LINE = re.compile(
    r"(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d) - INFO - __main__ - Step (\d+)/\d+, .*?Loss ([\d.]+),.*?StepTime ([\d.]+)s"
)


def percentile(values, q):
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--first", type=int, default=31)
    parser.add_argument("--last", type=int, default=130)
    args = parser.parse_args()

    steps = {}
    for line in (args.run_dir / "train.log").read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if m:
            ts, step, loss, step_time = m.groups()
            steps[int(step)] = (datetime.strptime(ts, "%m/%d/%Y %H:%M:%S"), float(loss), float(step_time))
    if not steps:
        raise SystemExit("no step lines in train.log")
    last = min(args.last, max(steps))
    window = [s for s in range(args.first, last + 1) if s in steps]
    if not window:
        raise SystemExit(f"no steps logged in [{args.first}, {last}]; latest is {max(steps)}")

    times = [steps[s][2] for s in window]
    print(f"steps {window[0]}-{window[-1]} ({len(window)} steps)")
    print(f"  StepTime  mean {statistics.mean(times):.3f}s  p50 {percentile(times, 0.5):.3f}s  "
          f"p95 {percentile(times, 0.95):.3f}s")
    if window[0] - 1 in steps:
        span = (steps[window[-1]][0] - steps[window[0] - 1][0]).total_seconds()
        print(f"  Wall / step {span / (window[-1] - window[0] + 1):.3f}s")
    first_step, last_step = min(steps), max(steps)
    print(f"loss: step {first_step} {steps[first_step][1]:.4f} -> step {last_step} {steps[last_step][1]:.4f}")
    total = (steps[last_step][0] - steps[first_step][0]).total_seconds()
    print(f"logged steps {first_step}-{last_step} took {total / 3600:.2f} h")

    probes = sorted(args.run_dir.glob("mem_rank*.csv"))
    if probes:
        start = steps.get(window[0] - 1, steps[window[0]])[0].timestamp()  # end of the step before
        end = steps[window[-1]][0].timestamp()
        gib = 1024**3
        window_peak, run_peak = [0, 0], [0, 0]
        for probe in probes:
            for line in probe.read_text().splitlines():
                try:
                    t, allocated, reserved = float(line.split(",")[0]), *map(int, line.split(",")[1:3])
                except ValueError:
                    continue
                run_peak = [max(run_peak[0], allocated), max(run_peak[1], reserved)]
                if start <= t <= end:
                    window_peak = [max(window_peak[0], allocated), max(window_peak[1], reserved)]
        print(f"PyTorch peak per GPU (max of {len(probes)} ranks): allocated {window_peak[0] / gib:.3f} GiB, "
              f"reserved {window_peak[1] / gib:.3f} GiB in the window; "
              f"whole run {run_peak[0] / gib:.3f} / {run_peak[1] / gib:.3f} GiB")

    smi = args.run_dir / "smi.csv"
    if smi.exists():
        rows = []
        with smi.open() as f:
            for row in csv.reader(f):
                try:
                    rows.append((datetime.strptime(row[0].strip()[:19], "%Y/%m/%d %H:%M:%S"),
                                 row[1].strip(), int(row[2]), [c.strip() for c in row[4:8]]))
                except (IndexError, ValueError):
                    continue
        start, end = steps[window[0]][0], steps[window[-1]][0]
        in_window = [r for r in rows if start <= r[0] <= end]
        if in_window:
            print(f"nvidia-smi peak per GPU: {max(r[2] for r in in_window) / 1024:.2f} GiB in the window, "
                  f"{max(r[2] for r in rows) / 1024:.2f} GiB over the whole run")
            # Older smi.csv files have no temperature/throttle columns.
            thermal = [r for r in in_window if len(r[3]) == 4]
            if thermal:
                max_temp = max(int(r[3][0]) for r in thermal)
                throttled = sorted({r[1] for r in thermal if "Active" in (r[3][2], r[3][3])})
                print(f"max GPU temperature in the window {max_temp} C; thermal throttling "
                      + (f"on GPU {', '.join(throttled)} -- step times are not trustworthy" if throttled else "none"))


if __name__ == "__main__":
    main()
