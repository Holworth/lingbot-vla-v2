#!/usr/bin/env bash
# torchrun --no-python target used by train.sh when NSYS_STEPS is set: every rank runs under
# Nsight Systems and writes ${NSYS_OUT}_rank<R>.nsys-rep. nsys_probe.py opens and closes the
# capture window and then ends each rank with exit code 0, so every nsys writes its report.
# (With --capture-range-end=stop-shutdown nsys killed the ranks itself; torchrun took the nonzero
# exit as a failure and killed the slower nsys processes, rank 0 among them, mid-report.)
PY=$1; shift
# A machine allows one GPU-metrics sampling session: rank 0 samples all 8 GPUs, the other ranks none.
if [ "$RANK" = 0 ]; then GPUM="--gpu-metrics-devices=all"; else GPUM="--gpu-metrics-devices=none"; fi
# shellcheck disable=SC2086  # NSYS_ARGS is a list of extra options
exec nsys profile -t cuda,nvtx \
  --capture-range=cudaProfilerApi --capture-range-end=stop --wait=primary \
  $GPUM -f true -o "${NSYS_OUT}_rank${RANK}" ${NSYS_ARGS:-} "$PY" "$@"
