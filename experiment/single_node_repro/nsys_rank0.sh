#!/usr/bin/env bash
# torchrun --no-python target used by train.sh when NSYS_STEPS is set: rank 0 runs under
# Nsight Systems, every other rank runs plainly. nsys_probe.py opens and closes the capture
# window, and nsys stops the job once the window closes (stop-shutdown).
#   nsys_rank0.sh <python> <script> [args...]
PY=$1; shift
if [ "$RANK" = 0 ]; then
  # shellcheck disable=SC2086  # NSYS_ARGS is a list of extra options
  exec nsys profile -t cuda,nvtx,osrt,cublas,cudnn \
    --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
    --gpu-metrics-devices="$LOCAL_RANK" -f true -o "$NSYS_OUT" ${NSYS_ARGS:-} "$PY" "$@"
fi
exec "$PY" "$@"
