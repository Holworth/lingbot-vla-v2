#!/usr/bin/env bash
# torchrun --no-python target used by train.sh when NSYS_STEPS is set: rank 0 runs under
# Nsight Systems, every other rank runs plainly. Only CUDA and NVTX are traced, and GPU metrics
# are sampled on rank 0's GPU only, to keep the report small. nsys_probe.py opens and closes the
# capture window and then ends every rank with exit code 0, so nsys can finish writing the report.
#   nsys_rank0.sh <python> <script> [args...]
PY=$1; shift
if [ "$RANK" = 0 ]; then
  # --gpu-metrics-devices numbers the GPUs its own way, not like CUDA (on the 8x RTX PRO 5000 node its 0 is
  # CUDA's 2), so look this rank's GPU up in its list by PCI address.
  pci=$("$PY" -c "import torch; p = torch.cuda.get_device_properties($LOCAL_RANK); \
print(f'{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.')")
  metrics_gpu=$(nsys profile --gpu-metrics-devices=help 2>&1 | grep -i "PCI\[$pci" | sed -E 's/^[[:space:]]*([0-9]+):.*/\1/')
  if [ -z "$metrics_gpu" ]; then echo "nsys_rank0.sh: no GPU-metrics device at PCI $pci" >&2; exit 1; fi
  echo "nsys_rank0.sh: rank 0 is on cuda:$LOCAL_RANK, PCI $pci, nsys GPU-metrics device $metrics_gpu" >&2
  # shellcheck disable=SC2086  # NSYS_ARGS is a list of extra options
  exec nsys profile -t cuda,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop --wait=primary \
    --gpu-metrics-devices="$metrics_gpu" -f true -o "$NSYS_OUT" ${NSYS_ARGS:-} "$PY" "$@"
fi
exec "$PY" "$@"
