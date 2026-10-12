#!/usr/bin/env bash
# 8x RTX PRO 5000 72GB layout: micro 32 x 8 GPUs x grad-accum 1 = global 256, with
# activation recompute so micro 32 fits in 72 GB. Everything else is train.sh.
#
#   OPTIMIZER=dist_muon|muon MODEL_ROOT=... DATA_LIST=... \
#     bash experiment/single_node_repro/train_pro5000.sh <run_name> [--key value ...]
#
# OPTIMIZER defaults to dist_muon. MICRO/GA/NPROC/MAX_STEPS/TIMING_ONLY work as in train.sh.
set -euo pipefail
NAME=${1:?usage: train_pro5000.sh <run_name> [--key value ...]}; shift
OPTIMIZER=${OPTIMIZER:-dist_muon}
case "$OPTIMIZER" in dist_muon|muon) ;; *) echo "OPTIMIZER must be dist_muon or muon" >&2; exit 1;; esac
export MICRO=${MICRO:-32} GA=${GA:-1}
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh" "$NAME" \
  --train.enable_gradient_checkpointing true --train.optimizer "$OPTIMIZER" "$@"
