#!/usr/bin/env bash
# Single-node RoboTwin 2.0 clean post-training with the official robotwin_dist_muon.yaml.
# Only paths and the batch layout differ from the official config: the official file assumes
# 32 GPUs (micro 32, global 1024); on one 8-GPU node micro 32 runs out of memory without
# recompute, so the default here is micro 16 x 8 GPUs x grad-accum 2 = global 256.
#
#   MODEL_ROOT=... DATA_LIST=... bash experiment/single_node_repro/train.sh <run_name> [--key value ...]
#
# Environment:
#   MODEL_ROOT   directory filled by download_models.sh (required)
#   DATA_LIST    list written by prepare_robotwin_clean.sh (required)
#   OUT_ROOT     runs go to $OUT_ROOT/<run_name> (default ./output)
#   PY           python of the training env (default: python on PATH)
#   NPROC        GPUs on this node (default 8)
#   MICRO, GA    micro batch per GPU and gradient accumulation (default 16, 2)
#   MAX_STEPS    default 5000; SAVE_STEPS default 2500
#   TIMING_ONLY  1 = stop as soon as step MAX_STEPS is logged, skipping the final
#                checkpoint (use with MAX_STEPS=130 for a quick speed test)
#   MEM_PROBE    1 (default) = run the trainer through mem_probe.py to record PyTorch
#                allocated/reserved peaks per rank; 0 = launch the trainer directly
# Extra arguments are passed to the trainer, e.g. `--train.optimizer muon` for plain Muon.
set -euo pipefail
NAME=${1:?usage: train.sh <run_name> [--key value ...]}; shift
: "${MODEL_ROOT:?set MODEL_ROOT (see download_models.sh)}"
: "${DATA_LIST:?set DATA_LIST (see prepare_robotwin_clean.sh)}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${PY:-python}
NPROC=${NPROC:-8} MICRO=${MICRO:-16} GA=${GA:-2}
GLOBAL=$((MICRO * GA * NPROC))
MAX_STEPS=${MAX_STEPS:-5000} SAVE_STEPS=${SAVE_STEPS:-2500} TIMING_ONLY=${TIMING_ONLY:-0}
MEM_PROBE=${MEM_PROBE:-1}
OUT_ROOT=${OUT_ROOT:-$REPO/output}
OUT=$OUT_ROOT/$NAME
mkdir -p "$OUT"
MODEL_ROOT=$(cd "$MODEL_ROOT" && pwd) DATA_LIST=$(cd "$(dirname "$DATA_LIST")" && pwd)/$(basename "$DATA_LIST")

# Fill the placeholder paths of the official config; everything else is copied verbatim.
"$PY" - "$REPO/configs/vla/robotwin/robotwin_dist_muon.yaml" "$OUT/config.yaml" <<EOF
import re, sys
src, dst = sys.argv[1:]
paths = {
    "model_path": "$MODEL_ROOT/lingbot-vla-v2-6b",
    "tokenizer_path": "$MODEL_ROOT/Qwen3-VL-4B-Instruct",
    "train_path": "$DATA_LIST",
    "output_dir": "$OUT",
    "moge_path": "$MODEL_ROOT/moge-2-vitb-normal/model.pt",
    "morgbd_path": "$MODEL_ROOT/lingbot-vla-v2-6b/depth/model.pt",
    "ckpt_path": "$MODEL_ROOT/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth",
    "config_path": "$MODEL_ROOT/lingbot-vla-v2-6b/dino_video/config.yaml",
}
text = open(src).read()
for key, value in paths.items():
    text, n = re.subn(rf"^(\s*{key}:)\s*\S+", rf"\g<1> {value}", text, flags=re.M)
    assert n == 1, f"{key}: expected one entry, found {n}"
open(dst, "w").write(text)
EOF

cd "$REPO"
ENTRY=(tasks/vla/train_lingbotvla.py)
if [ "$MEM_PROBE" = 1 ]; then
  ENTRY=(experiment/single_node_repro/mem_probe.py "${ENTRY[@]}")
  export MEM_PROBE_DIR=$OUT
fi
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Per-second GPU memory, utilization, temperature and thermal throttling; summarize_run.py
# reads it for peak memory and to flag a GPU that throttled (one slow GPU slows every step).
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown \
  --format=csv,noheader,nounits -l 1 > "$OUT/smi.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
"$PY" -m torch.distributed.run --standalone --nproc-per-node "$NPROC" "${ENTRY[@]}" \
  "$OUT/config.yaml" \
  --train.micro_batch_size "$MICRO" --train.global_batch_size "$GLOBAL" \
  --train.gradient_accumulation_steps "$GA" \
  --train.max_steps "$MAX_STEPS" --train.save_steps "$SAVE_STEPS" \
  "$@" > "$OUT/train.log" 2>&1 &
TR=$!
echo "training (pid $TR), log: $OUT/train.log"
if [ "$TIMING_ONLY" = 1 ]; then
  until grep -q "Step $MAX_STEPS/" "$OUT/train.log" 2>/dev/null || ! kill -0 $TR 2>/dev/null; do sleep 5; done
  sleep 3
  pkill -TERM -P $TR 2>/dev/null || true; kill -TERM $TR 2>/dev/null || true
fi
wait $TR || true
"$PY" "$REPO/experiment/single_node_repro/summarize_run.py" "$OUT" || true
