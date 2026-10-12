#!/usr/bin/env bash
# Download the pretrained checkpoint, the Qwen3-VL tokenizer/backbone and the MoGe-2 teacher.
#   MODEL_ROOT=/data/lingbot bash experiment/single_node_repro/download_models.sh
# Layout afterwards (train.sh expects exactly this):
#   $MODEL_ROOT/lingbot-vla-v2-6b/            (also holds depth/model.pt and dino_video/)
#   $MODEL_ROOT/Qwen3-VL-4B-Instruct/
#   $MODEL_ROOT/moge-2-vitb-normal/model.pt
set -euo pipefail
: "${MODEL_ROOT:?set MODEL_ROOT to the directory that will hold the weights}"
mkdir -p "$MODEL_ROOT"
HF=${HF:-hf}  # huggingface_hub CLI (`pip install -U huggingface_hub`); may be a command with arguments
$HF download robbyant/lingbot-vla-v2-6b --local-dir "$MODEL_ROOT/lingbot-vla-v2-6b"
$HF download Qwen/Qwen3-VL-4B-Instruct --local-dir "$MODEL_ROOT/Qwen3-VL-4B-Instruct"
$HF download Ruicheng/moge-2-vitb-normal --local-dir "$MODEL_ROOT/moge-2-vitb-normal"
for f in lingbot-vla-v2-6b/depth/model.pt lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth \
         lingbot-vla-v2-6b/dino_video/config.yaml moge-2-vitb-normal/model.pt; do
  test -f "$MODEL_ROOT/$f" || { echo "missing $MODEL_ROOT/$f" >&2; exit 1; }
done
echo "models ready under $MODEL_ROOT"
