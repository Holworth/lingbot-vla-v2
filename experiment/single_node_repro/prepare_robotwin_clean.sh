#!/usr/bin/env bash
# Build the RoboTwin 2.0 clean training set (50 tasks x 50 episodes, LeRobot v3.0)
# and the dataset list that train.sh passes as --data.train_path.
#   DATA_ROOT=/data/robotwin2 bash experiment/single_node_repro/prepare_robotwin_clean.sh
# Needs: git, unzip, uv (`pip install uv`), the huggingface_hub CLI (`hf`), ~60 GB free.
# Every input is pinned, so two people running this get the same frames.
set -Eeuo pipefail
: "${DATA_ROOT:?set DATA_ROOT to the directory that will hold raw and converted data}"
ROBOTWIN_DIR=${ROBOTWIN_DIR:-$DATA_ROOT/RoboTwin}
JOBS=${JOBS:-16}
HF=${HF:-hf}

HF_DATASET_REV=a967b852afa21a9cbf19a198f7e653109042e87c   # TianxingChen/RoboTwin2.0
ROBOTWIN_COMMIT=266f3aadf505a4f7fe9af0faa41a20f5f47cd123  # RoboTwin-Platform/RoboTwin
XPOLICYLAB_COMMIT=c37109c500be67d0dea6b36bf7337bbd26e763cd # its XPolicyLab submodule
# XPolicyLab's LeRobot v3.0 converter targets an older LeRobot; this public patch ports it
# to LeRobot 0.6.0 and switches to CPU H.264 encoding. Fetched at a fixed commit and checked.
PATCH_URL=https://raw.githubusercontent.com/ZiguanWang/Robotwin-radeon-cloud/27b2f6fcb6c715e2cbb6db4431381f87c94b86ed/docker/patches/xpolicylab-lerobot-v30.patch
PATCH_SHA256=e6436cfb2e8f696b6ac7deb2caa8768c0b57b431c146cd9e7da25b03b6470f67

mkdir -p "$DATA_ROOT"/{raw,demo_clean,lerobot,convert_logs}

echo "[1/5] download demo_clean zips"
$HF download TianxingChen/RoboTwin2.0 --repo-type dataset --revision "$HF_DATASET_REV" \
  --include "dataset/*/demo_clean.zip" --local-dir "$DATA_ROOT/raw"

echo "[2/5] unzip"
ls "$DATA_ROOT/raw/dataset" | xargs -P "$JOBS" -I{} unzip -qo "$DATA_ROOT/raw/dataset/{}/demo_clean.zip" -d "$DATA_ROOT/demo_clean"
n=$(ls "$DATA_ROOT/demo_clean" | wc -l); [ "$n" = 50 ] || { echo "expected 50 tasks, got $n" >&2; exit 1; }

echo "[3/5] converter env (LeRobot 0.6.0)"
VENV=$DATA_ROOT/lerobot-env
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.12 "$VENV"
uv pip install -q --python "$VENV/bin/python" "lerobot[dataset]==0.6.0" "h5py==3.14.0" pyyaml

echo "[4/5] RoboTwin + patched XPolicyLab converter"
if [ ! -d "$ROBOTWIN_DIR/.git" ]; then
  git clone -q https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_DIR"
fi
git -C "$ROBOTWIN_DIR" checkout -q "$ROBOTWIN_COMMIT"
# The converter does not need RoboTwin's LFS assets, so skip downloading them.
git -C "$ROBOTWIN_DIR" -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false \
  submodule update -q --init --recursive
test "$(git -C "$ROBOTWIN_DIR/XPolicyLab" rev-parse HEAD)" = "$XPOLICYLAB_COMMIT"
if git -C "$ROBOTWIN_DIR/XPolicyLab" diff --quiet; then
  curl -fsSL "$PATCH_URL" -o "$DATA_ROOT/xpolicylab-lerobot-v30.patch"
  echo "$PATCH_SHA256  $DATA_ROOT/xpolicylab-lerobot-v30.patch" | sha256sum -c -
  git -C "$ROBOTWIN_DIR/XPolicyLab" apply "$DATA_ROOT/xpolicylab-lerobot-v30.patch"
fi
ln -sfn "$DATA_ROOT/demo_clean" "$ROBOTWIN_DIR/data/demo_clean"

echo "[5/5] convert to LeRobot v3.0 (joint actions, 50 episodes per task)"
export HF_LEROBOT_HOME=$DATA_ROOT/lerobot LEROBOT_VIDEO_BACKEND=pyav LEROBOT_VIDEO_CODEC=h264
convert_task() {
  local t=$1 log=$DATA_ROOT/convert_logs/$1.log
  rm -rf "$HF_LEROBOT_HOME/${t}_joint_v30"
  if "$VENV/bin/python" XPolicyLab/scripts/transform_lerobot_v30_format.py "demo_clean.${t}.aloha_agilex" \
       --repo_id "${t}_joint_v30" --max_episode 50 > "$log" 2>&1 \
     && ! grep -q "^Failed files:" "$log" && test -f "$HF_LEROBOT_HOME/${t}_joint_v30/meta/info.json"; then
    echo "[converted] $t"
  else
    echo "[failed] $t (see $log)"; return 1
  fi
}
export -f convert_task; export DATA_ROOT VENV HF_LEROBOT_HOME LEROBOT_VIDEO_BACKEND LEROBOT_VIDEO_CODEC
cd "$ROBOTWIN_DIR"
ls data/demo_clean/ | sort | xargs -P "$JOBS" -n 1 bash -c 'convert_task "$1"' _

LIST=$DATA_ROOT/robotwin_demo_clean_joint_v30.txt
: > "$LIST"
for t in $(ls data/demo_clean/ | sort); do echo "robotwin $HF_LEROBOT_HOME/${t}_joint_v30" >> "$LIST"; done
echo "done: $(wc -l < "$LIST") datasets listed in $LIST"
