#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"


# ---- ENV ----
: "${OUT:?Set OUT before starting (export OUT=...)}"
:
VENV_PY="${VENV_PY:-/home/scs_deal_projects_notapebackup/user/shubhang/thesis/.venv/bin/python}"
[ -x "$VENV_PY" ] || VENV_PY="$(command -v python3 || command -v python)"
echo "[chain] using python: $VENV_PY"

export PYTHONPATH="/home/scs_deal_projects_notapebackup/user/shubhang/thesis:${PYTHONPATH:-}"

NUM_CLASSES=81
TRAIN_IMGS="$OUT/mini/train_imgs"
TRAIN_LBLS="$OUT/mini/train_lbls"
VAL_IMGS="$OUT/mini/val_imgs"
VAL_LBLS="$OUT/mini/val_lbls"
BS="${BS:-2}"    # add near top


# optional class weights (silently skip if missing)
CEW="$OUT/weights/ce_weights_mini.npy"
[ -f "$CEW" ] || CEW="$OUT/weights/ce_weights_miniq.npy"
EXTRA_CEW=()
[ -f "$CEW" ] && EXTRA_CEW=(--class_weights "$CEW")

LOGDIR="$OUT/seg_runs"
CHAIN_LOG="$LOGDIR/chain.log"
SUMMARY="$LOGDIR/summary.tsv"
mkdir -p "$LOGDIR"

# header once
[ -f "$SUMMARY" ] || echo -e "model\trun\tbest_epoch\tmIoU\tmDice\tpixAcc" > "$SUMMARY"

run_job () {
  local ARCH=$1 EPOCHS=$2 SIZE=$3 LR=$4 OUTDIR=$5 AUX=$6 INIT=${7:-}

  echo "[$(date)] START $ARCH | ${EPOCHS}ep | ${SIZE}px -> $OUTDIR" | tee -a "$CHAIN_LOG"
  mkdir -p "$OUTDIR"

  # Build args array to avoid word-splitting bugs
  ARGS=(
    -u -m sam_clip_full.seg.seg_main train
    --train_imgs "$TRAIN_IMGS" --train_lbls "$TRAIN_LBLS"
    --val_imgs   "$VAL_IMGS"   --val_lbls   "$VAL_LBLS"
    --num_classes "$NUM_CLASSES" --arch "$ARCH"
    --epochs "$EPOCHS" --bs "$BS" --size "$SIZE" --lr "$LR" --wd 1e-2
    --workers 8 --device cuda --out_dir "$OUTDIR"
  )
  # aux head?
  [ "$AUX" = "yes" ] && ARGS+=(--aux_loss)
  # class weights?
  ARGS+=("${EXTRA_CEW[@]}")
  # init checkpoint?
  [ -n "${INIT}" ] && ARGS+=(--init_ckpt "$INIT")

  # Run
  "$VENV_PY" "${ARGS[@]}" > "$OUTDIR/train.log" 2>&1 || {
    echo "[$(date)] $ARCH FAILED" | tee -a "$CHAIN_LOG"
  }

  # Append a one-line summary if metrics exist
  if [ -f "$OUTDIR/val_metrics.json" ]; then
    "$VENV_PY" - "$OUTDIR" "$ARCH" >> "$SUMMARY" <<'PY'
import json, os, sys
d, arch = sys.argv[1], sys.argv[2]
m=json.load(open(os.path.join(d,'val_metrics.json')))
print(f"{arch}\t{os.path.basename(d)}\t{m.get('epoch')}\t{m.get('miou',0):.4f}\t{m.get('mdice',0):.4f}\t{m.get('pixel_accuracy',0):.4f}")
PY
  fi

  echo "[$(date)] END $ARCH -> $OUTDIR" | tee -a "$CHAIN_LOG"
}

# ---- Job 1: DeepLabV3-ResNet50, warm-start 50ep @512 (your strong baseline) ----
INIT="$OUT/seg_runs/deeplabv3_resnet50_full_2025-11-10_18-13/best.pth"
J1="$OUT/seg_runs/dlv3r50_ft512_ep50_$(date +%F_%H-%M)"
run_job deeplabv3_resnet50 50 512 1e-4 "$J1" yes "$INIT"

# ---- Job 2: FCN-ResNet50 ImageNet, 16ep @512 ----
J2="$OUT/seg_runs/fcnr50_imnet512_ep16_$(date +%F_%H-%M)"
run_job fcn_resnet50 16 512 3e-4 "$J2" yes

# ---- Job 3 (optional): DeepLabV3-ResNet101 ImageNet, 24ep @512 ----
# J3="$OUT/seg_runs/dlv3r101_imnet512_ep24_$(date +%F_%H-%M)"
# run_job deeplabv3_resnet101 24 512 1e-4 "$J3" yes
