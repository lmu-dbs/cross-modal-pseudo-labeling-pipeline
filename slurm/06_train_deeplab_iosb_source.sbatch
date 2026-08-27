#!/usr/bin/env bash
#SBATCH --job-name=dl-iosb-src
#SBATCH --output=slurm_logs/deeplab_iosb_source_%j.out
#SBATCH --error=slurm_logs/deeplab_iosb_source_%j.err
#SBATCH --partition=major
#SBATCH --nodelist=worker-4,worker-5,worker-6,worker-9
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e

echo "Host:"
hostname
date
nvidia-smi || true

PROJECT_ROOT="$HOME/da-seg/thesis_da_segmentation/sam_clip_full"
BASE="/home/wiss/shubhangi/home/scs_deal_projects_notapebackup"
DATA_ROOT="$BASE/shared/DATASET/waste_dataset"

IOSB="$DATA_ROOT/iosb_packform9"

IOSB_TRAIN_IMGS="$IOSB/train_imgs"
IOSB_TRAIN_LBLS="$IOSB/train_lbls"
IOSB_VAL_IMGS="$IOSB/val_imgs"
IOSB_VAL_LBLS="$IOSB/val_lbls"

RUN_ROOT="$PROJECT_ROOT/camera_ready_runs"
SEG_ROOT="$RUN_ROOT/segmentation"

mkdir -p "$SEG_ROOT" slurm_logs

cd "$PROJECT_ROOT"
source ../venv/bin/activate

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export MPLCONFIGDIR="$RUN_ROOT/cache/matplotlib"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

echo "Python:"
which python
python --version

echo "Torch:"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

echo "Training source-only DeepLab on IOSB GT..."

python -m seg.seg_main train \
  --train_imgs "$IOSB_TRAIN_IMGS" \
  --train_lbls "$IOSB_TRAIN_LBLS" \
  --val_imgs "$IOSB_VAL_IMGS" \
  --val_lbls "$IOSB_VAL_LBLS" \
  --num_classes 9 \
  --ignore_index 255 \
  --epochs 50 \
  --bs 4 \
  --workers 4 \
  --crop_size 512 \
  --lr 1e-4 \
  --wd 1e-2 \
  --amp \
  --poly_lr \
  --out_dir "$SEG_ROOT/deeplab_iosb_source" \
  --save_preds

echo "Done."
ls -lh "$SEG_ROOT/deeplab_iosb_source"
