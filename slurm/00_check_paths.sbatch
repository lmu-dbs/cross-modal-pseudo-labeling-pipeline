#!/usr/bin/env bash
#SBATCH --job-name=check-paths
#SBATCH --output=slurm_logs/check_paths_%j.out
#SBATCH --error=slurm_logs/check_paths_%j.err
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
set -e

echo "Host:"
hostname

echo "Date:"
date

PROJECT_ROOT="$HOME/da-seg/thesis_da_segmentation/sam_clip_full"
BASE="/home/wiss/shubhangi/home/scs_deal_projects_notapebackup"
DATA_ROOT="$BASE/shared/DATASET/waste_dataset"

IOSB="$DATA_ROOT/iosb_packform9"
LOBBE="$DATA_ROOT/lobbe_fix"

IOSB_TRAIN_IMGS="$IOSB/train_imgs"
IOSB_TRAIN_LBLS="$IOSB/train_lbls"
IOSB_VAL_IMGS="$IOSB/val_imgs"
IOSB_VAL_LBLS="$IOSB/val_lbls"

LOBBE_VAL_IMGS="$LOBBE/val_imgs"
LOBBE_VAL_GT="$LOBBE/val_lbls_form9"

for p in \
  "$PROJECT_ROOT" \
  "$IOSB_TRAIN_IMGS" "$IOSB_TRAIN_LBLS" \
  "$IOSB_VAL_IMGS" "$IOSB_VAL_LBLS" \
  "$LOBBE_VAL_IMGS" "$LOBBE_VAL_GT"
do
  echo ""
  echo "$p"
  if [ -d "$p" ]; then
    echo "OK"
    find "$p" -maxdepth 1 -type f | head -3
  else
    echo "MISSING"
  fi
done

echo ""
echo "Python check:"
cd "$PROJECT_ROOT"
source ../venv/bin/activate
which python
python --version
timeout 60s python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available());print('device_count', torch.cuda.device_count())"

