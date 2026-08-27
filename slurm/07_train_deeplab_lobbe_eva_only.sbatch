#!/usr/bin/env bash
#SBATCH --job-name=dl-eva-only
#SBATCH --output=slurm_logs/deeplab_lobbe_eva_only_%j.out
#SBATCH --error=slurm_logs/deeplab_lobbe_eva_only_%j.err
#SBATCH --partition=major
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e

echo "Host:"
hostname
date

echo "GPU:"
nvidia-smi || true

PROJECT_ROOT="/home/wiss/shubhangi/da-seg/thesis_da_segmentation/sam_clip_full"
RUN_ROOT="$PROJECT_ROOT/camera_ready_runs"

cd "$PROJECT_ROOT"
source ../venv/bin/activate

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export MPLCONFIGDIR="$RUN_ROOT/cache/matplotlib"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" slurm_logs

echo "Python:"
which python
python --version

echo "Torch/CUDA:"
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY

echo "Starting DeepLab EVA-only adaptation..."
python scripts/train_deeplab_lobbe_eva_only.py --epochs 20 --bs 4 --workers 4

echo "Done. Output:"
ls -lh camera_ready_runs/segmentation/deeplab_lobbe_eva_only
