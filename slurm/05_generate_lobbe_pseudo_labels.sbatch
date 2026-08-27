#!/usr/bin/env bash
#SBATCH --job-name=lobbe-pseudo
#SBATCH --output=slurm_logs/lobbe_pseudo_%j.out
#SBATCH --partition=major
#SBATCH --nodelist=worker-4,worker-5,worker-6,worker-9
#SBATCH --error=slurm_logs/lobbe_pseudo_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -e

echo "Host:"
hostname
date

echo "GPU:"
nvidia-smi || true

PROJECT_ROOT="$HOME/da-seg/thesis_da_segmentation/sam_clip_full"
RUN_ROOT="$PROJECT_ROOT/camera_ready_runs"

cd "$PROJECT_ROOT"
source ../venv/bin/activate

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/..:$PYTHONPATH"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export MPLCONFIGDIR="$RUN_ROOT/cache/matplotlib"
export TRANSFORMERS_CACHE="$RUN_ROOT/cache/huggingface"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" slurm_logs

echo "Python:"
which python
python --version

echo "Torch/CUDA:"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

echo "Check EVA checkpoint:"
ls -lh "$RUN_ROOT/checkpoints/eva_clip_ft_iosb_packform9_FULL.pt"

echo "Starting Lobbe pseudo-label generation..."
python scripts/generate_lobbe_pseudo_labels.py \
  --max-images 500 \
  --sam-points-per-side 16 \
  --sam-max-side 640 \
  --sam-max-masks 60 \
  --min-mask-area 800 \
  --max-blip-per-image 6 \
  --blip-device cpu
echo "Done. Outputs:"
find "$RUN_ROOT/pseudo_labels" -maxdepth 3 -type d | sort
find "$RUN_ROOT/pseudo_labels/lobbe_eva_only/lbls" -maxdepth 1 -type f | wc -l
find "$RUN_ROOT/pseudo_labels/lobbe_eva_blip/lbls" -maxdepth 1 -type f | wc -l
ls -lh "$RUN_ROOT/pseudo_labels/lobbe_pseudo_generation_summary.json"
