#!/usr/bin/env bash
#SBATCH --job-name=eva-iosb
#SBATCH --output=slurm_logs/eva_iosb_%j.out
#SBATCH --error=slurm_logs/eva_iosb_%j.err
#SBATCH --partition=major
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -e

echo "Host:"
hostname
date

echo "GPU:"
nvidia-smi || true
echo "GPU:"
nvidia-smi || true

echo "CUDA_VISIBLE_DEVICES:"
echo $CUDA_VISIBLE_DEVICES

echo "Torch:"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
PROJECT_ROOT="$HOME/da-seg/thesis_da_segmentation/sam_clip_full"
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

echo "Torch:"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

echo "Starting EVA-CLIP IOSB training..."
python scripts/train_evaclip_iosb.py --epochs 8 --no-rebuild-csv

echo "Done. Checkpoints:"
ls -lh "$RUN_ROOT/checkpoints"
