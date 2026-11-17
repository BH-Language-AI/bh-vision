#!/bin/bash
#SBATCH --job-name=bh-vision-smo500m
#SBATCH --account=project_465003647
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=bh-vision-smo500m-%j.out
# BH-Vision prvi trening — SmolVLM2-2.2B LoRA bf16 (ROCm, bez bitsandbytes).
# Upotreba (na LUMI login node): sbatch bh-vision-smo500m.sbatch <job-dir>
# (job-dir sadrzi training/, configs/, scripts/, data/synth3/ — prekopirati iz BH-Language-AI/bh-vision repoa)
set -euo pipefail
JOBDIR="${1:?upotreba: sbatch bh-vision-smo500m.sbatch <job-dir>}"
module purge
module load Local-LAIF lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
export PYLIBS=/scratch/project_465003647/bh-vision/pylibs
export HF_HUB_CACHE=/scratch/project_465003647/bh-vision/hfcache
export HF_DATASETS_CACHE=/scratch/project_465003647/bh-vision/hfds
export TMPDIR=/scratch/project_465003647/bh-vision/tmp
export TRITON_CACHE_DIR=/scratch/project_465003647/bh-vision/triton
unset TRITON_HIP_ARCH_OVERRIDE
export SINGULARITYENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export SINGULARITYENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export SINGULARITYENV_PYTHONPATH="$PYLIBS"
if [ ! -d "$PYLIBS/transformers" ]; then
  echo "PYLIBS prazan — neocekivano (smo2b job ga je trebao napuniti)"; exit 1
fi
srun -n1 singularity exec "$SIF" env PYTHONPATH="$PYLIBS" python -c \
  "import torch; assert 'rocm' in torch.__version__, torch.__version__; print('ROCM-TORCH-OK', torch.__version__)"
cd "$JOBDIR"
srun --unbuffered singularity exec -B "$JOBDIR:/job" -B /scratch:/scratch "$SIF" \
  python -u /job/training/train.py --config /job/configs/smo500m-qlora.yaml \
    --train /job/data/synth3/train.jsonl --out /job/out/adapter-smo500m-v1 --quant none
echo TRAIN_DONE
