#!/bin/bash
#SBATCH --job-name=bh-vision-smo2b
#SBATCH --account=project_465003647
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=02:30:00
#SBATCH --output=bh-vision-smo2b-%j.out
# BH-Vision prvi trening — SmolVLM2-2.2B LoRA bf16 (ROCm, bez bitsandbytes).
# Upotreba (na LUMI login node): sbatch bh-vision-smo2b.sbatch <job-dir>
# (job-dir sadrzi training/, configs/, scripts/, data/synth3/ — prekopirati iz BH-Language-AI/bh-vision repoa)
set -euo pipefail
JOBDIR="${1:?upotreba: sbatch bh-vision-smo2b.sbatch <job-dir> [config] [train] [out]}"
CONFIG="${2:-configs/smo2b-qlora.yaml}"
TRAIN="${3:-data/synth3/train.jsonl}"
OUTD="${4:-out/adapter-smo2b-v1}"
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
mkdir -p "$PYLIBS" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" "$TRITON_CACHE_DIR"
# Jednom po projektu (kešira se na scratchu): python zavisnosti u PYLIBS (bez torcha — koristi se kontejnerski ROCm torch; bez bitsandbytes — CUDA-only).
srun -n1 singularity exec "$SIF" env PYTHONPATH="$PYLIBS" pip install -q --target="$PYLIBS" \
  "transformers==5.22.2" "trl==1.18.0" "peft==0.21.2" datasets pillow huggingface-hub num2words accelerate 2>&1 | tail -1
srun -n1 singularity exec "$SIF" env PYTHONPATH="$PYLIBS" python -c \
  "import shutil, glob, os; [shutil.rmtree(p, ignore_errors=True) for g in ('torch*','nvidia*','triton*','cuda*','torchaudio*','torchvision*') for p in glob.glob(os.path.join('$PYLIBS', g))]; print('PYLIBS-CIST')"
srun -n1 singularity exec "$SIF" env PYTHONPATH="$PYLIBS" python -c \
  "import torch; assert 'rocm' in torch.__version__, torch.__version__; print('ROCM-TORCH-OK', torch.__version__)"
cd "$JOBDIR"
srun --unbuffered singularity exec -B "$JOBDIR:/job" -B /scratch:/scratch "$SIF" \
  python -u /job/training/train.py --config /job/$CONFIG \
    --train /job/$TRAIN --out /job/$OUTD --quant none
echo TRAIN_DONE
