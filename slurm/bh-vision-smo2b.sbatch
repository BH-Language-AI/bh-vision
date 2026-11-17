#!/bin/bash
#SBATCH --job-name=bh-vision-smo2b
#SBATCH --account=project_465003647
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=02:30:00
#SBATCH --output=bh-vision-smo2b-%j.out
# BH-Vision prvi trening — SmolVLM2-2.2B LoRA bf16 (ROCm, bez bitsandbytes).
# Upotreba (na LUMI login node): sbatch bh-vision-smo2b.sbatch <job-dir>
# (job-dir sadrzi training/, configs/, scripts/, data/synth2/ — prekopirati iz BH-Language-AI/bh-vision repoa)
set -euo pipefail
JOBDIR="${1:?upotreba: sbatch bh-vision-smo2b.sbatch <job-dir>}"
module purge
module load Local-LAIF lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
export PYLIBS=/scratch/project_465003647/agentmujo/pylibs
export HF_HUB_CACHE=/scratch/project_465003647/agentmujo/hfcache
export SINGULARITYENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export SINGULARITYENV_PYTHONPATH="$PYLIBS"
mkdir -p "$PYLIBS" "$HF_HUB_CACHE"
# Jednom po projektu (kešira se na scratchu): python zavisnosti u PYLIBS (bez torcha — koristi se kontejnerski ROCm torch; bez bitsandbytes — CUDA-only).
srun -n1 singularity exec "$SIF" env PYTHONPATH="$PYLIBS" pip install -q --target="$PYLIBS" \
  "transformers==5.22.2" "trl==1.18.0" "peft==0.21.2" datasets pillow huggingface-hub num2words accelerate 2>&1 | tail -1
srun -n1 singularity exec "$SIF" env PYTHONPATH="$PYLIBS" python -c \
  "import torch; assert 'rocm' in torch.__version__, torch.__version__; print('ROCM-TORCH-OK', torch.__version__)"
cd "$JOBDIR"
srun --unbuffered singularity exec -B "$JOBDIR:/job" -B /scratch:/scratch "$SIF" \
  python -u /job/training/train.py --config /job/configs/smo2b-qlora.yaml \
    --train /job/data/synth2/train.jsonl --out /job/out/adapter-smo2b-v1 --quant none
echo TRAIN_DONE
