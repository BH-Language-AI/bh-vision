#!/bin/bash
#SBATCH --job-name=bh-vision-eval-base
#SBATCH --account=project_465003647
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=bh-vision-eval-base-%j.out
# BH-Vision eval — adapter vs val set, rezultat u eval.json. Brz posao (10-30 min).
# Upotreba: sbatch bh-vision-eval-base.sbatch <job-dir> [max-samples, default 0=svi]
set -euo pipefail
JOBDIR="${1:?upotreba: sbatch bh-vision-eval-base.sbatch <job-dir> [max-samples]}"
MAXN="${2:-0}"
module purge
module load Local-LAIF lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
export PYLIBS=/scratch/project_465003647/bh-vision/pylibs
export HF_HUB_CACHE=/scratch/project_465003647/bh-vision/hfcache
export HF_DATASETS_CACHE=/scratch/project_465003647/bh-vision/hfds
export TMPDIR=/scratch/project_465003647/bh-vision/tmp
export SINGULARITYENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export SINGULARITYENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export SINGULARITYENV_TMPDIR="$TMPDIR"
export SINGULARITYENV_PYTHONPATH="$PYLIBS"
cd "$JOBDIR"
srun --unbuffered singularity exec -B "$JOBDIR:/job" -B /scratch:/scratch "$SIF" \
  python -u /job/training/eval.py 
    --base HuggingFaceTB/SmolVLM2-2.2B-Instruct --eval /job/data/synth3/val.jsonl \
    --max-samples "$MAXN" --out /job/out/eval-base-2b-v1.json
echo EVAL_DONE
