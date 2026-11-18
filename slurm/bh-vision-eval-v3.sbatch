#!/bin/bash
#SBATCH --job-name=bh-vision-eval-v3
#SBATCH --account=project_465003647
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=bh-vision-eval-v3-%j.out
# BH-Vision eval v3 — PEFT adapter + checkpointi (sve kao --adapter uz isti base).
# Upotreba: sbatch bh-vision-eval-v3.sbatch <job-dir> [max-samples, default 0=svi]
set -euo pipefail
JOBDIR="${1:?upotreba: sbatch bh-vision-eval-v3.sbatch <job-dir> [max-samples]}"
MAXN="${2:-0}"
AD=/job/out/adapter-smo2b-v1
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
run_eval() { # $1=adapter-subdir ("" = final) $2=out-json
  # Napomena: bez -f provjere — /job postoji samo UNUTAR kontejnera, provjera na hostu uvijek padne.
  srun --unbuffered singularity exec -B "$JOBDIR:/job" -B /scratch:/scratch "$SIF" \
    python -u /job/training/eval.py --adapter "$AD/$1" \
      --base HuggingFaceTB/SmolVLM2-2.2B-Instruct --eval /job/data/synth3/val.jsonl \
      --max-samples "$MAXN" --out "/job/out/$2" || echo "EVAL-FAIL $1"
}
run_eval "" "eval-smo2b-v3-final.json"
run_eval "checkpoint-30" "eval-smo2b-v3-ckpt30.json"
run_eval "checkpoint-60" "eval-smo2b-v3-ckpt60.json"
echo EVAL_DONE
