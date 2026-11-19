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
# BH-Vision eval v3/v4 — PEFT adapter + checkpointi (sve kao --adapter uz isti base).
# Upotreba: sbatch bh-vision-eval-v3.sbatch <job-dir> [adapter-subdir] [max-samples] [eval-jsonl]
# Eval default v03-val; held-out: ... out/adapter-smo2b-v3 data/heldout-v01/eval.jsonl
set -euo pipefail
JOBDIR="${1:?upotreba: sbatch bh-vision-eval-v3.sbatch <job-dir> [adapter] [max-samples] [eval]}"
ADPART="${2:-out/adapter-smo2b-v1}"
AD="/job/$ADPART"
MAXN="${3:-0}"
EVALJSON="${4:-data/synth3/val.jsonl}"
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
      --base HuggingFaceTB/SmolVLM2-2.2B-Instruct --eval "/job/$EVALJSON" \
      --max-samples "$MAXN" --out "/job/out/$2" || echo "EVAL-FAIL $1"
}
PFX=$(basename "$ADPART")
run_eval "" "eval-$PFX-final.json"
for ck in 30 60 90 120; do run_eval "checkpoint-$ck" "eval-$PFX-ckpt$ck.json"; done
echo EVAL_DONE
