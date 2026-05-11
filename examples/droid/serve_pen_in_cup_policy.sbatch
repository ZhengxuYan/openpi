#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=droid-pen-cup-policy
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

echo "SLURM_JOBID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-4}"
echo "working directory=$(pwd)"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/iliad2/u/$USER/.cache/openpi}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/iliad2/u/$USER/.cache/uv}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export PATH="/iliad2/u/$USER/.local/bin:$HOME/.local/bin:$PATH"

mkdir -p "$OPENPI_DATA_HOME" "$UV_CACHE_DIR"

if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
elif [[ -f "/iris/u/$USER/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "/iris/u/$USER/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "/iliad/u/$USER/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "/iliad/u/$USER/miniconda3/etc/profile.d/conda.sh"
fi

if [[ "${DRY_RUN:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV:-openpi-rlds}" || true
fi

if [[ "${DRY_RUN:-0}" != "1" ]] && ! command -v uv >/dev/null 2>&1; then
  UV_BOOTSTRAP_DIR="${UV_BOOTSTRAP_DIR:-/iliad2/u/$USER/venvs/uv-bootstrap}"
  python3 -m venv "$UV_BOOTSTRAP_DIR"
  "$UV_BOOTSTRAP_DIR/bin/python" -m pip install --upgrade pip uv
  export PATH="$UV_BOOTSTRAP_DIR/bin:$PATH"
fi

POLICY="${POLICY:-finetuned}"
PORT="${PORT:-8000}"

case "$POLICY" in
  zero_shot)
    POLICY_CONFIG="${POLICY_CONFIG:-pi0_fast_droid}"
    POLICY_DIR="${POLICY_DIR:-gs://openpi-assets/checkpoints/pi0_fast_droid}"
    ;;
  finetuned)
    POLICY_CONFIG="${POLICY_CONFIG:-pi0_fast_droid_pen_in_cup_finetune_validation_10k}"
    POLICY_DIR="${POLICY_DIR:-/scr/tiangao/openpi_checkpoints/pi0_fast_droid_pen_in_cup_finetune_validation_10k}"
    ;;
  *)
    echo "Unsupported POLICY=$POLICY. Use POLICY=zero_shot or POLICY=finetuned." >&2
    exit 2
    ;;
esac

if [[ "$POLICY_DIR" == /* && ! -e "$POLICY_DIR" ]]; then
  echo "Warning: local POLICY_DIR does not exist on this node: $POLICY_DIR" >&2
fi

if command -v uv >/dev/null 2>&1; then
  echo "uv=$(uv --version)"
else
  echo "uv=not found"
fi
echo "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"
echo "POLICY=$POLICY"
echo "POLICY_CONFIG=$POLICY_CONFIG"
echo "POLICY_DIR=$POLICY_DIR"
echo "PORT=$PORT"

COMMAND=(
  uv run scripts/serve_policy.py policy:checkpoint
  "--policy.config=$POLICY_CONFIG"
  "--policy.dir=$POLICY_DIR"
  "--port=$PORT"
)

echo "COMMAND=${COMMAND[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${COMMAND[@]}"
