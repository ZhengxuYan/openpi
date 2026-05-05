#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --job-name=droid-pen-cup-knn

set -euo pipefail

echo "SLURM_JOBID=${SLURM_JOBID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "SLURM_TMPDIR=${SLURM_TMPDIR:-/tmp}"
echo "working directory=${SLURM_SUBMIT_DIR}"

unset DISPLAY
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TF_CPP_MIN_LOG_LEVEL=2
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

cd "${SLURM_SUBMIT_DIR}"

export PATH="/iliad2/u/${USER}/.local/bin:${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/iliad2/u/${USER}/.cache/uv}"

if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-openpi-rlds}"
elif [[ -f "/iris/u/${USER}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "/iris/u/${USER}/miniforge3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-openpi-rlds}"
elif [[ -f "/iliad/u/${USER}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "/iliad/u/${USER}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-openpi-rlds}"
fi

if ! command -v uv >/dev/null 2>&1; then
  UV_BOOTSTRAP="/iliad2/u/${USER}/venvs/uv-bootstrap"
  if [[ ! -x "${UV_BOOTSTRAP}/bin/uv" ]]; then
    echo "uv not found; bootstrapping uv into ${UV_BOOTSTRAP}"
    python3 -m venv "${UV_BOOTSTRAP}"
    "${UV_BOOTSTRAP}/bin/python" -m pip install --upgrade pip
    "${UV_BOOTSTRAP}/bin/python" -m pip install uv
  fi
  export PATH="${UV_BOOTSTRAP}/bin:${PATH}"
fi

uv --version
echo "UV_CACHE_DIR=${UV_CACHE_DIR}"

OUT_DIR="${OUT_DIR:-/iliad2/u/${USER}/droid_pen_in_cup_knn}"
WORK_DIR="${SLURM_TMPDIR:-/tmp}/${USER}/droid_pen_in_cup_knn_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}" "${WORK_DIR}"

ARGS=(
  --rlds-data-dir "${RLDS_DATA_DIR:-/iliad/group/datasets/droid}"
  --raw-data-dir "${RAW_DATA_DIR:-/iliad/group/datasets/droid_raw}"
  --output-dir "${OUT_DIR}"
  --work-dir "${WORK_DIR}"
  --batch-size "${BATCH_SIZE:-4}"
  --k "${K:-10}"
)

if [[ -n "${MAX_FRAMES:-}" ]]; then
  ARGS+=(--max-frames "${MAX_FRAMES}")
fi

uv run --no-sync --group rlds python scripts/droid_pen_in_cup_knn.py "${ARGS[@]}"
