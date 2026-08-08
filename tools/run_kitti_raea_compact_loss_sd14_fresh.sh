#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON="${PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    printf 'error: no Python interpreter found; activate the training environment or set PYTHON\n' >&2
    exit 2
fi
DATA_ROOT="${DATA_ROOT:-/media/shizhm/sda2/CS2S_results}"
KITTI_ROOT="${KITTI_ROOT:-/media/shizhm/Lenovo/KITTI_RAW}"
CACHE_ROOT="${CACHE_ROOT:-/media/shizhm/sda2/CS2S_cache_memmap}"
RUN_ROOT="${RUN_ROOT:-${DATA_ROOT}/kitti_ray_posterior}"
RUN_NAME="${RUN_NAME:-ray_posterior_utonia_dino_sd14_fresh_$(date +%Y%m%d_%H%M%S)}"
TARGET_STEP="${TARGET_STEP:-500000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-2500}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
LR="${LR:-1e-5}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${ROOT}/dataset/kitti_raw_sat_lidar/train_manifest.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-${ROOT}/dataset/kitti_raw_sat_lidar/test2_manifest.jsonl}"
SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-${VAL_MANIFEST}}"
SD_BASE_CKPT="${SD_BASE_CKPT:-${HOME}/Downloads/sd-v1-4.ckpt}"
LIDAR_CACHE_ROOT="${LIDAR_CACHE_ROOT:-${CACHE_ROOT}/utonia_ray_depth_all_fp16}"
DINO_CACHE_ROOT="${DINO_CACHE_ROOT:-${CACHE_ROOT}/dino_vits14_8x32_all_fp16}"
RESUME_CKPT="${RESUME_CKPT:-}"

for value in NUM_GPUS BATCH_PER_GPU WORKERS_PER_GPU TARGET_STEP SAVE_EVERY SAMPLE_EVERY; do
    if ! [[ "${!value}" =~ ^[0-9]+$ ]]; then
        printf 'error: %s must be a non-negative integer, got %q\n' "${value}" "${!value}" >&2
        exit 2
    fi
done
if (( NUM_GPUS < 1 || BATCH_PER_GPU < 1 )); then
    printf 'error: NUM_GPUS and BATCH_PER_GPU must be at least 1\n' >&2
    exit 2
fi
if ! "${PYTHON}" -c 'import omegaconf, torch' >/dev/null 2>&1; then
    printf 'error: %s is not the ControlS2S training interpreter; activate the environment or set PYTHON\n' \
        "${PYTHON}" >&2
    exit 2
fi

required_files=(
    "${ROOT}/configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea.yaml"
    "${ROOT}/tools/train_kitti_raea.py"
    "${SD_BASE_CKPT}"
    "${TRAIN_MANIFEST}"
    "${VAL_MANIFEST}"
)
if (( SAMPLE_EVERY > 0 )); then
    required_files+=("${SAMPLE_MANIFEST}")
fi
if [[ -n "${RESUME_CKPT}" ]]; then
    required_files+=("${RESUME_CKPT}")
fi
for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        printf 'error: required file not found: %s\n' "${path}" >&2
        exit 2
    fi
done
for path in "${LIDAR_CACHE_ROOT}" "${DINO_CACHE_ROOT}"; do
    if [[ ! -d "${path}" ]]; then
        printf 'error: required cache directory not found: %s\n' "${path}" >&2
        exit 2
    fi
done
if [[ ! -d "${KITTI_ROOT}" ]]; then
    printf 'error: KITTI_ROOT directory not found: %s\n' "${KITTI_ROOT}" >&2
    exit 2
fi
mkdir -p "${RUN_ROOT}"

printf 'run=%s gpus=%s batch_per_gpu=%s global_batch=%s lr=%s\n' \
    "${RUN_NAME}" "${NUM_GPUS}" "${BATCH_PER_GPU}" "$((NUM_GPUS * BATCH_PER_GPU))" "${LR}"

if (( NUM_GPUS > 1 )); then
    LAUNCH=("${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node "${NUM_GPUS}")
else
    LAUNCH=("${PYTHON}" -u)
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CKPT}" ]]; then
    RESUME_ARGS=(--resume-ckpt "${RESUME_CKPT}")
fi

cd "${ROOT}"
exec env PYTHONUNBUFFERED=1 "${LAUNCH[@]}" tools/train_kitti_raea.py \
    --config configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea.yaml \
    --sd-base-ckpt "${SD_BASE_CKPT}" \
    --kitti-root "${KITTI_ROOT}" \
    "${RESUME_ARGS[@]}" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest "${VAL_MANIFEST}" \
    --out-root "${RUN_ROOT}" \
    --run-name "${RUN_NAME}" \
    --steps "${TARGET_STEP}" \
    --batch-size "${BATCH_PER_GPU}" \
    --num-workers "${WORKERS_PER_GPU}" \
    --lr "${LR}" \
    --shuffle \
    --amp \
    --log-every 20 \
    --save-every "${SAVE_EVERY}" \
    --sample-every "${SAMPLE_EVERY}" \
    --sample-manifest "${SAMPLE_MANIFEST}" \
    --sample-num-samples 2 \
    --sample-ddim-steps 50 \
    --sample-probes normal \
    --keep-step-checkpoints 3 \
    --lidar-ray-feature-cache-root "${LIDAR_CACHE_ROOT}" \
    --image-semantic-cache-root "${DINO_CACHE_ROOT}" \
    --lidar-support-loss-weight 1.0 \
    --lidar-support-dilation 8 \
    --lidar-depth-loss-weight 1.0 \
    --lidar-semantic-alignment-weight 0.2 \
    --lidar-token-structure-target-ratio 0.08 \
    --lidar-token-structure-loss-weight 0.01 \
    --lidar-reference-window 3
