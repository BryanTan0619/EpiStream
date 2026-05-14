#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

RUN_TAG="${RUN_TAG:-memory_schema_vlm_v1}"

DATASET_DIR="${DATASET_DIR:-dataset_v1/output_${RUN_TAG}}"
UTILITY_DIR="${UTILITY_DIR:-cache/utility_${RUN_TAG}}"
PREPARED_CACHE_DIR="${PREPARED_CACHE_DIR:-cache/vlm_method2_regression_${RUN_TAG}}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-output/vlm_method2_regression_adv_${RUN_TAG}}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${MODEL_OUTPUT_DIR}/tensorboard}"

MEMORY_SOURCE="${MEMORY_SOURCE:-vlm}"
INTERVAL_DENSE="${INTERVAL_DENSE:-1.0}"
INTERVAL_SPARSE="${INTERVAL_SPARSE:-3.0}"
BOUNDARY_WINDOW="${BOUNDARY_WINDOW:-6.0}"
NUM_FRAMES="${NUM_FRAMES:-8}"

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-2e-5}"
THRESHOLD="${THRESHOLD:--0.3}"
SAVE_STEPS="${SAVE_STEPS:-3000}"
BOUNDARY_LOSS_WEIGHT="${BOUNDARY_LOSS_WEIGHT:-0.5}"
AUX_ADVANTAGE_THRESHOLD="${AUX_ADVANTAGE_THRESHOLD:--0.05}"
PEAK_MSE_WEIGHT="${PEAK_MSE_WEIGHT:-1.0}"
PEAK_MSE_ADVANTAGE_THRESHOLD="${PEAK_MSE_ADVANTAGE_THRESHOLD:--0.05}"
USE_AUX_POS_WEIGHT="${USE_AUX_POS_WEIGHT:-0}"

SKIP_DATASET="${SKIP_DATASET:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-1}"

export TLOG_BASE_DIR="${TLOG_BASE_DIR:-/path/to/tlog_files}"
export FRAME_BASE_DIR="${FRAME_BASE_DIR:-/path/to/frame_screenshots}"
export MEMORY_CACHE_PATH="${MEMORY_CACHE_PATH:-${PROJECT_DIR}/cache/memory/GT-Original.json}"

echo "======================================================================"
echo "GT-Original + VLM Memory Advantage SFT Pipeline"
echo "======================================================================"
echo "RUN_TAG             = ${RUN_TAG}"
echo "DATASET_DIR         = ${DATASET_DIR}"
echo "UTILITY_DIR         = ${UTILITY_DIR}"
echo "PREPARED_CACHE_DIR  = ${PREPARED_CACHE_DIR}"
echo "MODEL_OUTPUT_DIR    = ${MODEL_OUTPUT_DIR}"
echo "TENSORBOARD_LOG_DIR = ${TENSORBOARD_LOG_DIR}"
echo "MEMORY_CACHE_PATH   = ${MEMORY_CACHE_PATH}"
echo "TLOG_BASE_DIR       = ${TLOG_BASE_DIR}"
echo "FRAME_BASE_DIR      = ${FRAME_BASE_DIR}"
echo "MEMORY_SOURCE       = ${MEMORY_SOURCE}"
echo "SAMPLING            = dense ${INTERVAL_DENSE}s, sparse ${INTERVAL_SPARSE}s, boundary ±${BOUNDARY_WINDOW}s"
echo "TRAINING            = device ${DEVICE}, epochs ${EPOCHS}, batch ${BATCH_SIZE}, lr ${LR}, threshold ${THRESHOLD}, boundary_loss_weight ${BOUNDARY_LOSS_WEIGHT}"
echo "AUX TARGET          = advantage >= ${AUX_ADVANTAGE_THRESHOLD}, peak_mse_weight ${PEAK_MSE_WEIGHT} @ advantage >= ${PEAK_MSE_ADVANTAGE_THRESHOLD}, aux_pos_weight ${USE_AUX_POS_WEIGHT}"
echo "======================================================================"

if [[ "${FORCE_REBUILD}" == "1" ]]; then
  echo "[force] Removing stale utility/prepared cache files for this RUN_TAG..."
  rm -f "${UTILITY_DIR}/train_with_utility.jsonl" \
        "${UTILITY_DIR}/val_with_utility.jsonl" \
        "${UTILITY_DIR}/test_with_utility.jsonl" \
        "${PREPARED_CACHE_DIR}/train_prepared.jsonl" \
        "${PREPARED_CACHE_DIR}/val_prepared.jsonl"
fi

if [[ "${SKIP_DATASET}" != "1" ]]; then
  echo
  echo "[1/4] Constructing dataset from GT-Original windows and cached ${MEMORY_SOURCE} memory cards..."
  python3 dataset/dataset_construction.py \
    --memory_source "${MEMORY_SOURCE}" \
    --interval_dense "${INTERVAL_DENSE}" \
    --interval_sparse "${INTERVAL_SPARSE}" \
    --boundary_window "${BOUNDARY_WINDOW}" \
    --num_frames "${NUM_FRAMES}" \
    --output_dir "${DATASET_DIR}"
else
  echo
  echo "[1/4] SKIP_DATASET=1, reusing existing dataset at ${DATASET_DIR}"
fi

echo
echo "[2/4] Computing forward-looking utility with GT-Original windows..."
python3 src/compute_utility_forward_looking.py \
  --train_jsonl "${DATASET_DIR}/train.jsonl" \
  --val_jsonl "${DATASET_DIR}/val.jsonl" \
  --test_jsonl "${DATASET_DIR}/test.jsonl" \
  --output_dir "${UTILITY_DIR}"

echo
echo "[3/4] Preparing advantage regression cache..."
python3 src/sft_train.py \
  --mode precompute \
  --cache_dir "${PREPARED_CACHE_DIR}" \
  --train_cache "${UTILITY_DIR}/train_with_utility.jsonl" \
  --val_cache "${UTILITY_DIR}/val_with_utility.jsonl" \
  --threshold "${THRESHOLD}"

echo
echo "[4/4] Training advantage regression model..."
python3 src/sft_train.py \
  --mode train \
  --cache_dir "${PREPARED_CACHE_DIR}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --threshold "${THRESHOLD}" \
  --boundary_loss_weight "${BOUNDARY_LOSS_WEIGHT}" \
  --aux_advantage_threshold "${AUX_ADVANTAGE_THRESHOLD}" \
  --peak_mse_weight "${PEAK_MSE_WEIGHT}" \
  --peak_mse_advantage_threshold "${PEAK_MSE_ADVANTAGE_THRESHOLD}" \
  --use_aux_pos_weight "${USE_AUX_POS_WEIGHT}" \
  --tensorboard_log_dir "${TENSORBOARD_LOG_DIR}" \
  --output_dir "${MODEL_OUTPUT_DIR}" \
  --save_steps "${SAVE_STEPS}"

echo
echo "======================================================================"
echo "Pipeline complete"
echo "Dataset:       ${DATASET_DIR}"
echo "Utility cache: ${UTILITY_DIR}"
echo "Prepared:      ${PREPARED_CACHE_DIR}"
echo "Model output:  ${MODEL_OUTPUT_DIR}"
echo "TensorBoard:   ${TENSORBOARD_LOG_DIR}"
echo "======================================================================"
