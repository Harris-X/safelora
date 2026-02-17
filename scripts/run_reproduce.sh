#!/usr/bin/env bash
set -euo pipefail

# =========================
# Default config (edit here)
# =========================
BASE_MODEL_PATH="/data/xieqiuhao/tjy/downloaded_models/Llama-2-7b-hf"
CHAT_MODEL_PATH="/data/xieqiuhao/tjy/downloaded_models/Llama-2-7b-chat-hf"

OUT_ROOT="outputs/reproduce"
MAX_EVAL_SAMPLES="200"
USE_FP16="1"

# Training hyperparameters
NUM_TRAIN_EPOCHS="5"
LEARNING_RATE="5e-5"
LORA_R="8"
TARGET_MODULES="q_proj,v_proj"

# SafeLoRA hyperparameters
SELECT_LAYERS_TYPE="number"
NUM_PROJ_LAYERS="7"
THRESHOLD="0.35"

TRAIN_FILE="datasets/samsum_1000_bad.jsonl"
TEST_FILE="datasets/samsum_test.jsonl"
LORA_OUT="${OUT_ROOT}/lora_samsum_bad"
SAFE_OUT="${OUT_ROOT}/lora_samsum_bad_safelora"

echo "[1/4] Installing dependencies..."
python -m pip install -r requirements-reproduce.txt

echo "[2/4] LoRA fine-tuning on ${TRAIN_FILE} ..."
TRAIN_ARGS=(
  scripts/train_samsum_lora.py
  --chat_model_path "${CHAT_MODEL_PATH}"
  --train_file "${TRAIN_FILE}"
  --output_dir "${LORA_OUT}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --lora_r "${LORA_R}"
  --target_modules "${TARGET_MODULES}"
)
if [[ "${USE_FP16}" == "1" ]]; then
  TRAIN_ARGS+=(--fp16)
fi
python "${TRAIN_ARGS[@]}"

echo "[3/4] Applying SafeLoRA projection ..."
SAFE_ARGS=(
  scripts/apply_safelora.py
  --chat_model_path "${CHAT_MODEL_PATH}"
  --base_model_path "${BASE_MODEL_PATH}"
  --adapter_path "${LORA_OUT}"
  --output_adapter_path "${SAFE_OUT}"
  --select_layers_type "${SELECT_LAYERS_TYPE}"
  --num_proj_layers "${NUM_PROJ_LAYERS}"
  --threshold "${THRESHOLD}"
)
if [[ "${USE_FP16}" == "1" ]]; then
  SAFE_ARGS+=(--fp16)
fi
python "${SAFE_ARGS[@]}"

echo "[4/4] Evaluating projected adapter on ${TEST_FILE} ..."
EVAL_ARGS=(
  scripts/eval_samsum_rouge.py
  --chat_model_path "${CHAT_MODEL_PATH}"
  --adapter_path "${SAFE_OUT}"
  --test_file "${TEST_FILE}"
  --max_samples "${MAX_EVAL_SAMPLES}"
)
if [[ "${USE_FP16}" == "1" ]]; then
  EVAL_ARGS+=(--fp16)
fi
python "${EVAL_ARGS[@]}"

echo "Done. Output adapters are under: ${OUT_ROOT}"
