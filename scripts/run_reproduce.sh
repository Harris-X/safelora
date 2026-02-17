#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_reproduce.sh <BASE_MODEL_PATH> <CHAT_MODEL_PATH> [OUT_ROOT] [MAX_EVAL_SAMPLES] [USE_FP16]"
  echo "Example: bash scripts/run_reproduce.sh /models/Llama-2-7b-base /models/Llama-2-7b-chat outputs/reproduce 200 1"
  exit 1
fi

BASE_MODEL_PATH="$1"
CHAT_MODEL_PATH="$2"
OUT_ROOT="${3:-outputs/reproduce}"
MAX_EVAL_SAMPLES="${4:-200}"
USE_FP16="${5:-1}"

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
  --num_train_epochs 5
  --learning_rate 5e-5
  --lora_r 8
  --target_modules q_proj,v_proj
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
  --select_layers_type number
  --num_proj_layers 7
  --threshold 0.35
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
