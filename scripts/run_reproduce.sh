#!/usr/bin/env bash
set -euo pipefail

# =========================
# Default config (edit here)
# =========================
BASE_MODEL_PATH="/data_nvme1n1/xieqiuhao/tjy/downloaded_models/Llama-2-7b-hf"
CHAT_MODEL_PATH="/data_nvme1n1/xieqiuhao/tjy/downloaded_models/Llama-2-7b-chat-hf"

OUT_ROOT="outputs/reproduce"
MAX_EVAL_SAMPLES="-1"
USE_FP16="1"
GPU_ID="0"

# Run mode
AUTO_DETACH="1"

# Training hyperparameters
NUM_TRAIN_EPOCHS="5"
LEARNING_RATE="5e-5"
LORA_R="8"
TARGET_MODULES="q_proj,v_proj"

# Torch install config (must match your CUDA runtime)
TORCH_VERSION="2.4.1"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

# SafeLoRA hyperparameters
SELECT_LAYERS_TYPE="number"
NUM_PROJ_LAYERS="7"
THRESHOLD="0.35"

TRAIN_FILE="datasets/samsum_1000_bad.jsonl"
TEST_FILE="datasets/samsum_test.jsonl"
LORA_OUT="${OUT_ROOT}/lora_samsum_bad"
SAFE_OUT="${OUT_ROOT}/lora_samsum_bad_safelora"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"
METRICS_JSON="${OUT_ROOT}/metrics_${TIMESTAMP}.json"
SST2_BASE_METRICS_JSON="${OUT_ROOT}/sst2_base_metrics_${TIMESTAMP}.json"
SUMMARY_FILE="${OUT_ROOT}/run_summary_${TIMESTAMP}.txt"

if [[ "${AUTO_DETACH}" == "1" && "${RUN_REPRODUCE_NOHUP:-0}" != "1" ]]; then
  nohup env RUN_REPRODUCE_NOHUP=1 bash "$0" > "${LOG_FILE}" 2>&1 &
  PID="$!"
  echo "Started in background."
  echo "PID: ${PID}"
  echo "Log: ${LOG_FILE}"
  echo "Summary (will be generated): ${SUMMARY_FILE}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

{
  echo "run_timestamp=${TIMESTAMP}"
  echo "gpu_id=${GPU_ID}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "base_model_path=${BASE_MODEL_PATH}"
  echo "chat_model_path=${CHAT_MODEL_PATH}"
  echo "train_file=${TRAIN_FILE}"
  echo "test_file=${TEST_FILE}"
  echo "max_eval_samples=${MAX_EVAL_SAMPLES}"
  echo "lora_output_path=${LORA_OUT}"
  echo "safelora_output_path=${SAFE_OUT}"
  echo "metrics_output_path=${METRICS_JSON}"
  echo "sst2_base_metrics_output_path=${SST2_BASE_METRICS_JSON}"
  echo "log_file=${LOG_FILE}"
} > "${SUMMARY_FILE}"

echo "[1/4] Installing dependencies..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install --prefer-binary "numpy==1.26.4"
python -m pip install --prefer-binary --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
python -m pip install --prefer-binary -r requirements-reproduce.txt

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

echo "[4/5] Evaluating projected adapter on ${TEST_FILE} ..."
EVAL_ARGS=(
  scripts/eval_samsum_rouge.py
  --chat_model_path "${CHAT_MODEL_PATH}"
  --adapter_path "${SAFE_OUT}"
  --test_file "${TEST_FILE}"
  --max_samples "${MAX_EVAL_SAMPLES}"
  --metrics_output_path "${METRICS_JSON}"
)
if [[ "${USE_FP16}" == "1" ]]; then
  EVAL_ARGS+=(--fp16)
fi
python "${EVAL_ARGS[@]}"

echo "[5/5] Evaluating pre-finetune base(chat) model on SST2 ..."
SST2_BASE_ARGS=(
  scripts/eval_sst2_accuracy.py
  --model_path "${CHAT_MODEL_PATH}"
  --split validation
  --max_samples -1
  --metrics_output_path "${SST2_BASE_METRICS_JSON}"
)
if [[ "${USE_FP16}" == "1" ]]; then
  SST2_BASE_ARGS+=(--fp16)
fi
python "${SST2_BASE_ARGS[@]}"

if [[ -f "${METRICS_JSON}" ]]; then
  python - "$METRICS_JSON" "$SUMMARY_FILE" << 'PY'
import json
import sys

metrics_path = sys.argv[1]
summary_path = sys.argv[2]

with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

rouge1 = metrics.get("rouge1", None)
rougeL = metrics.get("rougeL", None)

with open(summary_path, "a", encoding="utf-8") as f:
    f.write(f"\nrouge1={rouge1}\n")
    f.write(f"rougeL={rougeL}\n")

print(f"Final metric rouge1: {rouge1}")
print(f"Final metric rougeL: {rougeL}")
PY
fi

if [[ -f "${SST2_BASE_METRICS_JSON}" ]]; then
  python - "$SST2_BASE_METRICS_JSON" "$SUMMARY_FILE" << 'PY'
import json
import sys

metrics_path = sys.argv[1]
summary_path = sys.argv[2]

with open(metrics_path, "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data.get("summary", {})
acc = summary.get("accuracy", None)
correct = summary.get("correct", None)
total = summary.get("total", None)

with open(summary_path, "a", encoding="utf-8") as f:
    f.write(f"\nsst2_base_accuracy={acc}\n")
    f.write(f"sst2_base_correct={correct}\n")
    f.write(f"sst2_base_total={total}\n")

print(f"SST2 base accuracy: {acc}")
PY
fi

echo "Done. Output adapters are under: ${OUT_ROOT}"
echo "Summary file: ${SUMMARY_FILE}"
echo "Metrics json: ${METRICS_JSON}"
