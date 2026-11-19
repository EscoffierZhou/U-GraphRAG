#!/usr/bin/env bash
# run_milestone_after.sh
# ====================
# Incremental MileStone_After runner on Linux RTX 4090 (runs AFTER WP6-WP11 finish
# without touching or invalidating any MileStone 3 outputs).

set -euo pipefail

MODEL="${1:-llama3_8b}"
SCALE="${2:-medium}"
QUANT="${3:-4bit}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${HF_ENDPOINT:-}" && -z "${COLAB_RELEASE_TAG:-}" && -z "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi
export PYTHONUNBUFFERED=1

echo "================================================================"
echo "[MileStone_After] Root=${ROOT_DIR} | Model=${MODEL} | Scale=${SCALE} | Quant=${QUANT}"
echo "================================================================"

echo "[Step 1] Zero-GPU Post-Hoc Split Conformal Audit over WP8 & WP6 Checkpoints..."
python -u results/MileStone4_Competitor/dist/milestone_after_posthoc_conformal.py || true

echo "[Step 2] Running SOTA Horse-Race (BGE-Reranker, CK-PLUG, InstructRAG-ICL, CAD, U-GraphRAG+)..."
python -u results/MileStone4_Competitor/dist/run_milestone_after_horserace.py \
  --model "${MODEL}" \
  --scale "${SCALE}" \
  --quantization "${QUANT}"

echo "[MileStone_After] Completed! Results -> results/MileStone4_Competitor/dist/outputs/"
