#!/usr/bin/env bash
# run_autodl_phase3.sh
# ====================
# Turnkey AutoDL GPU runner following AboutMileStone_3.md Sections II & VI:
#   Step 0 (`p0`)   : Close Milestone 2 P0-1..P0-4 on AutoDL first
#   Step 1 (`smoke`): Download Llama-3.1-8B + scale datasets and run small E2E smoke test
#   Step 2 (`all`)  : Execute ordered Milestone 3 pipeline:
#                     Phase A (WP8 Methodological Validity)
#                     Phase B (WP6 Cross-Family: qwen3b, qwen7b, llama3_8b)
#                     Phase C (WP7 Scaled Cross-Dataset + Zero-Tuning Transfer)
#                     Phase D (WP9 Single-Axis Stress Curves + WP10 Efficiency Pareto)
#                     Phase E (WP11 Paper Package: Figs 2-7, LaTeX, MILESTONE3_EXPERIMENT_REPORT.md)
#
# Usage:
#   bash scripts/phase3/run_autodl_phase3.sh smoke
#   bash scripts/phase3/run_autodl_phase3.sh p0
#   bash scripts/phase3/run_autodl_phase3.sh all medium core3 "42 1024 2026"
#   bash scripts/phase3/run_autodl_phase3.sh all full   all5  "42 1024 2026"

set -euo pipefail

MODE="${1:-smoke}"
SCALE="${2:-medium}"
RAW_MODELS="${3:-${MODEL_GROUP:-core3}}"
SEEDS="${4:-42 1024 2026}"
QUANT="${QUANT:-4bit}"
M2_GRAPH_MODE="${M2_GRAPH_MODE:-auto}"

# Resolve model options: supports core3 / extra2 / all5 group aliases or space-separated model names
if [[ "${RAW_MODELS}" =~ ^(core3|extra2|all5)$ ]]; then
  MODEL_OPTS=(--model-group "${RAW_MODELS}")
else
  read -ra MODEL_ARR <<< "${RAW_MODELS}"
  MODEL_OPTS=(--models "${MODEL_ARR[@]}")
fi

EXTRA_RUN_OPTS=()
if [[ "${CLEANUP_AFTER_MODEL:-0}" == "1" ]]; then
  EXTRA_RUN_OPTS+=(--cleanup-after-model)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${HF_ENDPOINT:-}" ]]; then
  if [[ -z "${COLAB_RELEASE_TAG:-}" && -z "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
    export HF_ENDPOINT="https://hf-mirror.com"
  fi
fi
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1

echo "================================================================"
echo "[AutoDL] Project Root : ${ROOT_DIR}"
echo "[AutoDL] Mode         : ${MODE} | Scale=${SCALE} | Quant=${QUANT}"
echo "[AutoDL] Models       : ${RAW_MODELS} | Seeds=${SEEDS}"
echo "================================================================"

nvidia-smi || true

case "${MODE}" in
  p0)
    echo "[Step 0a] Ensuring Qwen2.5-3B / Qwen2.5-7B + BGE weights & V5_FROZEN integrity..."
    python -u scripts/phase3/autodl_setup.py \
      --models qwen3b qwen7b \
      --scale "${SCALE}" \
      --download-models \
      --quantization "${QUANT}"
    echo "[Step 0b] Running Milestone 2 P0 Closure Fixes (P0-1 .. P0-4)..."
    python -u scripts/phase2/run_p0_closure.py
    python -u scripts/phase2/plot_p0_closure.py
    ;;

  smoke)
    echo "[Step 1] AutoDL Setup + Download models + Smoke Test..."
    python -u scripts/phase3/autodl_setup.py \
      "${MODEL_OPTS[@]}" \
      --scale smoke \
      --download-models \
      --quantization "${QUANT}"

    WP8_RESULT_FILE="results/MileStone3/WP8_MethodValidity/wp8_method_validity_results.json"
    WP8_DEFAULT_SKIP=0
    WP7_DEFAULT_SKIP=0
    if [[ "${RAW_MODELS}" == "extra2" ]]; then
      WP8_DEFAULT_SKIP=1
      WP7_DEFAULT_SKIP=1
    fi
    EFF_SKIP_WP8="${SKIP_WP8:-${WP8_DEFAULT_SKIP}}"
    EFF_SKIP_WP7="${SKIP_WP7:-${WP7_DEFAULT_SKIP}}"
    if [[ "${FORCE_WP8:-0}" == "1" ]]; then EFF_SKIP_WP8=0; fi
    if [[ "${FORCE_WP7:-0}" == "1" ]]; then EFF_SKIP_WP7=0; fi

    if [[ ("${EFF_SKIP_WP8}" == "1" || -f "${WP8_RESULT_FILE}") && "${FORCE_WP8:-0}" != "1" ]]; then
      echo "[Step 1b] WP8 already completed (${WP8_RESULT_FILE} found) or SKIP_WP8=1. Skipping smoke WP8."
    else
      echo "[Step 1b] Running small-scale WP8 smoke test..."
      python -u scripts/phase3/wp8_method_validity.py --model "${WP8_MODEL:-qwen3b}" --scale smoke --quantization "${QUANT}"
    fi

    python -u scripts/phase3/wp6_cross_family.py "${MODEL_OPTS[@]}" --datasets hotpotqa popqa --scale smoke --seeds 42 --m2-graph-mode "${M2_GRAPH_MODE}" --quantization "${QUANT}"
    if [[ "${EFF_SKIP_WP7}" == "1" ]]; then
      echo "[Step 1c] SKIP_WP7=1 (${RAW_MODELS}). Skipping smoke WP7."
    else
      python -u scripts/phase3/wp7_cross_dataset.py --model "${WP7_MODEL:-llama3_8b}" --scale smoke --quantization "${QUANT}"
    fi
    STRESS_SMOKE_OPTS=()
    if [[ "${SKIP_STRESS:-0}" == "1" || "${RAW_MODELS}" == "extra2" ]]; then
      STRESS_SMOKE_OPTS+=(--skip-stress)
    fi
    python -u scripts/phase3/wp9_wp10_stress_efficiency.py "${MODEL_OPTS[@]}" --scale smoke --m2-graph-mode "${M2_GRAPH_MODE}" --quantization "${QUANT}" "${STRESS_SMOKE_OPTS[@]}" "${EXTRA_RUN_OPTS[@]}"
    python -u scripts/phase3/wp11_build_paper_package.py
    echo "[AutoDL] Smoke test completed cleanly!"
    ;;

  all)
    WP8_RESULT_FILE="results/MileStone3/WP8_MethodValidity/wp8_method_validity_results.json"
    WP8_MODEL="${WP8_MODEL:-qwen3b}"
    WP7_MODEL="${WP7_MODEL:-llama3_8b}"
    STRESS_MODEL="${STRESS_MODEL:-llama3_8b}"
    if [[ "${RAW_MODELS}" == "extra2" ]]; then
      WP8_DEFAULT_SKIP=1
      WP7_DEFAULT_SKIP=1
      STRESS_DEFAULT_SKIP=1
    else
      WP8_DEFAULT_SKIP=0
      WP7_DEFAULT_SKIP=0
      STRESS_DEFAULT_SKIP=0
    fi
    EFF_SKIP_WP8="${SKIP_WP8:-${WP8_DEFAULT_SKIP}}"
    EFF_SKIP_WP7="${SKIP_WP7:-${WP7_DEFAULT_SKIP}}"
    EFF_SKIP_STRESS="${SKIP_STRESS:-${STRESS_DEFAULT_SKIP}}"
    if [[ "${FORCE_WP8:-0}" == "1" ]]; then EFF_SKIP_WP8=0; fi
    if [[ "${FORCE_WP7:-0}" == "1" ]]; then EFF_SKIP_WP7=0; fi
    if [[ "${FORCE_STRESS:-0}" == "1" ]]; then EFF_SKIP_STRESS=0; fi

    RESOLVED_MODELS_STR="$(python -c "from scripts.phase3.phase3_common import resolve_phase3_model_list; print(' '.join(resolve_phase3_model_list(None if '${RAW_MODELS}' in ('core3','extra2','all5') else '${RAW_MODELS}'.split(), '${RAW_MODELS}' if '${RAW_MODELS}' in ('core3','extra2','all5') else None)))")"
    read -ra RESOLVED_MODELS <<< "${RESOLVED_MODELS_STR}"
    echo "[AutoDL] Resolved Models: ${RESOLVED_MODELS[*]} | SKIP_WP8=${EFF_SKIP_WP8} | SKIP_WP7=${EFF_SKIP_WP7} | SKIP_STRESS=${EFF_SKIP_STRESS} | CLEANUP=${CLEANUP_AFTER_MODEL:-0}"

    if [[ "${CLEANUP_AFTER_MODEL:-0}" == "1" ]]; then
      # Sequential per-model lifecycle: avoids both disk overflow AND premature deletion before WP7/WP9/WP10
      for MK in "${RESOLVED_MODELS[@]}"; do
        echo "----------------------------------------------------------------"
        echo "[Model Lifecycle] Bootstrapping single model=${MK}..."
        echo "----------------------------------------------------------------"
        python -u scripts/phase3/autodl_setup.py \
          --models "${MK}" \
          --scale "${SCALE}" \
          --download-datasets \
          --download-models \
          --quantization "${QUANT}"

        if [[ "${EFF_SKIP_WP8}" != "1" && "${MK}" == "${WP8_MODEL}" && (! -f "${WP8_RESULT_FILE}" || "${FORCE_WP8:-0}" == "1") ]]; then
          echo "[Phase A] Running WP8 Methodological Validity Gate (model=${MK})..."
          python -u scripts/phase3/wp8_method_validity.py --model "${MK}" --scale "${SCALE}" --quantization "${QUANT}"
        fi

        echo "[Phase B] Running WP6 Cross-Family Generalization (model=${MK})..."
        python -u scripts/phase3/wp6_cross_family.py --models "${MK}" --datasets hotpotqa popqa 2wikimultihop musique --scale "${SCALE}" --seeds ${SEEDS} --m2-graph-mode "${M2_GRAPH_MODE}" --quantization "${QUANT}"

        if [[ "${EFF_SKIP_WP7}" != "1" && "${MK}" == "${WP7_MODEL}" ]]; then
          echo "[Phase C] Running WP7 Cross-Dataset & Zero-Tuning Transfer (model=${MK})..."
          python -u scripts/phase3/wp7_cross_dataset.py --model "${MK}" --scale "${SCALE}" --quantization "${QUANT}"
        fi

        PER_MODEL_STRESS_OPTS=(--skip-stress)
        if [[ "${EFF_SKIP_STRESS}" != "1" && "${MK}" == "${STRESS_MODEL}" ]]; then
          PER_MODEL_STRESS_OPTS=(--stress-model "${MK}")
        fi
        echo "[Phase D] Running WP9/WP10 + Weight Cleanup (model=${MK})..."
        python -u scripts/phase3/wp9_wp10_stress_efficiency.py --models "${MK}" --scale "${SCALE}" --m2-graph-mode "${M2_GRAPH_MODE}" --quantization "${QUANT}" "${PER_MODEL_STRESS_OPTS[@]}" --cleanup-after-model
        rm -rf ~/.cache/modelscope ~/.cache/huggingface/hub 2>/dev/null || true
      done
    else
      echo "[Bootstrap] Checking models, scaled datasets, and V5_FROZEN integrity..."
      python -u scripts/phase3/autodl_setup.py \
        "${MODEL_OPTS[@]}" \
        --scale "${SCALE}" \
        --download-datasets \
        --download-models \
        --quantization "${QUANT}"

      if [[ ("${EFF_SKIP_WP8}" == "1" || -f "${WP8_RESULT_FILE}") && "${FORCE_WP8:-0}" != "1" ]]; then
        echo "[Phase A] WP8 already completed (${WP8_RESULT_FILE} found) or SKIP_WP8=1. Skipping WP8 to save compute."
      else
        echo "[Phase A] Running WP8 Methodological Validity Gate (model=${WP8_MODEL})..."
        python -u scripts/phase3/wp8_method_validity.py --model "${WP8_MODEL}" --scale "${SCALE}" --quantization "${QUANT}"
      fi

      echo "[Phase B] Running WP6 Cross-Family Generalization (Tier 1)..."
      python -u scripts/phase3/wp6_cross_family.py "${MODEL_OPTS[@]}" --datasets hotpotqa popqa 2wikimultihop musique --scale "${SCALE}" --seeds ${SEEDS} --m2-graph-mode "${M2_GRAPH_MODE}" --quantization "${QUANT}"

      if [[ "${EFF_SKIP_WP7}" == "1" ]]; then
        echo "[Phase C] SKIP_WP7=1 (${RAW_MODELS}). Skipping WP7."
      else
        echo "[Phase C] Running WP7 Cross-Dataset & Zero-Tuning Transfer (model=${WP7_MODEL})..."
        python -u scripts/phase3/wp7_cross_dataset.py --model "${WP7_MODEL}" --scale "${SCALE}" --quantization "${QUANT}"
      fi

      BATCH_STRESS_OPTS=()
      if [[ "${EFF_SKIP_STRESS}" == "1" ]]; then
        BATCH_STRESS_OPTS+=(--skip-stress)
      else
        BATCH_STRESS_OPTS+=(--stress-model "${STRESS_MODEL}")
      fi
      echo "[Phase D] Running WP9 Stress Curves & WP10 Efficiency Pareto (Tier 3)..."
      python -u scripts/phase3/wp9_wp10_stress_efficiency.py "${MODEL_OPTS[@]}" --scale "${SCALE}" --m2-graph-mode "${M2_GRAPH_MODE}" --quantization "${QUANT}" "${BATCH_STRESS_OPTS[@]}"
    fi

    echo "[Phase E] Generating WP11 Paper Package (Figs 2-7, LaTeX tables, Failure Analysis, Report)..."
    python -u scripts/phase3/wp11_build_paper_package.py
    echo "[AutoDL] Full Milestone 3 pipeline finished!"
    ;;

  *)
    echo "Unknown mode '${MODE}'. Use: p0 | smoke | all"
    exit 1
    ;;
esac
