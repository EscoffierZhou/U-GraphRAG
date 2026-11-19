"""
scripts/run_step1_pilot.py
==========================
Step 1 Pilot Benchmark Runner under 8GB VRAM RTX 4070 Constraint.
Strictly serial execution across isolated subprocesses to guarantee 100% CUDA VRAM clean-up.

Workflow:
1. Stage A-1: Qwen2.5-3B-Instruct Smoke Test (5 items)
2. Stage A-2: Qwen2.5-3B-Instruct HotpotQA Dev Top-30 Evaluation -> results/step1_qwen25_3b_hotpot30.json
3. Stage B-1: Qwen2.5-7B-Instruct Smoke Test (5 items)
4. Stage B-2: Qwen2.5-7B-Instruct HotpotQA Dev Top-30 Evaluation -> results/step1_qwen25_7b_hotpot30.json
5. Verification and comparative analysis summary
"""

import sys
import json
import time
import subprocess
from pathlib import Path

# Paths setup
PYTHON_EXE = sys.executable
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_3B_DIR = BASE_DIR / "models" / "Qwen2.5-3B-Instruct"
MODEL_7B_DIR = BASE_DIR / "models" / "Qwen2.5-7B-Instruct"
DENSE_MODEL_DIR = BASE_DIR / "models" / "bge-small-en-v1.5"
HOTPOT_DATASET = BASE_DIR / "data" / "hotpotqa_dev_300.json"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_3B_JSON = RESULTS_DIR / "step1_qwen25_3b_hotpot30.json"
OUTPUT_7B_JSON = RESULTS_DIR / "step1_qwen25_7b_hotpot30.json"
OUTPUT_REPORT_MD = RESULTS_DIR / "STEP1_REPORT.md"

try:
    from src.evaluation_toolkit import DetailedReportGenerator
except ImportError:
    DetailedReportGenerator = None


def verify_prerequisites():
    print("=" * 72)
    print("Checking prerequisites for Step 1 Pilot Benchmark...")
    print(f"Working Directory: {BASE_DIR}")
    print(f"Python Executable: {PYTHON_EXE}")

    assert MODEL_3B_DIR.exists(), f"3B model directory not found: {MODEL_3B_DIR}"
    assert MODEL_7B_DIR.exists(), f"7B model directory not found: {MODEL_7B_DIR}"
    assert DENSE_MODEL_DIR.exists(), f"Dense model directory not found: {DENSE_MODEL_DIR}"
    assert HOTPOT_DATASET.exists(), f"HotpotQA dataset not found: {HOTPOT_DATASET}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("All required models, datasets, and directories are verified.")
    print("=" * 72)


def run_stage(cmd: list, desc: str) -> float:
    print("\n" + "=" * 72)
    print(f"STAGE EXECUTION: {desc}")
    print(f"COMMAND: {' '.join(cmd)}")
    print("=" * 72 + "\n")

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    for line in iter(proc.stdout.readline, ''):
        print(line, end='', flush=True)

    proc.stdout.close()
    ret_code = proc.wait()
    t_cost = time.time() - t0

    if ret_code != 0:
        raise RuntimeError(f"Stage execution failed (code {ret_code}): {desc}")

    print(f"\n[+] {desc} finished in {t_cost:.2f} seconds.")
    return t_cost


def print_comparison():
    print("\n" + "=" * 72)
    print("         STEP 1 PILOT BENCHMARK COMPARATIVE SUMMARY")
    print("=" * 72)

    if not OUTPUT_3B_JSON.exists() or not OUTPUT_7B_JSON.exists():
        print("Warning: Benchmark output files missing. Cannot show comparison.")
        return

    with open(OUTPUT_3B_JSON, "r", encoding="utf-8") as f:
        data_3b = json.load(f)
    with open(OUTPUT_7B_JSON, "r", encoding="utf-8") as f:
        data_7b = json.load(f)

    p3b = data_3b.get("system_profile", {})
    p7b = data_7b.get("system_profile", {})
    m3b = data_3b.get("metrics", {})
    m7b = data_7b.get("metrics", {})

    print(f"{'Metric':<32} | {'Qwen2.5-3B-Instruct':<18} | {'Qwen2.5-7B-Instruct':<18}")
    print("-" * 72)
    print(f"{'Exact Match (EM)':<32} | {m3b.get('exact_match', 0.0):<17.2f}% | {m7b.get('exact_match', 0.0):<17.2f}%")
    print(f"{'Token-level F1 Score':<32} | {m3b.get('f1_score', 0.0):<17.2f}% | {m7b.get('f1_score', 0.0):<17.2f}%")
    if 'rouge_1' in m7b or 'rouge_1' in m3b:
        print(f"{'ROUGE-1 F1 Score':<32} | {m3b.get('rouge_1', 0.0):<17.2f}% | {m7b.get('rouge_1', 0.0):<17.2f}%")
        print(f"{'ROUGE-L F1 Score':<32} | {m3b.get('rouge_l', 0.0):<17.2f}% | {m7b.get('rouge_l', 0.0):<17.2f}%")
        print(f"{'BLEU-1 Score':<32} | {m3b.get('bleu_1', 0.0):<17.2f}% | {m7b.get('bleu_1', 0.0):<17.2f}%")
    if 'sp_f1' in m7b or 'sp_f1' in m3b:
        print(f"{'Supporting Facts F1':<32} | {m3b.get('sp_f1', 0.0):<17.2f}% | {m7b.get('sp_f1', 0.0):<17.2f}%")
    print(f"{'Total Evaluation Samples':<32} | {data_3b.get('sample_count', 0):<18} | {data_7b.get('sample_count', 0):<18}")
    print(f"{'Total Wall Time (s)':<32} | {p3b.get('total_wall_sec', 0.0):<18.2f} | {p7b.get('total_wall_sec', 0.0):<18.2f}")
    print(f"{'Avg Latency per Query (s)':<32} | {p3b.get('avg_latency_sec', 0.0):<18.4f} | {p7b.get('avg_latency_sec', 0.0):<18.4f}")
    print(f"{'Avg Retrieval Latency (s)':<32} | {p3b.get('avg_retrieval_latency_sec', 0.0):<18.4f} | {p7b.get('avg_retrieval_latency_sec', 0.0):<18.4f}")
    print(f"{'Avg Generation Latency (s)':<32} | {p3b.get('avg_generation_latency_sec', 0.0):<18.4f} | {p7b.get('avg_generation_latency_sec', 0.0):<18.4f}")
    print(f"{'Avg Forward Calls per Query':<32} | {p3b.get('avg_forward_calls_per_sample', 1.0):<18.2f} | {p7b.get('avg_forward_calls_per_sample', 1.0):<18.2f}")
    print(f"{'Peak CUDA Allocated (MB)':<32} | {p3b.get('peak_cuda_allocated_mb', 0.0):<18.2f} | {p7b.get('peak_cuda_allocated_mb', 0.0):<18.2f}")
    print(f"{'Peak CUDA Reserved (MB)':<32} | {p3b.get('peak_cuda_reserved_mb', 0.0):<18.2f} | {p7b.get('peak_cuda_reserved_mb', 0.0):<18.2f}")
    print(f"{'Peak Process VRAM (MB)':<32} | {str(p3b.get('peak_nvml_process_mb', 'N/A')):<18} | {str(p7b.get('peak_nvml_process_mb', 'N/A')):<18}")
    print(f"{'8GB VRAM Headroom (MB)':<32} | {p3b.get('vram_headroom_8gb_mb', 0.0):<18.2f} | {p7b.get('vram_headroom_8gb_mb', 0.0):<18.2f}")
    print("=" * 72)

    # Generate Detailed Markdown Report
    if DetailedReportGenerator is not None:
        DetailedReportGenerator.generate_step1_report(data_3b, data_7b, OUTPUT_REPORT_MD)
        print(f"\n[+] Comprehensive Benchmark Report written to: {OUTPUT_REPORT_MD}")

    # Detailed Case Study Inspection
    samples_3b = data_3b.get("samples", [])
    samples_7b = data_7b.get("samples", [])
    print("\n" + "=" * 72)
    print("         SAMPLE CASE STUDY (3B vs 7B Predictions)")
    print("=" * 72)
    for i in range(min(len(samples_3b), len(samples_7b))):
        s3 = samples_3b[i]
        s7 = samples_7b[i]
        f1_7 = s7.get("f1", 0.0)
        # Show cases with non-zero F1 or representative interesting cases
        if f1_7 > 0.0 or i in [0, 1, 2, 5, 8]:
            print(f"\n[Case #{i+1}] ID: {s3.get('sample_id')}")
            print(f"  Question:     {s3.get('question')}")
            print(f"  Ground Truth: {s3.get('ground_truth')}")
            print(f"  3B Pred:      {s3.get('prediction')} (EM={s3.get('em')}, F1={s3.get('f1')})")
            print(f"  7B Pred:      {s7.get('prediction')} (EM={s7.get('em')}, F1={s7.get('f1')})")
            ret_titles = [d.get("title", "") for d in s7.get("retrieved_docs", [])]
            print(f"  Top-3 Docs:   {ret_titles}")
    print("=" * 72)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Step 1 Pilot Runner")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "3b_smoke", "3b_hotpot", "7b_smoke", "7b_hotpot", "compare"],
        help="Stage to execute (default: all)"
    )
    args = parser.parse_args()

    verify_prerequisites()
    start_time = time.time()

    cmd_smoke_3b = [
        PYTHON_EXE, "-u", "src/baseline_rag.py",
        "--model-path", str(MODEL_3B_DIR),
        "--dense-model-path", str(DENSE_MODEL_DIR),
        "--smoke-test"
    ]
    cmd_hotpot_3b = [
        PYTHON_EXE, "-u", "src/baseline_rag.py",
        "--model-path", str(MODEL_3B_DIR),
        "--dense-model-path", str(DENSE_MODEL_DIR),
        "--dataset-path", str(HOTPOT_DATASET),
        "--limit", "30",
        "--top-k", "3",
        "--output-json", str(OUTPUT_3B_JSON)
    ]
    cmd_smoke_7b = [
        PYTHON_EXE, "-u", "src/baseline_rag.py",
        "--model-path", str(MODEL_7B_DIR),
        "--dense-model-path", str(DENSE_MODEL_DIR),
        "--smoke-test"
    ]
    cmd_hotpot_7b = [
        PYTHON_EXE, "-u", "src/baseline_rag.py",
        "--model-path", str(MODEL_7B_DIR),
        "--dense-model-path", str(DENSE_MODEL_DIR),
        "--dataset-path", str(HOTPOT_DATASET),
        "--limit", "30",
        "--top-k", "3",
        "--output-json", str(OUTPUT_7B_JSON)
    ]

    if args.stage in ("all", "3b_smoke"):
        run_stage(cmd_smoke_3b, "Stage A-1: Qwen2.5-3B Smoke Test (5 questions)")

    if args.stage in ("all", "3b_hotpot"):
        run_stage(cmd_hotpot_3b, "Stage A-2: Qwen2.5-3B HotpotQA Dev Top-30 Evaluation")

    if args.stage in ("all", "7b_smoke"):
        run_stage(cmd_smoke_7b, "Stage B-1: Qwen2.5-7B Smoke Test (5 questions)")

    if args.stage in ("all", "7b_hotpot"):
        run_stage(cmd_hotpot_7b, "Stage B-2: Qwen2.5-7B HotpotQA Dev Top-30 Evaluation")

    total_duration = time.time() - start_time
    print(f"\nExecution finished in {total_duration:.2f} seconds.")

    print_comparison()


if __name__ == "__main__":
    main()
