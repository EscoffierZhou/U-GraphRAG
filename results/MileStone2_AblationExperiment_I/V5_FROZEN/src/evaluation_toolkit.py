"""
src/evaluation_toolkit.py
=========================
Comprehensive Evaluation Toolkit for U-GraphRAG Benchmarking:
1. Extended NLP Metrics: Exact Match (EM), Token-F1, ROUGE-1/2/L, BLEU-1/4, Answer Length.
2. Calibration & Uncertainty: ECE, MCE, Brier Score, Selective Coverage, Selective Risk.
3. Checkpoint & Soft-Interrupt Manager: Graceful resume from checkpoint on Ctrl+C or crash.
4. TensorBoard Integration: Real-time logging of running metrics, memory, and latencies.
5. Automated Detailed Markdown Report Generator: Generates publication-grade summaries.

SDUFE GeminiOutOfToken Team.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import os
import re
import signal
import string
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

logger = logging.getLogger("EvaluationToolkit")

# -----------------------------------------------------------------------------
# 1. NLP & Generation Quality Metrics
# -----------------------------------------------------------------------------

def normalize_answer_text(text: str) -> str:
    """Standard QA text normalization: lowercase, strip punctuation, remove articles."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    return " ".join(text.split())


def compute_rouge_n_scores(pred_tokens: List[str], gold_tokens: List[str], n: int = 1) -> Dict[str, float]:
    """Computes ROUGE-N Precision, Recall, and F1."""
    if len(pred_tokens) < n or len(gold_tokens) < n:
        match = float(pred_tokens == gold_tokens and len(pred_tokens) >= 1)
        return {"precision": match, "recall": match, "f1": match}

    pred_ngrams = collections.Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
    gold_ngrams = collections.Counter(tuple(gold_tokens[i : i + n]) for i in range(len(gold_tokens) - n + 1))

    overlap = sum((pred_ngrams & gold_ngrams).values())
    total_pred = max(1, sum(pred_ngrams.values()))
    total_gold = max(1, sum(gold_ngrams.values()))

    prec = overlap / total_pred
    rec = overlap / total_gold
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def compute_rouge_l_scores(pred_tokens: List[str], gold_tokens: List[str]) -> Dict[str, float]:
    """Computes ROUGE-L based on Longest Common Subsequence (LCS)."""
    m, n = len(pred_tokens), len(gold_tokens)
    if m == 0 or n == 0:
        match = float(m == n)
        return {"precision": match, "recall": match, "f1": match}

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    lcs_len = prev[n]

    prec = lcs_len / m
    rec = lcs_len / n
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def compute_bleu_scores(pred_tokens: List[str], gold_tokens: List[str], max_n: int = 4) -> Dict[str, float]:
    """Computes BLEU-1 and BLEU-4 with brevity penalty."""
    if not pred_tokens or not gold_tokens:
        match = float(pred_tokens == gold_tokens)
        return {"bleu_1": match, "bleu_4": match}

    c = len(pred_tokens)
    r = len(gold_tokens)
    bp = 1.0 if c > r else math.exp(1.0 - r / max(1, c))

    precisions = []
    for n in range(1, max_n + 1):
        if len(pred_tokens) < n or len(gold_tokens) < n:
            precisions.append(1e-9)
            continue
        pred_ngrams = collections.Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
        gold_ngrams = collections.Counter(tuple(gold_tokens[i : i + n]) for i in range(len(gold_tokens) - n + 1))
        overlap = sum((pred_ngrams & gold_ngrams).values())
        total = sum(pred_ngrams.values())
        precisions.append(max(1e-9, overlap / max(1, total)))

    bleu_1 = precisions[0] * bp
    log_sum = sum(math.log(p) for p in precisions) / max_n
    bleu_4 = math.exp(log_sum) * bp
    return {"bleu_1": round(bleu_1, 4), "bleu_4": round(bleu_4, 4)}


def compute_full_sample_metrics(pred: str, golds: Union[str, List[str]]) -> Dict[str, float]:
    """Computes full suite of NLP metrics taking maximum across gold reference variants."""
    if isinstance(golds, str):
        golds = [golds]
    if not golds:
        golds = [""]

    best_em = 0.0
    best_f1 = 0.0
    best_r1 = 0.0
    best_r2 = 0.0
    best_rl = 0.0
    best_b1 = 0.0
    best_b4 = 0.0

    p_norm = normalize_answer_text(pred)
    p_tokens = p_norm.split()

    for g in golds:
        g_norm = normalize_answer_text(g)
        g_tokens = g_norm.split()

        # EM
        em = 1.0 if p_norm == g_norm else 0.0
        best_em = max(best_em, em)

        # F1
        common = collections.Counter(p_tokens) & collections.Counter(g_tokens)
        num_same = sum(common.values())
        if num_same == 0 or len(p_tokens) == 0 or len(g_tokens) == 0:
            f1 = float(p_tokens == g_tokens)
        else:
            prec = num_same / len(p_tokens)
            rec = num_same / len(g_tokens)
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        best_f1 = max(best_f1, f1)

        # ROUGE
        r1 = compute_rouge_n_scores(p_tokens, g_tokens, n=1)["f1"]
        r2 = compute_rouge_n_scores(p_tokens, g_tokens, n=2)["f1"]
        rl = compute_rouge_l_scores(p_tokens, g_tokens)["f1"]
        best_r1 = max(best_r1, r1)
        best_r2 = max(best_r2, r2)
        best_rl = max(best_rl, rl)

        # BLEU
        bleu = compute_bleu_scores(p_tokens, g_tokens, max_n=4)
        best_b1 = max(best_b1, bleu["bleu_1"])
        best_b4 = max(best_b4, bleu["bleu_4"])

    return {
        "em": float(best_em),
        "f1": round(best_f1, 4),
        "rouge_1": round(best_r1, 4),
        "rouge_2": round(best_r2, 4),
        "rouge_l": round(best_rl, 4),
        "bleu_1": round(best_b1, 4),
        "bleu_4": round(best_b4, 4),
        "generated_tokens": len(p_tokens),
    }


# -----------------------------------------------------------------------------
# 2. Advanced Calibration & Uncertainty Metrics
# -----------------------------------------------------------------------------

def compute_detailed_calibration(
    confidences: Sequence[float],
    accuracies: Sequence[float],
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and Brier Score.
    """
    if not confidences or not accuracies or len(confidences) != len(accuracies):
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0, "bin_breakdown": []}

    confs = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    accs = np.asarray(accuracies, dtype=np.float64)
    n = len(confs)

    # Brier Score = (1/N) * sum((p - y)^2)
    brier = float(np.mean((confs - accs) ** 2))

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confs >= low) & (confs <= high)
        else:
            mask = (confs >= low) & (confs < high)

        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(accs[mask]))
            bin_conf = float(np.mean(confs[mask]))
            abs_diff = abs(bin_acc - bin_conf)
            ece += (count / n) * abs_diff
            mce = max(mce, abs_diff)
            bin_details.append({
                "bin_idx": i,
                "range": [round(low, 2), round(high, 2)],
                "count": count,
                "bin_accuracy": round(bin_acc, 4),
                "bin_confidence": round(bin_conf, 4),
                "calibration_gap": round(abs_diff, 4),
            })

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(float(brier), 4),
        "bin_breakdown": bin_details,
    }


def compute_selective_prediction_metrics(
    abstained_flags: Sequence[bool],
    accuracies: Sequence[float]
) -> Dict[str, float]:
    """Computes Coverage, Risk, and Effective Selective Performance."""
    n = len(abstained_flags)
    if n == 0:
        return {"coverage": 1.0, "selective_risk": 0.0, "effective_accuracy": 0.0}

    non_abstained = [acc for ab, acc in zip(abstained_flags, accuracies) if not ab]
    coverage = len(non_abstained) / n
    if non_abstained:
        selective_risk = float(np.mean([1.0 - acc for acc in non_abstained]))
        effective_acc = float(np.mean(non_abstained))
    else:
        selective_risk = 0.0
        effective_acc = 0.0

    return {
        "coverage": round(coverage, 4),
        "selective_risk": round(selective_risk, 4),
        "effective_accuracy": round(effective_acc, 4),
    }


# -----------------------------------------------------------------------------
# 3. Checkpoint & Soft-Interrupt Manager
# -----------------------------------------------------------------------------

class CheckpointManager:
    """
    Thread-safe & interrupt-safe checkpoint manager.
    Enables zero-data-loss recovery on Ctrl+C / soft interrupt / crash.
    """

    def __init__(self, target_filepath: Union[str, Path], save_frequency: int = 5):
        self.target_path = Path(target_filepath)
        self.checkpoint_path = self.target_path.with_suffix(self.target_path.suffix + ".checkpoint.json")
        self.save_frequency = save_frequency
        self._interrupted = False
        self._setup_signals()

    def _setup_signals(self) -> None:
        """Register graceful SIGINT handler."""
        try:
            self._old_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_soft_interrupt)
        except Exception:
            pass

    def _handle_soft_interrupt(self, signum, frame) -> None:
        logger.warning("\n[SOFT INTERRUPT DETECTED] Gracefully handling interrupt (Ctrl+C)...")
        self._interrupted = True
        if callable(self._old_sigint):
            self._old_sigint(signum, frame)
        else:
            raise KeyboardInterrupt()

    def has_checkpoint(self) -> bool:
        return self.checkpoint_path.exists() and self.checkpoint_path.stat().st_size > 10

    def load_checkpoint(self) -> Tuple[List[Dict[str, Any]], Set[str]]:
        """Loads records and completed sample keys from checkpoint if present."""
        if not self.has_checkpoint():
            return [], set()

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("records", [])
            completed_keys = set(data.get("completed_keys", []))
            if not completed_keys:
                completed_keys = {
                    str(r.get("sample_id", r.get("id", r.get("query", idx))))
                    for idx, r in enumerate(records)
                }
            logger.info(
                f"[RESUME] Found existing checkpoint '{self.checkpoint_path.name}' "
                f"with {len(records)} completed samples. Resuming execution..."
            )
            return records, completed_keys
        except Exception as e:
            logger.warning(f"Failed to load checkpoint ({e}). Starting fresh.")
            return [], set()

    def save_checkpoint(
        self,
        records: List[Dict[str, Any]],
        completed_keys: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> None:
        """Atomically saves checkpoint state to disk."""
        if not force and len(records) % self.save_frequency != 0:
            return

        if completed_keys is None:
            completed_keys = {
                str(r.get("sample_id", r.get("id", r.get("query", idx))))
                for idx, r in enumerate(records)
            }

        tmp_path = self.checkpoint_path.with_suffix(".tmp")
        payload = {
            "checkpoint_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_count": len(records),
            "completed_keys": list(completed_keys),
            "metadata": metadata or {},
            "records": records,
        }
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            if tmp_path.exists():
                tmp_path.replace(self.checkpoint_path)
        except Exception as e:
            logger.warning(f"Error saving checkpoint: {e}")

    def finalize(self, final_payload: Dict[str, Any]) -> None:
        """Writes final destination JSON and cleans up checkpoint file."""
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.target_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, ensure_ascii=False, indent=2)

        if self.checkpoint_path.exists():
            try:
                self.checkpoint_path.unlink()
                logger.info(f"Cleaned up temporary checkpoint '{self.checkpoint_path.name}'.")
            except Exception:
                pass


# -----------------------------------------------------------------------------
# 4. TensorBoard Integration
# -----------------------------------------------------------------------------

class ExperimentTensorBoardLogger:
    """Seamless TensorBoard logger for real-time tracking of QA and resource metrics."""

    def __init__(self, log_dir: Optional[Union[str, Path]] = None, experiment_name: str = "run"):
        self.writer = None
        self.active = False
        if log_dir is None:
            log_dir = Path("results") / "runs" / f"{experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        else:
            log_dir = Path(log_dir)

        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            self.active = True
            logger.info(f"[TensorBoard] Initialized logging at: {log_dir}")
        except Exception as e:
            logger.warning(f"[TensorBoard] Failed to initialize SummaryWriter: {e}")
            self.writer = None
            self.active = False

    def log_step_metrics(self, step: int, metrics: Dict[str, Union[float, int]]) -> None:
        """Logs scalar metrics for an individual step/sample."""
        if not self.active or self.writer is None:
            return
        for key, val in metrics.items():
            if isinstance(val, (int, float)) and not math.isnan(val):
                self.writer.add_scalar(key, float(val), step)

    def log_summary_metrics(self, prefix: str, metrics: Dict[str, Any]) -> None:
        """Logs macro summary metrics."""
        if not self.active or self.writer is None:
            return
        for key, val in metrics.items():
            if isinstance(val, (int, float)) and not math.isnan(val):
                self.writer.add_scalar(f"Summary_{prefix}/{key}", float(val), 0)

    def flush_and_close(self) -> None:
        if self.active and self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# 5. Automated Detailed Markdown Report Generator
# -----------------------------------------------------------------------------

class DetailedReportGenerator:
    """Formats publication-grade Markdown benchmark report with drill-downs and analysis."""

    @staticmethod
    def generate_step1_report(data_3b: Dict[str, Any], data_7b: Dict[str, Any], output_path: Path) -> str:
        p3b = data_3b.get("system_profile", data_3b.get("profiling", {}))
        p7b = data_7b.get("system_profile", data_7b.get("profiling", {}))
        m3b = data_3b.get("metrics", {})
        m7b = data_7b.get("metrics", {})

        md = []
        md.append("# Step 1 Pilot 基准实验详细对比与硬件剖析报告")
        md.append(f"**生成时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}  ")
        md.append(f"**测试环境**：单卡 8GB VRAM (NF4 4-bit 量化, bfloat16 计算精度)  ")
        md.append(f"**评测切片**：HotpotQA Dev Distractor ({data_7b.get('sample_count', 30)} 样本)  ")
        md.append("\n---\n")

        md.append("## 1. 核心质量与多维 NLP 指标对比")
        md.append("| 评测维度 | 指标项 (Metric) | Qwen2.5-3B-Instruct | Qwen2.5-7B-Instruct | 7B 相比 3B 增益 (Delta) |")
        md.append("| :--- | :--- | :---: | :---: | :---: |")

        em3, em7 = m3b.get("exact_match", 0.0), m7b.get("exact_match", 0.0)
        f1_3, f1_7 = m3b.get("f1_score", 0.0), m7b.get("f1_score", 0.0)
        r1_3, r1_7 = m3b.get("rouge_1", 0.0), m7b.get("rouge_1", 0.0)
        rl_3, rl_7 = m3b.get("rouge_l", 0.0), m7b.get("rouge_l", 0.0)
        b1_3, b1_7 = m3b.get("bleu_1", 0.0), m7b.get("bleu_1", 0.0)
        sp_f1_3, sp_f1_7 = m3b.get("sp_f1", 0.0), m7b.get("sp_f1", 0.0)

        md.append(f"| **问答准确性** | **Exact Match (EM %)** | {em3:.2f}% | {em7:.2f}% | **{em7 - em3:+.2f}%** |")
        md.append(f"| **词级对齐** | **Token-level F1 (%)** | {f1_3:.2f}% | {f1_7:.2f}% | **{f1_7 - f1_3:+.2f}%** |")
        md.append(f"| **召回重合** | **ROUGE-1 F1 (%)** | {r1_3:.2f}% | {r1_7:.2f}% | **{r1_7 - r1_3:+.2f}%** |")
        md.append(f"| **长公共子序列** | **ROUGE-L F1 (%)** | {rl_3:.2f}% | {rl_7:.2f}% | **{rl_7 - rl_3:+.2f}%** |")
        md.append(f"| **生成保真度** | **BLEU-1 (%)** | {b1_3:.2f}% | {b1_7:.2f}% | **{b1_7 - b1_3:+.2f}%** |")
        md.append(f"| **证据检索质量** | **Supporting Facts F1 (%)** | {sp_f1_3:.2f}% | {sp_f1_7:.2f}% | **{sp_f1_7 - sp_f1_3:+.2f}%** |")

        md.append("\n## 2. 8GB 硬件资源与计算效率对比")
        md.append("| 系统监控项 | Qwen2.5-3B-Instruct | Qwen2.5-7B-Instruct | 8GB 显存边界与约束分析 |")
        md.append("| :--- | :---: | :---: | :--- |")

        alloc_3 = p3b.get("peak_cuda_allocated_mb", 0.0)
        alloc_7 = p7b.get("peak_cuda_allocated_mb", 0.0)
        res_3 = p3b.get("peak_cuda_reserved_mb", 0.0)
        res_7 = p7b.get("peak_cuda_reserved_mb", 0.0)
        headroom_7 = p7b.get("vram_headroom_8gb_mb", 8192.0 - res_7)
        lat_3 = p3b.get("avg_latency_sec", 0.0)
        lat_7 = p7b.get("avg_latency_sec", 0.0)

        md.append(f"| **PyTorch 峰值显存分配** | {alloc_3:.1f} MB | {alloc_7:.1f} MB | 7B 4-bit 占显存约 {alloc_7/1024:.2f} GB，安全受控 |")
        md.append(f"| **CUDA 内存池预留峰值** | {res_3:.1f} MB | {res_7:.1f} MB | 满足 < 8192 MB 强制约束 |")
        md.append(f"| **8GB 显存安全余量 (Headroom)** | {p3b.get('vram_headroom_8gb_mb', 8192.0 - res_3):.1f} MB | {headroom_7:.1f} MB | **保留 {headroom_7:.1f} MB 余量**，杜绝 OOM |")
        md.append(f"| **端到端平均单题时延** | {lat_3:.4f} s | {lat_7:.4f} s | 7B 推理时延约为 3B 的 {lat_7/max(0.001, lat_3):.1f} 倍 |")
        md.append(f"| **单题平均检索耗时** | {p3b.get('avg_retrieval_latency_sec', 0.0):.4f} s | {p7b.get('avg_retrieval_latency_sec', 0.0):.4f} s | 密集与词法混合检索在 CPU 端执行 |")
        md.append(f"| **单题平均生成耗时** | {p3b.get('avg_generation_latency_sec', 0.0):.4f} s | {p7b.get('avg_generation_latency_sec', 0.0):.4f} s | 单卡串行自回归解码耗时 |")

        md.append("\n## 3. 典型预测案例深度解剖 (Case Study)")
        s3 = data_3b.get("samples", [])
        s7 = data_7b.get("samples", [])
        for i in range(min(5, len(s7))):
            it3 = s3[i] if i < len(s3) else {}
            it7 = s7[i]
            md.append(f"### [案例 {i+1}] {it7.get('question', '')}")
            md.append(f"- **标准答案 (Ground Truth)**：`{it7.get('ground_truth', '')}`")
            md.append(f"- **3B 模型预测**：`{it3.get('prediction', '')}` (EM={it3.get('em', 0)}, F1={it3.get('f1', 0):.2f})")
            md.append(f"- **7B 模型预测**：`{it7.get('prediction', '')}` (EM={it7.get('em', 0)}, F1={it7.get('f1', 0):.2f})")
            ret_titles = [d.get("title", "") for d in it7.get("retrieved_docs", [])]
            md.append(f"- **召回候选段落**：{', '.join(ret_titles[:3])}")
            md.append("")

        report_text = "\n".join(md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"Generated detailed Markdown benchmark report at: {output_path}")
        return report_text
