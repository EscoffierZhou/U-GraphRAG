"""
common.py
=========
Shared infrastructure for Phase II (WP0-WP5) experiments on a single 8GB VRAM GPU.

Provides:
1. V5_FROZEN integrity verification (SHA256SUMS.txt) and safe write guards.
2. Sequential single-model 4-bit NF4 load / teardown utilities for 8GB GPUs.
3. Atomic checkpointing (every 25 samples by default) and resumable runs.
4. Shared QA / Supporting-Facts / Calibration / Risk-Coverage (AURC / E-AURC / AUPRC)
   and statistical hypothesis testing (paired bootstrap, sign-flip, paired t, Wilcoxon, d_z).
5. Shared entity extraction, M2 interaction matrix construction (entity+JS, dense cosine,
   entity-only, random, hard-conflict pruning), and logit-based next-token probes.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

# Ensure project root is importable regardless of working directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_rag import (
    QwenNF4Generator,
    exact_match_score,
    f1_score,
    normalize_answer,
)
from src.evaluation_toolkit import (
    compute_detailed_calibration,
    compute_selective_prediction_metrics,
)
from src.m1_adaptive_retriever import compute_entropy, js_divergence
from src.m2_evidence_optimizer import (
    EvidenceCandidate,
    InteractionMatrix,
    TwoStageM2Optimizer,
    calibrate_reliability,
)
from src.m3_calibrated_decoder import (
    CalibratedDecoderConfig,
    M3CalibratedDecoder,
    compute_auroc,
    compute_risk_coverage_curve,
)
from src.u_graphrag_pipeline import compute_supporting_facts_metrics

SRC_DIR: Path = PROJECT_ROOT / "src"
DATA_DIR: Path = PROJECT_ROOT / "data"
PHASE2_DATA_DIR: Path = DATA_DIR / "phase2"
PHASE2_CACHE_DIR: Path = PHASE2_DATA_DIR / "caches"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MILE1_DIR: Path = (
    RESULTS_DIR / "MileStone1"
    if (RESULTS_DIR / "MileStone1").exists()
    else RESULTS_DIR / "MileStone1_Hajimete_RAG"
)
MILE2_DIR: Path = (
    RESULTS_DIR / "MileStone2"
    if (RESULTS_DIR / "MileStone2").exists()
    else RESULTS_DIR / "MileStone2_AblationExperiment_I"
)
V5_FROZEN_DIR: Path = (
    RESULTS_DIR / "MileStone2" / "V5_FROZEN"
    if (RESULTS_DIR / "MileStone2" / "V5_FROZEN" / "SHA256SUMS.txt").exists()
    else RESULTS_DIR / "MileStone2_AblationExperiment_I" / "V5_FROZEN"
)

WP_DIRS: Dict[str, Path] = {
    "WP1": MILE2_DIR / "WP1_M1_EU",
    "WP2": MILE2_DIR / "WP2_M2_Optimization",
    "WP3": MILE2_DIR / "WP3_M3_Calibration",
    "WP4": MILE2_DIR / "WP4_Generalization",
    "WP5": MILE2_DIR / "WP5_Robustness_Profile",
    "P0": MILE2_DIR / "P0_Closure_Fixes",
}

M1_SIGNAL_TAXONOMY: Dict[str, Dict[str, str]] = {
    "v5_proxy": {
        "canonical_name": "V5 Algebraic EU Proxy",
        "symbol": "U_EU^v5",
        "role": "Primary production M1 routing estimator (UncertaintyDecoupler.decompose)",
        "au_component": "Within-context algebraic divergence entropy (U_AU^v5)",
    },
    "paraphrase_disagreement": {
        "canonical_name": "Paraphrase-Induced Predictive Disagreement",
        "symbol": "U_EU^para",
        "legacy_alias": "bald_u_eu / bald_mi",
        "role": "Diagnostic & alternative query-perturbation estimator (not Bayesian parameter MI)",
        "au_component": "Within-Paraphrase Predictive Entropy (U_AU^para / bald_u_au)",
    },
    "raw_entropy_h_total": {
        "canonical_name": "Raw Predictive Entropy",
        "symbol": "H_total",
        "role": "Single-pass un-decomposed entropy baseline",
    },
    "self_consistency_5x": {
        "canonical_name": "5-Sample Self-Consistency Dispersion",
        "symbol": "1 - max_freq/5",
        "role": "Sampling dispersion comparator",
    },
}

BACKBONE_ALIASES: Dict[str, Tuple[str, str]] = {
    "qwen3b": ("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"),
    "qwen7b": ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    "mistral7b": ("Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
    "llama3_8b": ("Meta-Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
}

_STOPWORDS: Set[str] = {
    "the", "a", "an", "in", "on", "at", "for", "of", "to", "by", "with", "from",
    "and", "or", "when", "what", "who", "where", "which", "why", "how", "is",
    "was", "are", "were", "yes", "no", "it", "this", "that", "these", "those",
    "he", "she", "they", "his", "her", "document", "title", "context", "question",
    "answer", "references",
}


# -----------------------------------------------------------------------------
# Logging & File Integrity Guards
# -----------------------------------------------------------------------------

def setup_phase2_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Create or retrieve a formatted Phase II logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        assert_safe_output_path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        existing_paths = {getattr(h, "baseFilename", None) for h in logger.handlers}
        if str(log_file.resolve()) not in existing_paths:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(fh)
    return logger


def close_phase2_log_handlers(prefix: str = "phase2.") -> None:
    """Close and detach open FileHandlers for Phase II loggers (safe Windows temp cleanup)."""
    manager_dict = logging.Logger.manager.loggerDict
    for name, logger_obj in list(manager_dict.items()):
        if not str(name).startswith(prefix) or not isinstance(logger_obj, logging.Logger):
            continue
        for handler in list(logger_obj.handlers):
            if isinstance(handler, logging.FileHandler):
                try:
                    handler.close()
                except Exception:
                    pass
                logger_obj.removeHandler(handler)


def assert_safe_output_path(path: Path) -> Path:
    """Prevent accidental writes into immutable V5_FROZEN or MileStone1 archives."""
    resolved = path.resolve()
    for forbidden_root in (V5_FROZEN_DIR.resolve(), MILE1_DIR.resolve()):
        try:
            resolved.relative_to(forbidden_root)
            raise ValueError(
                f"Refusing to write inside protected archive directory: {resolved}"
            )
        except ValueError as err:
            if "protected archive" in str(err):
                raise
    return path


def atomic_write_json(path: Path, payload: Any) -> Path:
    """Atomically write JSON after checking protected-directory guards."""
    assert_safe_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)
    return path


def load_json_file(path: Path) -> Any:
    """Read UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_v5_frozen_checksums(v5_dir: Path = V5_FROZEN_DIR) -> Dict[str, Any]:
    """
    Verify every file listed in V5_FROZEN/SHA256SUMS.txt.
    Raises RuntimeError if SHA256SUMS.txt is missing or any artifact fails verification.
    """
    manifest_path = v5_dir / "SHA256SUMS.txt"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing frozen V5 manifest: {manifest_path}")

    checked = 0
    mismatches: List[str] = []
    missing: List[str] = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            parts_3 = raw.split(None, 2)
            if len(parts_3) >= 3 and parts_3[1].isdigit():
                expected_hex = parts_3[0].strip().lower()
                rel_path = parts_3[2].strip()
            else:
                parts_2 = raw.split(None, 1)
                if len(parts_2) != 2:
                    continue
                expected_hex = parts_2[0].strip().lower()
                rel_path = parts_2[1].strip()
            target = v5_dir / Path(rel_path.replace("\\", "/"))
            if not target.exists():
                missing.append(rel_path)
                continue
            hasher = hashlib.sha256()
            with open(target, "rb") as tf:
                while True:
                    chunk = tf.read(1 << 20)
                    if not chunk:
                        break
                    hasher.update(chunk)
            actual_hex = hasher.hexdigest().lower()
            if actual_hex != expected_hex:
                mismatches.append(f"{rel_path}: expected={expected_hex}, got={actual_hex}")
            checked += 1

    if missing or mismatches or checked == 0:
        raise RuntimeError(
            f"V5_FROZEN integrity check failed (checked={checked}, "
            f"missing={missing[:5]}, mismatches={mismatches[:5]})"
        )
    return {"status": "ok", "verified_count": checked}


def load_v5_balanced_params(preset: str = "balanced_optimal") -> Dict[str, Any]:
    """Load frozen V5 grid-searched parameter set from V5_FROZEN."""
    grid_path = V5_FROZEN_DIR / "results" / "grid_search_results.json"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing frozen grid search file: {grid_path}")
    data = load_json_file(grid_path)
    rec = data.get("recommended_parameter_sets", {})
    if preset not in rec:
        raise KeyError(f"Preset '{preset}' not found in {list(rec.keys())}")
    return dict(rec[preset])


# -----------------------------------------------------------------------------
# Checkpoint Manager
# -----------------------------------------------------------------------------

class Phase2CheckpointManager:
    """
    Resumable JSON checkpoint manager that flushes atomically every `flush_every` writes.
    """

    def __init__(self, checkpoint_path: Path, flush_every: int = 25) -> None:
        self.checkpoint_path = assert_safe_output_path(checkpoint_path)
        self.flush_every = max(1, int(flush_every))
        self.records: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._dirty_count: int = 0
        self.load()

    def load(self) -> None:
        if self.checkpoint_path.exists():
            try:
                data = load_json_file(self.checkpoint_path)
                if isinstance(data, dict):
                    self.records = dict(data.get("records", {}))
                    self.metadata = dict(data.get("metadata", {}))
            except Exception:
                self.records = {}
                self.metadata = {}

    def has(self, key: str) -> bool:
        return str(key) in self.records

    def get(self, key: str, default: Any = None) -> Any:
        return self.records.get(str(key), default)

    def record(self, key: str, value: Any) -> None:
        self.records[str(key)] = value
        self._dirty_count += 1
        if self._dirty_count >= self.flush_every:
            self.flush(force=True)

    def flush(self, force: bool = True) -> None:
        if not force and self._dirty_count == 0:
            return
        was_dirty = self._dirty_count
        atomic_write_json(
            self.checkpoint_path,
            {"metadata": self.metadata, "records": self.records},
        )
        self._dirty_count = 0
        if was_dirty > 0:
            print(
                f"[ckpt] {self.checkpoint_path.name}: {len(self.records)} records saved",
                flush=True,
            )


# -----------------------------------------------------------------------------
# 8GB VRAM Single-Model Sequential Lifecycle
# -----------------------------------------------------------------------------

def resolve_model_path(model_key_or_path: str) -> str:
    """Resolve short backbone key ('qwen3b', 'qwen7b', 'mistral7b', 'llama3_8b') or explicit path."""
    key = str(model_key_or_path).strip()
    low = key.lower()
    if low in BACKBONE_ALIASES:
        local_name, hf_repo = BACKBONE_ALIASES[low]
        cand_roots = [MODELS_DIR, Path("/root/autodl-tmp/models"), Path("/tmp/models")]
        custom_root = os.environ.get("UGRAPHRAG_MODELS_DIR")
        if custom_root:
            cand_roots.insert(0, Path(custom_root))
        for base_dir in cand_roots:
            local_dir = base_dir / local_name
            if local_dir.exists() and (any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))):
                return str(local_dir)
        return hf_repo

    candidate = Path(key)
    if candidate.exists():
        return str(candidate)
    in_models = MODELS_DIR / key
    if in_models.exists():
        return str(in_models)
    return key


def get_vram_snapshot_mb() -> Dict[str, float]:
    """Read PyTorch CUDA allocated/reserved peak memory stats in MB."""
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0,
        }
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
    }


def reset_cuda_peak() -> None:
    """Reset PyTorch CUDA peak stats and empty cache safely."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def teardown_generator(generator: Any) -> None:
    """Explicitly release LLM weights from 8GB GPU memory."""
    if generator is not None:
        if hasattr(generator, "model"):
            try:
                generator.model = None
            except Exception:
                pass
        if hasattr(generator, "tokenizer"):
            try:
                generator.tokenizer = None
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def sequential_load_generator(
    model_key_or_path: str,
    dry_run: bool = False,
    max_new_tokens: int = 48,
    quantization: str = "4bit",
) -> QwenNF4Generator:
    """Load one 4-bit NF4 (or bf16) generator cleanly after flushing GC and CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    resolved = resolve_model_path(model_key_or_path)
    device = "cuda" if torch.cuda.is_available() and not dry_run else "cpu"
    return QwenNF4Generator(
        model_name_or_path=resolved,
        dry_run=dry_run,
        device=device,
        max_new_tokens=max_new_tokens,
        quantization=quantization,
    )


# -----------------------------------------------------------------------------
# Answer Evaluation, Supporting Facts, Calibration, Risk-Coverage & Statistics
# -----------------------------------------------------------------------------

def extract_gold_answers(sample: Dict[str, Any]) -> List[str]:
    """Extract normalized list of valid gold answer strings from heterogeneous QA datasets."""
    golds: List[str] = []
    for key in ("possible_answers", "answers"):
        val = sample.get(key)
        if isinstance(val, list):
            for item in val:
                s = str(item).strip()
                if s and s not in golds:
                    golds.append(s)
        elif isinstance(val, str) and val.strip():
            s = val.strip()
            if s not in golds:
                golds.append(s)
    for key in ("ground_truth", "answer"):
        val = sample.get(key)
        if val is not None:
            s = str(val).strip()
            if s and s not in golds:
                golds.append(s)
    return golds or [""]


def extract_gold_titles(sample: Dict[str, Any]) -> Set[str]:
    """Extract normalized gold supporting-fact document titles/ids."""
    if isinstance(sample.get("gold_titles"), list):
        return {str(t).strip() for t in sample["gold_titles"] if str(t).strip()}

    sf = sample.get("supporting_facts")
    titles: Set[str] = set()
    if isinstance(sf, dict):
        for t in sf.get("title", []) or []:
            if str(t).strip():
                titles.add(str(t).strip())
    elif isinstance(sf, list):
        for item in sf:
            if isinstance(item, (list, tuple)) and item:
                titles.add(str(item[0]).strip())
            elif isinstance(item, dict) and item.get("title"):
                titles.add(str(item["title"]).strip())
    return titles


def evaluate_qa_prediction(
    prediction: str,
    gold_candidates: Sequence[str],
    is_abstained: bool = False,
) -> Dict[str, float]:
    """
    Compute raw EM/F1 (un-abstained candidate quality over gold candidates)
    alongside effective EM/F1 under abstention (0.0 when abstained).
    Preserving raw EM/F1 even when `is_abstained=True` is essential for measuring
    Harmful Abstention Rate and Valid Abstention Precision without metric distortion.
    """
    golds = [str(g) for g in gold_candidates if str(g).strip()] or [""]
    if not str(prediction).strip():
        raw_em = 0.0
        raw_f1 = 0.0
    else:
        raw_em = float(max(exact_match_score(prediction, g) for g in golds))
        raw_f1 = float(max(f1_score(prediction, g) for g in golds))

    eff_em = 0.0 if is_abstained else raw_em
    eff_f1 = 0.0 if is_abstained else raw_f1
    return {
        "em": raw_em,
        "f1": raw_f1,
        "effective_em": eff_em,
        "effective_f1": eff_f1,
    }


def trapz_area(y: Sequence[float], x: Sequence[float]) -> float:
    """NumPy 1.x / 2.x safe trapezoidal numerical integration."""
    y_arr = np.asarray(list(y), dtype=float)
    x_arr = np.asarray(list(x), dtype=float)
    if len(x_arr) < 2:
        return 0.0
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y_arr, x_arr))


def compute_auprc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute deterministic step-function Area Under Precision-Recall Curve."""
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0 or len(y) == 0:
        return 0.0

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    ranks = np.arange(1, len(y_sorted) + 1)
    precisions = tp / ranks
    recalls = tp / n_pos

    rec_pts = np.concatenate([[0.0], recalls])
    prec_pts = np.concatenate([[1.0], precisions])
    return float(np.sum((rec_pts[1:] - rec_pts[:-1]) * prec_pts[1:]))


def compute_aurc_metrics(
    confidences: Sequence[float],
    accuracies: Sequence[float],
    n_thresholds: int = 50,
) -> Dict[str, Any]:
    """
    Compute Area Under Risk-Coverage curve (AURC), Oracle AURC, Excess-AURC (E-AURC),
    and Coverage at target selective risk caps (5%, 10%, 20%).
    """
    n = len(confidences)
    if n == 0 or len(accuracies) != n:
        return {
            "aurc": 0.0,
            "oracle_aurc": 0.0,
            "e_aurc": 0.0,
            "coverage_at_risk_05": 0.0,
            "coverage_at_risk_10": 0.0,
            "coverage_at_risk_20": 0.0,
            "curve_points": [],
        }

    # Sort descending by confidence so coverage increases monotonically from 1/n -> 1.0
    pairs = sorted(zip(confidences, accuracies), key=lambda p: p[0], reverse=True)
    cum_err = 0.0
    covs: List[float] = [0.0]
    risks: List[float] = [0.0]
    for k, (_, acc) in enumerate(pairs, start=1):
        cum_err += max(0.0, 1.0 - float(acc))
        covs.append(k / n)
        risks.append(cum_err / k)

    aurc = trapz_area(risks, covs)

    # Oracle ordering (all correct answers ranked ahead of all wrong answers)
    oracle_accs = sorted([float(a) for a in accuracies], reverse=True)
    cum_err_o = 0.0
    o_covs: List[float] = [0.0]
    o_risks: List[float] = [0.0]
    for k, acc in enumerate(oracle_accs, start=1):
        cum_err_o += max(0.0, 1.0 - acc)
        o_covs.append(k / n)
        o_risks.append(cum_err_o / k)
    oracle_aurc = trapz_area(o_risks, o_covs)

    # Also compute binned RC points via src helper for compatibility
    rc_curve = compute_risk_coverage_curve(list(confidences), list(accuracies), n_thresholds=n_thresholds)

    def _max_cov_below_risk(max_risk: float) -> float:
        feasible = [c for c, r in zip(covs[1:], risks[1:]) if r <= max_risk + 1e-12]
        return float(max(feasible)) if feasible else 0.0

    return {
        "aurc": round(float(aurc), 6),
        "oracle_aurc": round(float(oracle_aurc), 6),
        "e_aurc": round(float(max(0.0, aurc - oracle_aurc)), 6),
        "coverage_at_risk_05": round(_max_cov_below_risk(0.05) * 100.0, 2),
        "coverage_at_risk_10": round(_max_cov_below_risk(0.10) * 100.0, 2),
        "coverage_at_risk_20": round(_max_cov_below_risk(0.20) * 100.0, 2),
        "curve_points": rc_curve.get("curve_points", []),
    }


def compute_calibration_and_risk_suite(
    confidences: Sequence[float],
    accuracies: Sequence[float],
    abstained_flags: Optional[Sequence[bool]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Unified Phase II calibration, AUROC, AUPRC, AURC/E-AURC and abstention suite."""
    confs = [float(max(0.0, min(1.0, c))) for c in confidences]
    accs = [float(a) for a in accuracies]
    bin_labels = [1 if a >= 0.5 else 0 for a in accs]
    n = len(confs)

    cal = compute_detailed_calibration(confs, accs, n_bins=n_bins)
    auroc_correct = float(compute_auroc(confs, bin_labels))
    auroc_error = float(compute_auroc([1.0 - c for c in confs], [1 - y for y in bin_labels]))
    auprc_error = float(compute_auprc([1 - y for y in bin_labels], [1.0 - c for c in confs]))
    aurc_res = compute_aurc_metrics(confs, accs)

    if abstained_flags is None:
        abstained_flags = [False] * n
    sel = compute_selective_prediction_metrics(list(abstained_flags), accs)

    abstained_indices = [i for i, ab in enumerate(abstained_flags) if ab]
    if abstained_indices:
        n_ab = len(abstained_indices)
        harmful_ab = sum(1 for i in abstained_indices if accs[i] >= 0.5)
        valid_ab = n_ab - harmful_ab
        harmful_rate_of_abstained_pct = (harmful_ab / n_ab) * 100.0
        valid_rate_of_abstained_pct = (valid_ab / n_ab) * 100.0
        harmful_rate_overall_pct = (harmful_ab / n) * 100.0 if n > 0 else 0.0
        valid_rate_overall_pct = (valid_ab / n) * 100.0 if n > 0 else 0.0
    else:
        harmful_rate_of_abstained_pct = 0.0
        valid_rate_of_abstained_pct = 0.0
        harmful_rate_overall_pct = 0.0
        valid_rate_overall_pct = 0.0

    cov_val = float(sel["coverage"])
    ab_rate_val = max(0.0, 1.0 - cov_val)
    sel_acc_val = float(sel["effective_accuracy"])
    sel_risk_val = float(sel["selective_risk"])
    base_risk_val = float(sum(1.0 - a for a in accs) / n) if n else 0.0
    risk_red_val = (base_risk_val - sel_risk_val) if ab_rate_val > 0.0 else 0.0

    return {
        "n_samples": n,
        "mean_confidence": round(float(sum(confs) / n), 4) if n else 0.0,
        "mean_accuracy": round(float(sum(accs) / n), 4) if n else 0.0,
        "ece": round(float(cal["ece"]), 4),
        "mce": round(float(cal["mce"]), 4),
        "brier_score": round(float(cal["brier_score"]), 4),
        "auroc_confidence": round(auroc_correct, 4),
        "auroc_error": round(auroc_error, 4),
        "auprc_error": round(auprc_error, 4),
        "aurc": aurc_res["aurc"],
        "oracle_aurc": aurc_res["oracle_aurc"],
        "e_aurc": aurc_res["e_aurc"],
        "coverage_at_risk_05": aurc_res["coverage_at_risk_05"],
        "coverage_at_risk_10": aurc_res["coverage_at_risk_10"],
        "coverage_at_risk_20": aurc_res["coverage_at_risk_20"],
        "selective_coverage_pct": round(cov_val * 100.0, 2),
        "abstention_rate_pct": round(ab_rate_val * 100.0, 2),
        "selective_accuracy_pct": round(sel_acc_val * 100.0, 2),
        "selective_risk_pct": round(sel_risk_val * 100.0, 2),
        "risk_reduction_pct": round(risk_red_val * 100.0, 2),
        "valid_abstention_precision_pct": round(valid_rate_of_abstained_pct, 2),
        "harmful_abstention_rate_pct": round(harmful_rate_of_abstained_pct, 2),
        "valid_abstention_overall_pct": round(valid_rate_overall_pct, 2),
        "harmful_abstention_overall_pct": round(harmful_rate_overall_pct, 2),
        "bin_breakdown": cal.get("bin_breakdown", []),
        "curve_points": aurc_res["curve_points"],
    }


def paired_bootstrap_with_tests(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute paired statistical comparison for metric differences (A - B):
      - Mean A, Mean B, Mean Delta (A - B)
      - 95% paired bootstrap confidence interval
      - Two-sided Monte Carlo paired permutation (sign-flip) p-value
      - Paired t-test p-value and Wilcoxon signed-rank p-value (when non-degenerate)
      - Paired effect size Cohen's d_z
    """
    a = np.asarray(list(scores_a), dtype=float)
    b = np.asarray(list(scores_b), dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Mismatched sample lengths: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return {
            "n": 0,
            "mean_a": 0.0,
            "mean_b": 0.0,
            "mean_delta": 0.0,
            "ci_95_low": 0.0,
            "ci_95_high": 0.0,
            "sign_flip_p_value": 1.0,
            "paired_t_stat": 0.0,
            "paired_t_p_value": 1.0,
            "wilcoxon_p_value": 1.0,
            "cohens_dz": 0.0,
        }

    diff = a - b
    obs_delta = float(np.mean(diff))
    rng = np.random.default_rng(seed)

    # 1. Paired bootstrap 95% CI
    idx_mat = rng.integers(0, n, size=(n_boot, n))
    boot_deltas = diff[idx_mat].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])

    # 2. Paired permutation sign-flip p-value (two-sided)
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    perm_deltas = (diff[None, :] * signs).mean(axis=1)
    p_perm = float((np.sum(np.abs(perm_deltas) >= abs(obs_delta) - 1e-12) + 1) / (n_boot + 1))

    # 3. Effect size Cohen's d_z
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    cohens_dz = float(obs_delta / sd) if sd > 1e-12 else 0.0
    t_stat = float(obs_delta / (sd / math.sqrt(n))) if (n > 1 and sd > 1e-12) else 0.0

    # 4. Scipy exact parametric/nonparametric p-values when installed
    t_p_val = p_perm
    wilcoxon_p_val = p_perm
    try:
        from scipy import stats as scipy_stats

        if sd > 1e-12 and n > 1:
            t_res = scipy_stats.ttest_rel(a, b)
            t_p_val = float(t_res.pvalue)
        elif abs(obs_delta) < 1e-12:
            t_p_val = 1.0

        nz = diff[np.abs(diff) > 1e-12]
        if len(nz) >= 1:
            w_res = scipy_stats.wilcoxon(nz, zero_method="wilcox")
            wilcoxon_p_val = float(w_res.pvalue)
        else:
            wilcoxon_p_val = 1.0
    except Exception:
        # Normal approximation fallback for paired t-test when scipy unavailable
        if n > 1 and sd > 1e-12:
            z = abs(t_stat)
            t_p_val = float(math.erfc(z / math.sqrt(2.0)))

    return {
        "n": int(n),
        "mean_a": round(float(np.mean(a)), 6),
        "mean_b": round(float(np.mean(b)), 6),
        "mean_delta": round(obs_delta, 6),
        "ci_95_low": round(float(ci_low), 6),
        "ci_95_high": round(float(ci_high), 6),
        "sign_flip_p_value": round(p_perm, 6),
        "paired_t_stat": round(t_stat, 4),
        "paired_t_p_value": round(float(t_p_val), 6),
        "wilcoxon_p_value": round(float(wilcoxon_p_val), 6),
        "cohens_dz": round(cohens_dz, 4),
    }


def adjust_p_values_multiple_comparisons(p_values: Sequence[float]) -> Dict[str, List[float]]:
    """
    Compute Holm-Bonferroni step-down adjusted p-values (FWER control) and
    Benjamini-Hochberg adjusted q-values (FDR control) for a family of tests.
    """
    m = len(p_values)
    if m == 0:
        return {"holm_p_values": [], "bh_fdr_q_values": []}

    indexed = sorted(enumerate(float(max(0.0, min(1.0, p))) for p in p_values), key=lambda x: x[1])

    # Holm-Bonferroni step-down (FWER)
    holm_sorted: List[float] = []
    running_max = 0.0
    for rank_0, (_, p) in enumerate(indexed):
        adj = min(1.0, (m - rank_0) * p)
        running_max = max(running_max, adj)
        holm_sorted.append(running_max)

    holm_out = [0.0] * m
    for (orig_idx, _), adj_val in zip(indexed, holm_sorted):
        holm_out[orig_idx] = round(float(adj_val), 6)

    # Benjamini-Hochberg step-up (FDR)
    bh_sorted = [0.0] * m
    running_min = 1.0
    for rev_rank_0 in range(m - 1, -1, -1):
        _, p = indexed[rev_rank_0]
        adj = min(1.0, (m / (rev_rank_0 + 1)) * p)
        running_min = min(running_min, adj)
        bh_sorted[rev_rank_0] = running_min

    bh_out = [0.0] * m
    for (orig_idx, _), q_val in zip(indexed, bh_sorted):
        bh_out[orig_idx] = round(float(q_val), 6)

    return {"holm_p_values": holm_out, "bh_fdr_q_values": bh_out}


def apply_multiple_comparison_correction(
    tests_map: Dict[str, Dict[str, Any]],
    p_field: str = "paired_t_p_value",
) -> Dict[str, Dict[str, Any]]:
    """
    Attach `holm_p_value` and `bh_fdr_q_value` across a family of paired tests in-place/copy.
    """
    keys = [k for k, v in tests_map.items() if isinstance(v, dict) and p_field in v]
    raw_ps = [float(tests_map[k][p_field]) for k in keys]
    adj = adjust_p_values_multiple_comparisons(raw_ps)
    out = {k: dict(v) for k, v in tests_map.items()}
    for k, hp, bq in zip(keys, adj["holm_p_values"], adj["bh_fdr_q_values"]):
        out[k] = {
            **out[k],
            f"{p_field}_holm": hp,
            f"{p_field}_bh_fdr": bq,
        }
    return out


# -----------------------------------------------------------------------------
# Shared Text / Entity / Candidate / Interaction Utilities
# -----------------------------------------------------------------------------

def extract_entities(text: str, title: str = "") -> Set[str]:
    """Extract normalized named entities, title tokens, and 4-digit years."""
    ents: Set[str] = set()
    if title:
        t_norm = title.strip().lower()
        if t_norm and t_norm not in _STOPWORDS:
            ents.add(t_norm)
        for tok in re.findall(r"[A-Za-z0-9]{4,}", t_norm):
            if tok not in _STOPWORDS:
                ents.add(tok)

    for phrase in re.findall(r"\b[A-Z][a-zA-Z0-9'-]+(?:\s+[A-Z][a-zA-Z0-9'-]+){0,3}\b", text or ""):
        low = phrase.strip().lower()
        if len(low) >= 3 and low not in _STOPWORDS:
            ents.add(low)
            for sub in low.split():
                if len(sub) >= 4 and sub not in _STOPWORDS:
                    ents.add(sub)

    for yr in re.findall(r"\b(?:1[0-9]{3}|20[0-2][0-9])\b", text or ""):
        ents.add(yr)
    return ents


def compute_token_count(text: str) -> int:
    """Fast deterministic token approximation (whitespace + punctuation words)."""
    return max(1, len(re.findall(r"\S+", text or "")))


def text_shingles(text: str, k: int = 3) -> Set[str]:
    """Build k-word shingles for MinHash/Jaccard overlap checks."""
    toks = re.findall(r"\w+", (text or "").lower())
    if len(toks) < k:
        return set(toks)
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def jaccard_sim(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity in [0, 1]."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    return float(len(a & b) / union) if union > 0 else 0.0


def build_prompt(
    question: str,
    docs: Sequence[Dict[str, Any]],
    generator: Optional[QwenNF4Generator] = None,
    reliability_weighted: bool = False,
) -> str:
    """
    Build QA prompt cleanly, optionally prefixing M2 reliability weights into each doc header.
    """
    formatted_docs: List[Dict[str, Any]] = []
    for idx, d in enumerate(docs):
        d_copy = dict(d)
        if reliability_weighted:
            rel = float(d.get("calibrated_reliability", d.get("score", 0.5)))
            title = str(d.get("title", d.get("doc_id", f"Doc {idx + 1}")))
            d_copy["title"] = f"{title} | Reliability={rel:.2f}"
        formatted_docs.append(d_copy)

    if generator is not None and hasattr(generator, "build_prompt"):
        return generator.build_prompt(question, formatted_docs)

    ctx_texts = "\n".join(
        f"[{i + 1}] {d.get('title', 'Doc')}: {d.get('text', '')}"
        for i, d in enumerate(formatted_docs)
    )
    if ctx_texts:
        return (
            "You are a precise extractive QA assistant. Based strictly on the provided references, "
            "output ONLY the exact short answer (e.g., 'yes', 'no', an entity name, number, or date). "
            "Do not repeat the references or explain.\n\n"
            f"References:\n{ctx_texts}\n\n"
            f"Question: {question}\n"
            "Short Answer:"
        )
    return (
        "Output ONLY the exact short answer with no explanation.\n\n"
        f"Question: {question}\n"
        "Short Answer:"
    )


def dict_docs_to_evidence_candidates(
    docs: Sequence[Dict[str, Any]],
    reliability_offset: float = 0.0,
    gamma_1: float = 0.8,
    gamma_2: float = 0.2,
    calibrate: bool = True,
) -> List[EvidenceCandidate]:
    """Convert raw cached/retrieved dicts to calibrated EvidenceCandidate objects."""
    candidates: List[EvidenceCandidate] = []
    for i, doc in enumerate(docs):
        doc_id = str(doc.get("doc_id", doc.get("id", doc.get("title", f"doc_{i}"))))
        title = str(doc.get("title", doc_id))
        text = str(doc.get("text", doc.get("content", "")))
        raw_score = float(doc.get("score", doc.get("fusion_score", max(0.1, 1.0 - 0.05 * i))))
        au = float(doc.get("aleatoric_uncertainty", 0.0))
        er = float(doc.get("epistemic_reduction", 0.0))
        raw_ents = doc.get("entities") or []
        ents = {str(e).strip().lower() for e in raw_ents if str(e).strip()}
        ents |= extract_entities(text, title=title)
        meta = dict(doc.get("metadata", {}))
        meta.setdefault("title", title)
        meta["original_idx"] = i
        meta["token_count"] = int(doc.get("token_count", compute_token_count(text)))

        candidates.append(
            EvidenceCandidate(
                doc_id=doc_id,
                text=text,
                score=raw_score,
                entities=ents,
                aleatoric_uncertainty=au,
                epistemic_reduction=er,
                metadata=meta,
            )
        )

    if calibrate and candidates:
        calibrate_reliability(candidates, method="minmax", gamma_1=gamma_1, gamma_2=gamma_2)
        if abs(reliability_offset) > 1e-9:
            for c in candidates:
                c.calibrated_reliability = max(
                    0.0, min(1.0, float(c.calibrated_reliability) + float(reliability_offset))
                )
    return candidates


def _tfidf_cosine_matrix(texts: Sequence[str]) -> np.ndarray:
    """Pure CPU cosine similarity matrix for dense-graph control in WP2."""
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        mat = TfidfVectorizer().fit_transform(list(texts))
        return np.asarray(cosine_similarity(mat), dtype=float)
    except Exception:
        # Lightweight token-set cosine fallback
        sets = [set(re.findall(r"\w+", t.lower())) for t in texts]
        out = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                denom = math.sqrt(max(1, len(sets[i]) * len(sets[j])))
                val = len(sets[i] & sets[j]) / denom
                out[i, j] = out[j, i] = val
        return out


def build_interaction_matrix_for_docs(
    candidates: List[EvidenceCandidate],
    per_doc_dists: Optional[Dict[str, Dict[str, float]]] = None,
    use_minhash_redundancy: bool = True,
    use_js_conflict: bool = True,
    graph_mode: str = "entity_js",
    rng: Optional[random.Random] = None,
    conflict_prune_threshold: Optional[float] = None,
    overlap_conflict_shingle_gate: Optional[float] = None,
) -> Tuple[InteractionMatrix, List[EvidenceCandidate]]:
    """
    Construct M2 InteractionMatrix across WP2 / P0-1 ablation modes:
      - 'entity_js': V5-style entity bridge complementarity + 3-shingle MinHash redundancy + JS conflict
      - 'no_graph': uniform complementarity & zero graph interactions (for A1)
      - 'entity_cooccur': pure entity overlap without JS or shingle adjustments
      - 'dense_cosine': TF-IDF / dense vector cosine topology control
      - 'random': uniform random edge topology control (matched range)
    Optionally applies hard conflict threshold pruning before optimization or
    overlap_conflict_shingle_gate (if set > 0) for P0-1 disentanglement diagnostics.
    """
    active_candidates = list(candidates)
    n = len(active_candidates)
    dists = per_doc_dists or {}

    if graph_mode == "no_graph":
        return InteractionMatrix(n), active_candidates

    shingle_map = {c.doc_id: text_shingles(c.text, k=3) for c in active_candidates}
    cosine_mat = _tfidf_cosine_matrix([c.text for c in active_candidates]) if graph_mode == "dense_cosine" else None
    local_rng = rng or random.Random(42)

    def _pair_metrics(ci: EvidenceCandidate, cj: EvidenceCandidate, i: int, j: int) -> Tuple[float, float, float, float]:
        ent_jacc = jaccard_sim(ci.entities, cj.entities)
        red = jaccard_sim(shingle_map[ci.doc_id], shingle_map[cj.doc_id]) if use_minhash_redundancy else 0.0
        raw_red = jaccard_sim(shingle_map[ci.doc_id], shingle_map[cj.doc_id])

        conf = 0.0
        if use_js_conflict and ci.doc_id in dists and cj.doc_id in dists:
            raw_js = float(js_divergence(dists[ci.doc_id], dists[cj.doc_id]))
            if overlap_conflict_shingle_gate is not None and overlap_conflict_shingle_gate > 0.0:
                # Diagnostic overlap-conditioned conflict: suppress conflict penalty between disjoint bridge pages
                align_gate = 1.0 if raw_red >= float(overlap_conflict_shingle_gate) else 0.0
                conf = align_gate * (raw_js * (0.25 + 0.50 * ent_jacc) + 0.12 * raw_red)
            else:
                conf = raw_js * (0.08 + 0.35 * ent_jacc) if ent_jacc > 0.0 else raw_js * 0.05

        if graph_mode == "dense_cosine":
            sim = float(cosine_mat[i, j]) if cosine_mat is not None else 0.0
            comp = sim
            weight = sim - 0.35 * conf
        elif graph_mode == "entity_cooccur":
            comp = ent_jacc
            weight = ent_jacc
        elif graph_mode == "random":
            comp = local_rng.uniform(0.0, 0.6)
            weight = local_rng.uniform(-0.1, 0.7)
        else:
            comp = ent_jacc
            weight = comp + 0.15 * red - 0.35 * conf
        return float(comp), float(red), float(conf), float(weight)

    if conflict_prune_threshold is not None and len(active_candidates) > 1:
        # Remove lower-reliability member of strongly conflicting pairs above threshold
        kept = list(active_candidates)
        dropped: Set[int] = set()
        for i in range(len(kept)):
            if i in dropped:
                continue
            for j in range(i + 1, len(kept)):
                if j in dropped:
                    continue
                _, _, conf, _ = _pair_metrics(kept[i], kept[j], i, j)
                if conf >= float(conflict_prune_threshold):
                    if kept[i].calibrated_reliability >= kept[j].calibrated_reliability:
                        dropped.add(j)
                    else:
                        dropped.add(i)
                        break
        if len(dropped) < len(kept):
            active_candidates = [c for idx, c in enumerate(kept) if idx not in dropped]
            if graph_mode == "dense_cosine":
                cosine_mat = _tfidf_cosine_matrix([c.text for c in active_candidates])

    m = len(active_candidates)
    mat = InteractionMatrix(m)
    for i in range(m):
        for j in range(i + 1, m):
            comp, red, conf, weight = _pair_metrics(active_candidates[i], active_candidates[j], i, j)
            mat.set_pair(i, j, comp=comp, red=red, conf=conf, weight=weight)
    return mat, active_candidates


# -----------------------------------------------------------------------------
# Shared Neural Next-Token Distribution Probe (GPU + Dry-Run Deterministic)
# -----------------------------------------------------------------------------

def _dryrun_token_distribution(
    question: str,
    doc_texts: Sequence[str],
    variant_seed: str = "",
) -> Dict[str, Any]:
    """
    Deterministic, content-sensitive probability distribution for CPU dry-run verification.
    Reacts realistically to missing docs, multi-doc agreement, numeric/year conflict, and paraphrases.
    """
    q_terms = {w.lower() for w in re.findall(r"\b[a-zA-Z0-9]{3,}\b", question) if w.lower() not in _STOPWORDS}
    joined = " ".join(doc_texts)
    years = re.findall(r"\b(?:1[0-9]{3}|20[0-2][0-9])\b", joined)
    distinct_years = len(set(years))

    if not doc_texts:
        # Closed-book probe: uncertainty grows for rare / long / obscure queries
        q_hash = int(hashlib.md5(f"{question}|{variant_seed}".encode("utf-8")).hexdigest()[:8], 16)
        sharpness = 0.6 + 1.8 * ((q_hash % 100) / 99.0)
    else:
        doc_terms = {w.lower() for w in re.findall(r"\b[a-zA-Z0-9]{3,}\b", joined) if w.lower() not in _STOPWORDS}
        cov = len(q_terms & doc_terms) / max(1, len(q_terms))
        # High year conflict flattens distribution; strong coverage sharpens it
        conflict_penalty = max(0.0, 0.35 * max(0, distinct_years - 2))
        sharpness = max(0.35, 0.7 + 2.8 * cov - conflict_penalty)

    base_seed = int(hashlib.sha256(f"{question}||{joined[:400]}||{variant_seed}".encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(base_seed)
    vocab = [f"tok_{i}" for i in range(12)]
    logits = [rng.gauss(0.0, 0.35) for _ in vocab]
    # Anchor dominant answer token from first entity or year present in context
    anchor = years[0] if years else (sorted(q_terms)[0] if q_terms else "tok_0")
    dom_idx = int(hashlib.md5(anchor.encode("utf-8")).hexdigest()[:4], 16) % len(vocab)
    logits[dom_idx] += sharpness
    if distinct_years >= 2:
        sec_idx = (dom_idx + 1) % len(vocab)
        logits[sec_idx] += sharpness * 0.85

    max_l = max(logits)
    exps = [math.exp(x - max_l) for x in logits]
    z = sum(exps)
    probs = [e / z for e in exps]
    dist = {tok: float(p) for tok, p in zip(vocab, probs)}
    ent = float(compute_entropy(dist))
    max_p = float(max(probs))
    return {
        "dist": dist,
        "entropy": ent,
        "norm_entropy": ent / math.log(len(vocab)),
        "max_prob": max_p,
    }


def compute_next_token_distribution(
    generator: Optional[QwenNF4Generator],
    question: str,
    doc_texts: Sequence[str],
    top_k_vocab: int = 48,
    variant_seed: str = "",
) -> Dict[str, Any]:
    """
    Compute first-answer-token softmax distribution from `generator.model` (or deterministic dry-run).
    Safe on 8GB GPU: single forward pass under `torch.no_grad()` without KV-cache growth.
    """
    if generator is None or getattr(generator, "dry_run", True) or getattr(generator, "model", None) is None:
        return _dryrun_token_distribution(question, doc_texts, variant_seed=variant_seed)

    docs_payload = [{"title": f"Doc_{i + 1}", "text": t} for i, t in enumerate(doc_texts)]
    prompt = build_prompt(question, docs_payload, generator=generator, reliability_weighted=False)
    inputs = generator.tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=2048
    ).to(generator.model.device)
    with torch.no_grad():
        try:
            out = generator.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        except TypeError:
            out = generator.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False)
        logits = out.logits[0, -1, :].float()
        probs_full = torch.softmax(logits, dim=-1)
        k = min(int(top_k_vocab), probs_full.numel())
        top_probs, top_ids = torch.topk(probs_full, k=k)
        # Compute full-vocab entropy safely guarding 0 * (-inf) against NaN
        log_probs_full = torch.log_softmax(logits, dim=-1)
        entropy_terms = torch.nan_to_num(probs_full * log_probs_full, nan=0.0, neginf=0.0, posinf=0.0)
        entropy_val = float(max(0.0, -entropy_terms.sum().item()))
        max_prob_val = float(top_probs[0].item())
        norm_ent = entropy_val / max(1e-9, math.log(float(probs_full.numel())))

    top_probs_list = top_probs.detach().cpu().tolist()
    top_ids_list = top_ids.detach().cpu().tolist()
    norm_sum = max(1e-12, float(sum(top_probs_list)))
    dist = {str(int(tid)): float(p / norm_sum) for tid, p in zip(top_ids_list, top_probs_list)}

    del inputs, out, logits, probs_full, log_probs_full, entropy_terms, top_probs, top_ids
    if torch.cuda.is_available() and (torch.cuda.memory_reserved() / (1024 ** 2)) > 5150.0:
        torch.cuda.empty_cache()
    return {
        "dist": dist,
        "entropy": float(entropy_val),
        "norm_entropy": float(norm_ent),
        "max_prob": float(max_prob_val),
    }


def select_m2_evidence_from_pool(
    question: str,
    pool: Sequence[Dict[str, Any]],
    per_doc_dists: Optional[Dict[str, Dict[str, float]]],
    params: Dict[str, Any],
    pool_k: int = 8,
    max_k: int = 5,
    algorithm: str = "exact_enum",
    recency_aware: bool = False,
    conflict_prune_threshold: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Shared M2-A + M2-B selection & organization helper used consistently across WP3, WP4, and WP5.
    Returns `(ordered_m3_ready_docs, telemetry_dict)`.
    """
    trimmed = list(pool)[:pool_k]
    if not trimmed:
        return [], {"evaluated_subsets": 0, "m2_elapsed_ms": 0.0, "pruned_pool_len": 0}

    g1 = float(params.get("m2_gamma_1", 0.8))
    g2 = float(params.get("m2_gamma_2", 0.2))
    lam_div = float(params.get("m2_lambda_div", 0.4))
    rel_off = float(params.get("m2_reliability_offset", 0.22))
    min_k = int(params.get("m2_min_docs", 2))

    raw_map = {str(d.get("doc_id", d.get("id", d.get("title", "")))): d for d in trimmed}
    cands = dict_docs_to_evidence_candidates(
        trimmed, reliability_offset=rel_off, gamma_1=g1, gamma_2=g2, calibrate=True
    )
    imat, active_cands = build_interaction_matrix_for_docs(
        cands,
        per_doc_dists or {},
        use_minhash_redundancy=True,
        use_js_conflict=bool(per_doc_dists),
        graph_mode="entity_js",
        conflict_prune_threshold=conflict_prune_threshold,
    )
    opt = TwoStageM2Optimizer(
        lambda_coverage=g1,
        lambda_complementarity=g2,
        lambda_redundancy=lam_div,
        lambda_conflict=1.0,
        selection_algorithm=algorithm,
        recency_aware=recency_aware,
    )
    res = opt.optimize(
        active_cands,
        interaction_matrix=imat,
        max_k=min(max_k, len(active_cands)),
        min_k=min(min_k, len(active_cands)),
        query_entities=extract_entities(question),
        calibrate=False,
    )
    ordered_docs: List[Dict[str, Any]] = []
    for c in res.ordered_docs:
        base = dict(raw_map.get(c.doc_id, {}))
        ordered_docs.append({
            **base,
            "doc_id": c.doc_id,
            "id": c.doc_id,
            "title": c.metadata.get("title", base.get("title", c.doc_id)),
            "text": c.text,
            "reliability": float(c.calibrated_reliability),
            "calibrated_reliability": float(c.calibrated_reliability),
            "uncertainty": float(c.aleatoric_uncertainty),
            "score": float(c.score),
            "token_count": int(c.metadata.get("token_count", compute_token_count(c.text))),
        })

    return ordered_docs, {
        "evaluated_subsets": int(res.m2a_result.evaluated_subsets_count),
        "m2_elapsed_ms": round(float(res.total_elapsed_ms), 3),
        "pruned_pool_len": len(active_cands),
    }


def build_calibrated_decoder(
    params: Dict[str, Any],
    max_new_tokens: int = 48,
    override_thresholds: Optional[Tuple[float, float]] = None,
) -> M3CalibratedDecoder:
    """Construct M3CalibratedDecoder consistently from a Phase II or frozen V5 parameter dict."""
    ab_tau = float(params.get("m3_abstain_threshold", 0.05))
    conf_tau = float(params.get("m3_confidence_threshold", 0.38))
    if override_thresholds is not None:
        ab_tau, conf_tau = override_thresholds
    beta = float(params.get("m3_beta", 0.0))
    cfg = CalibratedDecoderConfig(
        abstain_threshold=ab_tau,
        confidence_threshold=conf_tau,
        beta=beta,
        enable_contrastive=(beta > 0.0),
        alpha_trunc=float(params.get("m3_alpha_trunc", 0.1)),
        max_new_tokens=max_new_tokens,
        preserve_evidence_order=True,
        confidence_temperature=float(params.get("m3_confidence_temperature", 2.0)),
        confidence_tok_weight=float(params.get("m3_confidence_tok_weight", 0.6)),
        confidence_platt_a=float(params.get("m3_confidence_platt_a", 0.8174)),
        confidence_platt_b=float(params.get("m3_confidence_platt_b", -0.2053)),
        contrastive_conf_guard=float(params.get("m3_contrastive_conf_guard", 1.0)),
        protect_eos=bool(params.get("m3_protect_eos", True)),
    )
    return M3CalibratedDecoder(config=cfg)


def compute_paraphrase_disagreement(
    generator: Optional[QwenNF4Generator],
    question: str,
    docs: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Paraphrase-Induced Predictive Disagreement decomposition across semantically
    invariant paraphrase prompts (preserving complete retrieved evidence so multi-hop
    supporting chains are never broken by LOO dropping).

    NOTE on P0-2 Taxonomy: Because expectation is over query paraphrases q^(t) rather than
    posterior weights theta, this measures Paraphrase-Induced Predictive Disagreement
    (legacy keys: bald_u_eu / bald_h_total) and Within-Paraphrase Predictive Entropy
    (legacy key: bald_u_au), NOT Bayesian parameter mutual information I(Y; theta | x, D).
    """
    q = question.strip()
    paras = [
        q,
        f"Provide the short factual answer for: {q}",
        f"Concisely answer the following question: {q}",
        f"Identify the exact answer to: {q}",
    ]
    doc_texts = [str(d.get("text", "")) for d in docs]
    variants: List[Tuple[str, List[str], str]] = [(pq, doc_texts, f"para_{i}") for i, pq in enumerate(paras)]

    dists = [
        compute_next_token_distribution(generator, pq, p_texts, variant_seed=vseed)["dist"]
        for pq, p_texts, vseed in variants
    ]
    if not dists:
        return {
            "bald_h_total": 0.0,
            "bald_u_eu": 0.0,
            "bald_u_au": 0.0,
            "paraphrase_disagreement_eu": 0.0,
            "within_paraphrase_entropy_au": 0.0,
        }

    vocab: Set[str] = set()
    for d in dists:
        vocab.update(d.keys())
    if not vocab:
        return {
            "bald_h_total": 0.0,
            "bald_u_eu": 0.0,
            "bald_u_au": 0.0,
            "paraphrase_disagreement_eu": 0.0,
            "within_paraphrase_entropy_au": 0.0,
        }

    m = float(len(dists))
    mean_dist = {tok: sum(d.get(tok, 0.0) for d in dists) / m for tok in vocab}
    z = sum(mean_dist.values())
    if z > 1e-12:
        mean_dist = {k: v / z for k, v in mean_dist.items()}

    h_total = float(compute_entropy(mean_dist))
    u_au = float(sum(float(compute_entropy(d)) for d in dists) / len(dists))
    u_eu = max(0.0, h_total - u_au)
    return {
        "bald_h_total": round(h_total, 6),
        "bald_u_eu": round(u_eu, 6),
        "bald_u_au": round(u_au, 6),
        "paraphrase_disagreement_eu": round(u_eu, 6),
        "within_paraphrase_entropy_au": round(u_au, 6),
    }


def compute_bald_uncertainty(
    generator: Optional[QwenNF4Generator],
    question: str,
    docs: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """Backwards-compatible alias for `compute_paraphrase_disagreement`."""
    return compute_paraphrase_disagreement(generator, question, docs)


def compute_unified_m1_signal(
    generator: Optional[QwenNF4Generator],
    question: str,
    docs: Sequence[Dict[str, Any]],
    eu_mode: str = "v5_proxy",
    decoupler: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Unified M1 routing signal entry point aligned with P0-2 taxonomy:
      - 'v5_proxy' (default production estimator): UncertaintyDecoupler.decompose()
      - 'paraphrase_disagreement' / 'bald_mi': Paraphrase-Induced Predictive Disagreement
      - 'raw_entropy': Total single-pass predictive entropy H_total
    """
    mode = str(eu_mode).strip().lower()
    if mode in ("paraphrase_disagreement", "bald_mi"):
        p_res = compute_paraphrase_disagreement(generator, question, docs)
        return {
            "u_eu": float(p_res["paraphrase_disagreement_eu"]),
            "u_au": float(p_res["within_paraphrase_entropy_au"]),
            "h_total": float(p_res["bald_h_total"]),
            "mode": "paraphrase_disagreement",
        }
    if mode == "raw_entropy":
        doc_texts = [str(d.get("text", "")) for d in docs]
        probe = compute_next_token_distribution(generator, question, doc_texts, variant_seed="raw")
        return {
            "u_eu": float(probe["entropy"]),
            "u_au": 0.0,
            "h_total": float(probe["entropy"]),
            "mode": "raw_entropy",
        }

    from src.m1_adaptive_retriever import UncertaintyDecoupler

    dec = decoupler if decoupler is not None else UncertaintyDecoupler()
    formatted_docs = [
        {"id": d.get("doc_id", str(i)), "text": str(d.get("text", "")), "entities": d.get("entities", [])}
        for i, d in enumerate(docs)
    ]
    v5_m = dec.decompose(
        question,
        formatted_docs,
        model_fn=lambda q, txts: compute_next_token_distribution(generator, q, txts, variant_seed="v5")["dist"],
    )
    return {
        "u_eu": float(v5_m.epistemic_uncertainty),
        "u_au": float(v5_m.aleatoric_uncertainty),
        "h_total": float(v5_m.total_entropy),
        "mode": "v5_proxy",
    }


class FastDryRunDenseEncoder:
    """
    Lightweight offline dense encoder stub matching SentenceTransformer.encode(...)
    for instantaneous CPU dry-run & unit testing without loading HuggingFace weights.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _vec(self, text: str) -> torch.Tensor:
        v = torch.zeros(self.dim, dtype=torch.float32)
        for tok in re.findall(r"\b\w+\b", (text or "").lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0
        norm = torch.norm(v, p=2)
        if float(norm.item()) > 1e-9:
            v = v / norm
        return v

    def encode(self, sentences: Any, convert_to_tensor: bool = True, show_progress_bar: bool = False) -> Any:
        if isinstance(sentences, str):
            t = self._vec(sentences)
            return t if convert_to_tensor else t.numpy()
        rows = [self._vec(str(s)) for s in sentences]
        mat = torch.stack(rows, dim=0) if rows else torch.zeros((0, self.dim), dtype=torch.float32)
        return mat if convert_to_tensor else mat.numpy()
