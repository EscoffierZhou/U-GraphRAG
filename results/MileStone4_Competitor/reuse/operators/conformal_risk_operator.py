"""
conformal_risk_operator.py
==========================
Split Conformal Prediction & Conformal Risk Control operator distilled from
`repos/conformal-rag-guardrails/src/models/core_algorithm.py#L455-L478` (CRG, arXiv:2512.15068)
and Angelopoulos et al. Conformal Risk Control, tailored for zero-GPU-cost
finite-sample selective abstention on Dev/Test calibrated probabilities.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from scripts.phase2.common import compute_calibration_and_risk_suite


def fit_split_conformal_abstention(
    dev_confs: Sequence[float],
    dev_f1s: Sequence[float],
    alpha_harmful: float = 0.15,
    correct_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Fit finite-sample Split Conformal abstention threshold on Dev items so that at most
    `alpha_harmful` fraction of answerable/correct predictions (`f1 >= correct_threshold`)
    are wrongly abstained on future exchangeable test samples:
      quantile level q_idx = ceil((n_pos + 1) * (1 - alpha_harmful)) - 1 over s_i = 1 - conf_i.
    """
    confs = np.asarray(list(dev_confs), dtype=float)
    f1s = np.asarray(list(dev_f1s), dtype=float)
    pos_mask = f1s >= float(correct_threshold)
    pos_scores = 1.0 - confs[pos_mask]
    n_pos = int(len(pos_scores))

    if n_pos == 0:
        return {
            "tau_conf": 0.5,
            "n_pos_dev": 0,
            "alpha_harmful": float(alpha_harmful),
            "finite_sample_retain_prob": 0.0,
        }

    # Finite-sample conformal quantile on nonconformity s_i = 1 - conf_i
    alpha = max(1e-4, min(0.99, float(alpha_harmful)))
    q_rank = int(math.ceil((n_pos + 1) * (1.0 - alpha)))
    q_rank = max(1, min(n_pos, q_rank))
    sorted_s = np.sort(pos_scores)
    s_hat = float(sorted_s[q_rank - 1])
    tau_conf = max(0.0, min(1.0, 1.0 - s_hat))
    retain_guarantee = 1.0 - float(math.floor((n_pos + 1) * alpha)) / float(n_pos + 1)
    return {
        "tau_conf": round(tau_conf, 6),
        "nonconformity_q": round(s_hat, 6),
        "n_pos_dev": n_pos,
        "alpha_harmful": float(alpha),
        "finite_sample_retain_prob": round(max(0.0, min(1.0, retain_guarantee)), 4),
    }


def fit_conformal_risk_control(
    dev_confs: Sequence[float],
    dev_f1s: Sequence[float],
    target_risk: float = 0.35,
    delta_ucb: float = 0.05,
    min_coverage: float = 0.30,
) -> Dict[str, Any]:
    """
    Learn threshold `tau_conf` satisfying upper-confidence-bounded selective risk <= `target_risk`
    (finite-sample Hoeffding margin `sqrt(log(1/delta)/(2*n_acc))` over accepted subset on Dev)
    whenever achievable above `min_coverage`, else picks minimum-risk threshold >= `min_coverage`.
    """
    confs = np.asarray(list(dev_confs), dtype=float)
    f1s = np.asarray(list(dev_f1s), dtype=float)
    n = len(confs)
    if n == 0:
        return {"tau_conf": 0.5, "achieved_feasible": False, "dev_ucb_risk": 1.0}

    cands = sorted(set(float(c) for c in confs))
    best_feasible: Optional[Tuple[float, float, float]] = None
    fallback_best: Tuple[float, float, float] = (cands[0], 1.0, 1.0)

    for tau in cands:
        keep = confs >= tau
        n_acc = int(np.sum(keep))
        cov = float(n_acc) / float(n)
        if cov < min_coverage or n_acc == 0:
            continue
        emp_risk = float(np.mean(1.0 - f1s[keep]))
        margin = math.sqrt(math.log(1.0 / max(1e-6, delta_ucb)) / (2.0 * n_acc))
        ucb_risk = min(1.0, emp_risk + margin)
        if ucb_risk <= float(target_risk):
            # Maximize coverage while respecting finite-sample risk UCB
            if best_feasible is None or cov > best_feasible[1]:
                best_feasible = (tau, cov, ucb_risk)
        if emp_risk < fallback_best[1]:
            fallback_best = (tau, emp_risk, ucb_risk)

    if best_feasible is not None:
        return {
            "tau_conf": round(float(best_feasible[0]), 6),
            "achieved_feasible": True,
            "dev_coverage_pct": round(best_feasible[1] * 100.0, 2),
            "dev_ucb_risk_pct": round(best_feasible[2] * 100.0, 2),
        }
    return {
        "tau_conf": round(float(fallback_best[0]), 6),
        "achieved_feasible": False,
        "dev_emp_risk_pct": round(fallback_best[1] * 100.0, 2),
        "dev_ucb_risk_pct": round(fallback_best[2] * 100.0, 2),
    }


def evaluate_conformal_abstention(
    test_confs: Sequence[float],
    test_f1s: Sequence[float],
    tau_conf: float,
    max_ev_rels: Optional[Sequence[float]] = None,
    abstain_rel_floor: float = 0.05,
) -> Dict[str, Any]:
    """Evaluate conformal threshold cleanly with full calibration & risk suite."""
    flags = []
    for i, c in enumerate(test_confs):
        rel_bad = (max_ev_rels is not None) and (float(max_ev_rels[i]) < float(abstain_rel_floor))
        flags.append(bool(rel_bad or float(c) < float(tau_conf)))
    suite = compute_calibration_and_risk_suite(list(test_confs), list(test_f1s), abstained_flags=flags)
    eff_f1 = [0.0 if ab else float(f) for ab, f in zip(flags, test_f1s)]
    return {
        "tau_conf": round(float(tau_conf), 6),
        "raw_f1_pct": round(float(np.mean(test_f1s)) * 100.0, 2) if test_f1s else 0.0,
        "effective_f1_pct": round(float(np.mean(eff_f1)) * 100.0, 2) if eff_f1 else 0.0,
        "coverage_pct": suite["selective_coverage_pct"],
        "selective_accuracy_pct": suite["selective_accuracy_pct"],
        "selective_risk_pct": suite["selective_risk_pct"],
        "harmful_abstention_rate_pct": suite["harmful_abstention_rate_pct"],
        "valid_abstention_precision_pct": suite["valid_abstention_precision_pct"],
        "ece": suite["ece"],
        "auroc_confidence": suite["auroc_confidence"],
        "aurc": suite["aurc"],
    }
