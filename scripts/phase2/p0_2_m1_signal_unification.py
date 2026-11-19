"""
p0_2_m1_signal_unification.py
==============================
MileStone 2 P0-2 & P0-4 Closure Experiment:
  1. Unifies M1 Uncertainty Signal Taxonomy across production (`v5_proxy`: UncertaintyDecoupler)
     and diagnostic comparators (`paraphrase_disagreement`, `within_paraphrase_entropy`,
     `raw_entropy_h_total`, `self_consistency_5x`), eliminating Bayesian MI overclaiming.
  2. Re-evaluates existing WP1 closed-book & 4-quadrant checkpoints with unified taxonomy
     plus bootstrap 95% CIs and Holm-Bonferroni / BH-FDR adjusted statistics.
  3. Expands Controlled Evidence-Replacement Causal Intervention (`Gold -> Missing` vs
     `Gold -> Conflict`) from N=60 to N=150 on HotpotQA across BOTH Qwen2.5-3B and Qwen2.5-7B,
     measuring BOTH `v5_proxy` (primary deployed estimator) and `paraphrase_disagreement`
     alongside `raw_entropy_h_total` to verify statistical separation across all signals.

Writes strictly to `results/MileStone2/P0_Closure_Fixes/`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cache_builder import load_or_build_cache
from scripts.phase2.common import (
    M1_SIGNAL_TAXONOMY,
    WP_DIRS,
    Phase2CheckpointManager,
    apply_multiple_comparison_correction,
    atomic_write_json,
    compute_next_token_distribution,
    compute_paraphrase_disagreement,
    get_vram_snapshot_mb,
    load_json_file,
    paired_bootstrap_with_tests,
    reset_cuda_peak,
    sequential_load_generator,
    setup_phase2_logger,
    teardown_generator,
    verify_v5_frozen_checksums,
)
from src.m1_adaptive_retriever import UncertaintyDecoupler
from src.m3_calibrated_decoder import compute_auroc

logger = setup_phase2_logger("phase2.p0_2")


def _compute_dual_signals(
    generator: Any,
    question: str,
    docs: Sequence[Dict[str, Any]],
    decoupler: UncertaintyDecoupler,
) -> Dict[str, float]:
    """Compute deployed V5 Proxy, Paraphrase Disagreement, and Raw Entropy on identical inputs."""
    doc_texts = [str(d.get("text", "")) for d in docs]
    raw_probe = compute_next_token_distribution(generator, question, doc_texts, variant_seed="base")
    v5_res = decoupler.decompose(
        question,
        [{"id": d.get("doc_id", str(i)), "text": str(d.get("text", "")), "entities": d.get("entities", [])} for i, d in enumerate(docs)],
        model_fn=lambda q, txts: compute_next_token_distribution(generator, q, txts, variant_seed="v5")["dist"],
    )
    para_res = compute_paraphrase_disagreement(generator, question, docs)
    return {
        "v5_u_eu_proxy": round(float(v5_res.epistemic_uncertainty), 6),
        "v5_u_au_proxy": round(float(v5_res.aleatoric_uncertainty), 6),
        "paraphrase_disagreement_eu": float(para_res["paraphrase_disagreement_eu"]),
        "within_paraphrase_entropy_au": float(para_res["within_paraphrase_entropy_au"]),
        "raw_entropy_h_total": round(float(raw_probe["entropy"]), 6),
    }


def _summarize_wp1_q4_alignment(models: Sequence[str]) -> Dict[str, Any]:
    """Read existing WP1 checkpoints cleanly (read-only) to build canonical 4-quadrant comparison table."""
    out: Dict[str, Any] = {}
    for mk in models:
        ckpt_p = WP_DIRS["WP1"] / f"ckpt_wp1_{mk}.json"
        if not ckpt_p.exists():
            continue
        recs = load_json_file(ckpt_p).get("records", {})
        q4 = [v for k, v in recs.items() if k.startswith("q4::")]
        if not q4:
            continue

        quads = {}
        for qn in ("Q1_known_consistent", "Q2_unknown_sufficient", "Q3_known_conflicting", "Q4_unknown_missing"):
            sub = [r for r in q4 if r.get("quadrant") == qn]
            if not sub:
                continue
            quads[qn] = {
                "count": len(sub),
                "mean_v5_eu_proxy": round(float(np.mean([r["v5_u_eu_proxy"] for r in sub])), 4),
                "mean_v5_au_proxy": round(float(np.mean([r["v5_u_au_proxy"] for r in sub])), 4),
                "mean_paraphrase_eu": round(float(np.mean([r["bald_u_eu"] for r in sub])), 4),
                "mean_within_para_au": round(float(np.mean([r["bald_u_au"] for r in sub])), 4),
                "mean_raw_entropy": round(float(np.mean([r["raw_entropy_h_total"] for r in sub])), 4),
            }

        sep_labels = [int(r["expected_eu_high"]) for r in q4]
        out[mk] = {
            "n_quadrant_probes": len(q4),
            "quadrant_means": quads,
            "insufficient_or_conflict_vs_sufficient_auroc": {
                "v5_u_eu_proxy": round(float(compute_auroc([r["v5_u_eu_proxy"] for r in q4], sep_labels)), 4),
                "paraphrase_disagreement_eu": round(float(compute_auroc([r["bald_u_eu"] for r in q4], sep_labels)), 4),
                "raw_entropy_h_total": round(float(compute_auroc([r["raw_entropy_h_total"] for r in q4], sep_labels)), 4),
            },
        }
    return out


def run_expanded_replacement_for_model(
    model_key: str,
    hp_items: Sequence[Dict[str, Any]],
    out_dir: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    """Run N=150 controlled Gold->Missing & Gold->Conflict interventions across BOTH M1 estimators."""
    ckpt = Phase2CheckpointManager(out_dir / f"ckpt_p0_2_{model_key}.json", flush_every=25)
    reset_cuda_peak()
    generator = sequential_load_generator(model_key, dry_run=dry_run, max_new_tokens=24)
    decoupler = UncertaintyDecoupler()
    rows: List[Dict[str, Any]] = []

    try:
        for it in hp_items:
            iid = str(it["id"])
            rkey = f"repl150::{iid}"
            if ckpt.has(rkey):
                rows.append(ckpt.get(rkey))
                continue

            pool = it.get("pool", [])
            gold = [d for d in pool if d.get("is_gold")]
            dist = [d for d in pool if not d.get("is_gold")]
            conf_doc = it.get("synthetic_conflict_doc")
            if not gold or not dist:
                continue

            docs_gold = gold[:2] + dist[:1]
            docs_miss = dist[:3]
            docs_conf = ([gold[0], conf_doc] + dist[:1]) if conf_doc else dist[:3]

            m_gold = _compute_dual_signals(generator, it["question"], docs_gold, decoupler)
            m_miss = _compute_dual_signals(generator, it["question"], docs_miss, decoupler)
            m_conf = _compute_dual_signals(generator, it["question"], docs_conf, decoupler)

            rec = {
                "id": iid,
                "gold": m_gold,
                "missing": m_miss,
                "conflict": m_conf,
            }
            ckpt.record(rkey, rec)
            rows.append(rec)

        ckpt.flush(force=True)
        vram_mb = get_vram_snapshot_mb()
    finally:
        teardown_generator(generator)

    # Build paired tests family with Holm-Bonferroni & BH-FDR multiple comparison correction
    paired_raw: Dict[str, Dict[str, Any]] = {
        "v5_eu_missing_minus_gold": paired_bootstrap_with_tests(
            [r["missing"]["v5_u_eu_proxy"] for r in rows],
            [r["gold"]["v5_u_eu_proxy"] for r in rows],
        ),
        "v5_au_conflict_minus_gold": paired_bootstrap_with_tests(
            [r["conflict"]["v5_u_au_proxy"] for r in rows],
            [r["gold"]["v5_u_au_proxy"] for r in rows],
        ),
        "para_eu_missing_minus_gold": paired_bootstrap_with_tests(
            [r["missing"]["paraphrase_disagreement_eu"] for r in rows],
            [r["gold"]["paraphrase_disagreement_eu"] for r in rows],
        ),
        "para_au_conflict_minus_gold": paired_bootstrap_with_tests(
            [r["conflict"]["within_paraphrase_entropy_au"] for r in rows],
            [r["gold"]["within_paraphrase_entropy_au"] for r in rows],
        ),
        "raw_entropy_missing_minus_gold": paired_bootstrap_with_tests(
            [r["missing"]["raw_entropy_h_total"] for r in rows],
            [r["gold"]["raw_entropy_h_total"] for r in rows],
        ),
        "raw_entropy_conflict_minus_gold": paired_bootstrap_with_tests(
            [r["conflict"]["raw_entropy_h_total"] for r in rows],
            [r["gold"]["raw_entropy_h_total"] for r in rows],
        ),
    }

    # Double-dissociation specificity check: delta EU (missing) vs delta EU (conflict), delta AU (conflict) vs delta AU (missing)
    paired_raw["para_eu_specificity_missing_minus_conflict"] = paired_bootstrap_with_tests(
        [r["missing"]["paraphrase_disagreement_eu"] - r["gold"]["paraphrase_disagreement_eu"] for r in rows],
        [r["conflict"]["paraphrase_disagreement_eu"] - r["gold"]["paraphrase_disagreement_eu"] for r in rows],
    )
    paired_raw["para_au_specificity_conflict_minus_missing"] = paired_bootstrap_with_tests(
        [r["conflict"]["within_paraphrase_entropy_au"] - r["gold"]["within_paraphrase_entropy_au"] for r in rows],
        [r["missing"]["within_paraphrase_entropy_au"] - r["gold"]["within_paraphrase_entropy_au"] for r in rows],
    )

    corrected = apply_multiple_comparison_correction(paired_raw, p_field="wilcoxon_p_value")
    corrected = apply_multiple_comparison_correction(corrected, p_field="paired_t_p_value")

    return {
        "model": model_key,
        "n_tested": len(rows),
        "vram_peak_mb": vram_mb,
        "condition_means": {
            cond: {
                metric: round(float(np.mean([r[cond][metric] for r in rows])), 4) if rows else 0.0
                for metric in (
                    "v5_u_eu_proxy",
                    "v5_u_au_proxy",
                    "paraphrase_disagreement_eu",
                    "within_paraphrase_entropy_au",
                    "raw_entropy_h_total",
                )
            }
            for cond in ("gold", "missing", "conflict")
        },
        "paired_interventions": corrected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-2: Unified M1 Signal Alignment & Expanded N=150 Replacement Test.")
    parser.add_argument("--models", nargs="+", default=["qwen3b", "qwen7b"])
    parser.add_argument("--repl-limit", type=int, default=150)
    parser.add_argument("--out-dir", type=str, default=str(WP_DIRS["P0"]))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    setup_phase2_logger("phase2.p0_2", out_dir / "p0_2_run.log")
    v5_info = verify_v5_frozen_checksums()
    t0 = time.time()

    hp_cache = load_or_build_cache("hotpotqa", pool_k=16, allow_synthetic_fallback=args.dry_run)
    pool_candidates = [it for it in hp_cache.get("items", []) if it.get("split") in ("dev", "test")]
    if not pool_candidates:
        pool_candidates = list(hp_cache.get("items", []))
    target_items = pool_candidates[: (15 if args.dry_run else args.repl_limit)]

    runs: Dict[str, Any] = {}
    for mk in args.models:
        logger.info(f"[{mk}] Running P0-2 N={len(target_items)} dual-estimator replacement interventions...")
        runs[mk] = run_expanded_replacement_for_model(mk, target_items, out_dir, dry_run=args.dry_run)

    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": args.models,
            "repl_limit": len(target_items),
            "dry_run": args.dry_run,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "v5_integrity": v5_info,
        "m1_signal_taxonomy": M1_SIGNAL_TAXONOMY,
        "wp1_four_quadrant_signal_comparison": _summarize_wp1_q4_alignment(args.models),
        "expanded_replacement_interventions": runs,
    }
    out_file = out_dir / "p0_2_m1_signal_unification.json"
    atomic_write_json(out_file, payload)
    logger.info(f"Saved P0-2 signal unification results -> {out_file}")


if __name__ == "__main__":
    main()
