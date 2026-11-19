"""
wp7_cross_dataset.py
====================
Milestone 3 Phase C (WP7): Cross-Dataset Scaling & Zero-Tuning Task Transfer.

Implements AboutMileStone_3.md Sec 7:
  1. Task Attribute Matrix evaluation across PopQA (long-tail/M1), HotpotQA (2-hop/M1+M2),
     2WikiMultiHopQA (relational/M2), and MuSiQue (complex multi-hop/M2+M3) on scaled splits.
  2. Strict In-Domain Tuning vs Out-of-Domain Zero-Tuning Transfer Comparison:
       - Source A (HotpotQA Dev): tune M1 quantile threshold, M3 calibration (T, w_tok, platt_a, platt_b),
         and Dev risk-constrained tau_conf -> freeze -> zero-tuning evaluate on PopQA, 2Wiki, MuSiQue Test!
       - Source B (PopQA Dev): tune long-tail settings -> freeze -> zero-tuning evaluate on HotpotQA, 2Wiki, MuSiQue Test!
     Reports OOD vs In-Domain deltas, cross-dataset variance, and Holm/BH significance tests.

Writes strictly to `results/MileStone3/WP7_CrossDataset/`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.p0_3_m3_adaptive_gating import evaluate_policy_on_traces, select_dev_risk_constrained_threshold
from scripts.phase2.wp3_m3_calibration import _collect_decode_traces, _fit_dev_calibration_configs
from scripts.phase3.phase3_common import (
    WP3_DIRS,
    Phase2CheckpointManager,
    apply_multiple_comparison_correction,
    atomic_write_json,
    compute_unified_m1_signal,
    get_vram_snapshot_mb,
    is_phase3_backbone_available,
    load_phase3_dataset_cache,
    load_v5_balanced_params,
    paired_bootstrap_with_tests,
    reset_cuda_peak,
    sequential_load_phase3_generator,
    setup_phase2_logger,
    teardown_generator,
    verify_v5_frozen_checksums,
)
from src.m1_adaptive_retriever import UncertaintyDecoupler

logger = setup_phase2_logger("phase3.wp7")

TASK_ATTRIBUTE_HYPOTHESES: Dict[str, Dict[str, str]] = {
    "popqa": {"primary_focus": "Long-tail entities & parametric knowledge gaps", "primary_module": "M1"},
    "hotpotqa": {"primary_focus": "2-hop evidence bridging", "primary_module": "M1+M2"},
    "2wikimultihop": {"primary_focus": "Relational chain & entity consistency", "primary_module": "M2"},
    "musique": {"primary_focus": "Complex multi-hop composition & risk", "primary_module": "M2+M3"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="WP7: Scaled Cross-Dataset & Zero-Tuning Task Transfer.")
    parser.add_argument("--model", type=str, default="llama3_8b")
    parser.add_argument("--datasets", nargs="+", default=["popqa", "hotpotqa", "2wikimultihop", "musique"])
    parser.add_argument("--scale", type=str, default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "bf16"])
    parser.add_argument("--allow-hf-online", action="store_true")
    parser.add_argument("--out-dir", type=str, default=str(WP3_DIRS["WP7"]))
    parser.add_argument("--out-file", type=str, default="wp7_cross_dataset_transfer.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    setup_phase2_logger("phase3.wp7", out_dir / "wp7_run.log")
    v5_info = verify_v5_frozen_checksums()
    params = load_v5_balanced_params()
    ab_tau = float(params.get("m3_abstain_threshold", 0.05))
    t0 = time.time()

    if not args.dry_run and not is_phase3_backbone_available(args.model, allow_hf_online=args.allow_hf_online):
        raise RuntimeError(f"Model '{args.model}' is not cached locally. Run autodl_setup.py --download-models first.")

    ckpt = Phase2CheckpointManager(out_dir / f"ckpt_wp7_{args.model}_{args.scale}.json", flush_every=25)
    ckpt.metadata.update({
        "model": args.model,
        "datasets": list(args.datasets),
        "scale": args.scale,
        "quantization": args.quantization,
        "schema_version": "wp7_v2",
    })
    reset_cuda_peak()
    generator = sequential_load_phase3_generator(args.model, dry_run=args.dry_run, max_new_tokens=48, quantization=args.quantization)
    decoupler = UncertaintyDecoupler()

    dev_configs: Dict[str, Dict[str, Any]] = {}
    test_gated_traces: Dict[str, List[Dict[str, Any]]] = {}

    try:
        for ds in args.datasets:
            dev_items, test_items = load_phase3_dataset_cache(ds, scale=args.scale, allow_synthetic_fallback=args.dry_run)

            class _Ns:
                def __init__(self, p: str) -> None:
                    self._p = p
                def has(self, k: str) -> bool:
                    return ckpt.has(f"{self._p}::{k}")
                def get(self, k: str, d: Any = None) -> Any:
                    return ckpt.get(f"{self._p}::{k}", d)
                def record(self, k: str, v: Any) -> None:
                    ckpt.record(f"{self._p}::{k}", v)
                def flush(self, force: bool = True) -> None:
                    ckpt.flush(force=force)

            ns = _Ns(ds)
            dev_b0 = _collect_decode_traces(dev_items, generator, params, beta=0.0, reliability_prompt=False, tag="dev_b0_v2", ckpt=ns)
            dev_eu_vals = []
            for it in dev_items:
                ek = f"wp7_eu::{ds}::{it['id']}"
                if ckpt.has(ek):
                    dev_eu_vals.append(float(ckpt.get(ek)["u_eu"]))
                else:
                    eu = float(compute_unified_m1_signal(generator, it["question"], it["pool"][:3], eu_mode="v5_proxy", decoupler=decoupler)["u_eu"])
                    ckpt.record(ek, {"u_eu": eu})
                    dev_eu_vals.append(eu)

            # Calibrate source Dev M1 tau (75th percentile or V5 0.18), Full Platt, and risk-constrained tau_conf
            tau_eu_in_domain = round(float(np.quantile(dev_eu_vals, 0.75)) if dev_eu_vals else 0.18, 4)
            platt = _fit_dev_calibration_configs(dev_b0)["full_platt"]
            pol = select_dev_risk_constrained_threshold(dev_b0, platt, abstain_threshold=ab_tau)
            dev_configs[ds] = {
                "n_dev": len(dev_items),
                "tau_eu": tau_eu_in_domain,
                "platt": platt,
                "adaptive_tau_conf": float(pol["adaptive_tau_conf"]),
            }

            t_b0 = _collect_decode_traces(test_items, generator, params, beta=0.0, reliability_prompt=False, tag="test_b0_v2", ckpt=ns)
            t_b12 = _collect_decode_traces(test_items, generator, params, beta=0.12, reliability_prompt=False, tag="test_b012_v2", ckpt=ns)
            test_eus = []
            for it in test_items:
                ek = f"wp7_eu::{ds}::{it['id']}"
                if ckpt.has(ek):
                    test_eus.append(float(ckpt.get(ek)["u_eu"]))
                else:
                    eu = float(compute_unified_m1_signal(generator, it["question"], it["pool"][:3], eu_mode="v5_proxy", decoupler=decoupler)["u_eu"])
                    ckpt.record(ek, {"u_eu": eu})
                    test_eus.append(eu)

            # Store paired b0/b012 + eu so transfer configs can even vary gating tau cleanly without re-decoding
            test_gated_traces[ds] = [
                {"b0": r0, "b12": r12, "u_eu": eu}
                for r0, r12, eu in zip(t_b0, t_b12, test_eus)
            ]
        ckpt.flush(force=True)
    finally:
        teardown_generator(generator)

    # Evaluate In-Domain vs Out-of-Domain Transfer Matrix strictly on unseen Test sets
    transfer_sources = [s for s in ("hotpotqa", "popqa") if s in dev_configs]
    transfer_matrix: Dict[str, Any] = {}
    in_domain_summary: Dict[str, Any] = {}

    def _eval_transfer(source_cfg: Dict[str, Any], target_ds: str) -> Dict[str, Any]:
        pairs = test_gated_traces[target_ds]
        tau_e = float(source_cfg["tau_eu"])
        # Pair b12 answer metrics with matching b0 distribution telemetry so Dev b0 Platt scores unshifted confidences
        trs = [
            {
                **p["b0"],
                "raw_answer": p["b12"]["raw_answer"],
                "em": p["b12"]["em"],
                "f1": p["b12"]["f1"],
            }
            if p["u_eu"] > tau_e
            else p["b0"]
            for p in pairs
        ]
        ev = evaluate_policy_on_traces(trs, source_cfg["platt"], ab_tau, float(source_cfg["adaptive_tau_conf"]))
        return ev

    for target_ds in args.datasets:
        res_in = _eval_transfer(dev_configs[target_ds], target_ds)
        in_domain_summary[target_ds] = {
            **TASK_ATTRIBUTE_HYPOTHESES.get(target_ds, {}),
            "n_test": res_in["n"],
            "metrics": {k: v for k, v in res_in.items() if not k.startswith("_")},
        }

        per_source = {}
        sig_vs_in: Dict[str, Any] = {}
        for src_ds in transfer_sources:
            res_ood = _eval_transfer(dev_configs[src_ds], target_ds)
            label = "in_domain" if src_ds == target_ds else f"from_{src_ds}"
            per_source[label] = {
                "f1_pct": res_ood["raw_f1_pct"],
                "em_pct": res_ood["raw_em_pct"],
                "ece": res_ood["ece"],
                "brier_score": res_ood["brier_score"],
                "aurc": res_ood["aurc"],
                "coverage_pct": res_ood["coverage_pct"],
                "selective_risk_pct": res_ood["selective_risk_pct"],
                "ece_delta_vs_indomain": round(res_ood["ece"] - res_in["ece"], 4),
                "brier_delta_vs_indomain": round(res_ood["brier_score"] - res_in["brier_score"], 4),
            }
            if src_ds != target_ds:
                sig_vs_in[f"{src_ds}_to_{target_ds}_brier"] = paired_bootstrap_with_tests(
                    res_ood["_per_sample_briers"], res_in["_per_sample_briers"]
                )
        transfer_matrix[target_ds] = {
            "evaluations": per_source,
            "ood_vs_indomain_brier_tests": apply_multiple_comparison_correction(sig_vs_in),
        }

    # Aggregate mean +/- std across datasets
    f1_vals = [in_domain_summary[d]["metrics"]["raw_f1_pct"] for d in args.datasets]
    ece_vals = [in_domain_summary[d]["metrics"]["ece"] for d in args.datasets]
    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "datasets": args.datasets,
            "scale": args.scale,
            "quantization": args.quantization,
            "dry_run": args.dry_run,
            "elapsed_sec": round(time.time() - t0, 2),
            "vram_peak_mb": get_vram_snapshot_mb(),
        },
        "v5_integrity": v5_info,
        "dev_source_configs": dev_configs,
        "in_domain_task_matrix": in_domain_summary,
        "zero_tuning_transfer_matrix": transfer_matrix,
        "cross_dataset_summary": {
            "mean_f1_pct": round(float(np.mean(f1_vals)), 2),
            "std_f1_pct": round(float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0, 2),
            "mean_ece": round(float(np.mean(ece_vals)), 4),
            "std_ece": round(float(np.std(ece_vals, ddof=1)) if len(ece_vals) > 1 else 0.0, 4),
        },
    }
    out_file = out_dir / args.out_file
    atomic_write_json(out_file, payload)
    logger.info(f"Saved WP7 cross-dataset transfer results -> {out_file}")


if __name__ == "__main__":
    main()
