"""
milestone_after_posthoc_conformal.py
====================================
Zero-GPU-cost Split Conformal Risk Control & Finite-Sample Abstention upgrade
over already-completed Milestone 3 WP8 / WP6 checkpoints (`ckpt_wp8_*.json` / `ckpt_wp6_*.json`).

No GPU inference is repeated: reads Dev `(T, w_tok)` Platt calibration parameters
and stored probability traces, fits finite-sample conformal quantiles strictly on Dev,
and audits Test coverage, selective risk, harmful abstention rate, and risk UCB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results.MileStone4_Competitor.reuse import resolve_milestone3_dir, resolve_v5_frozen_dir
from results.MileStone4_Competitor.reuse.operators.conformal_risk_operator import (
    evaluate_conformal_abstention,
    fit_conformal_risk_control,
    fit_split_conformal_abstention,
)
from scripts.phase2.common import Phase2CheckpointManager, atomic_write_json, verify_v5_frozen_checksums
from scripts.phase2.p0_3_m3_adaptive_gating import select_dev_risk_constrained_threshold
from scripts.phase2.wp3_m3_calibration import _fit_dev_calibration_configs, _score_trace_with_cal_params


def _collect_wp8_dataset_traces(
    ckpt: Phase2CheckpointManager,
    dataset_name: str,
) -> Dict[str, Any]:
    """Extract Dev (`dev_b0_v2`) and Test (`test_b0_v2`) traces for WP8.3 post-hoc conformal analysis."""
    dev_prefix = f"wp8_3::{dataset_name}::trace::dev_b0_v2::"
    test_prefix = f"wp8_3::{dataset_name}::trace::test_b0_v2::"
    test_b12_prefix = f"wp8_3::{dataset_name}::trace::test_b012_v2::"

    dev_traces = [v for k, v in ckpt.records.items() if k.startswith(dev_prefix)]
    test_b0 = {k[len(test_prefix) :]: v for k, v in ckpt.records.items() if k.startswith(test_prefix)}
    test_b12 = {k[len(test_b12_prefix) :]: v for k, v in ckpt.records.items() if k.startswith(test_b12_prefix)}
    return {"dev": dev_traces, "test_b0": test_b0, "test_b12": test_b12}


def _collect_wp6_dataset_traces(
    ckpt: Phase2CheckpointManager,
    dataset_name: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """Extract Dev (`wp6_dev_v2`) and Full_UGraphRAG Test (`wp6_res_v2`) traces from WP6 ckpt."""
    dev_prefix = f"wp6_dev_v2::{dataset_name}::trace::dev_b0::"
    res_prefix = f"wp6_res_v2::{dataset_name}::{seed}::Full_UGraphRAG::"
    cal_key = f"wp6_dev_cal_v2::{dataset_name}"

    dev_traces = [v for k, v in ckpt.records.items() if k.startswith(dev_prefix)]
    test_recs = [v for k, v in ckpt.records.items() if k.startswith(res_prefix)]
    cal_bundle = ckpt.get(cal_key) if ckpt.has(cal_key) else None
    return {"dev": dev_traces, "test_recs": test_recs, "cal_bundle": cal_bundle}


def run_posthoc_conformal_audit(
    wp8_ckpt_path: Optional[Path] = None,
    wp6_ckpt_paths: Sequence[Path] = (),
    alphas_harmful: Sequence[float] = (0.10, 0.15, 0.20, 0.30),
    target_risks: Sequence[float] = (0.35, 0.45, 0.55),
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run zero-cost finite-sample Split Conformal audit across available WP8/WP6 checkpoints."""
    v5_info = verify_v5_frozen_checksums(v5_dir=resolve_v5_frozen_dir())
    results: Dict[str, Any] = {"v5_integrity": v5_info, "wp8_conformal": {}, "wp6_conformal": {}}

    if wp8_ckpt_path and Path(wp8_ckpt_path).exists():
        ckpt8 = Phase2CheckpointManager(Path(wp8_ckpt_path), flush_every=100)
        for ds in ("hotpotqa", "musique"):
            bundle = _collect_wp8_dataset_traces(ckpt8, ds)
            dev_tr = bundle["dev"]
            test_b0_map = bundle["test_b0"]
            if not dev_tr or not test_b0_map:
                continue
            platt = _fit_dev_calibration_configs(dev_tr)["full_platt"]
            dev_confs = [_score_trace_with_cal_params(tr, platt["T"], platt["w_tok"], platt["platt_a"], platt["platt_b"]) for tr in dev_tr]
            dev_f1s = [float(tr["f1"]) for tr in dev_tr]
            dev_rels = [float(tr.get("max_ev_rel", 0.5)) for tr in dev_tr]

            test_list = list(test_b0_map.values())
            test_confs = [_score_trace_with_cal_params(tr, platt["T"], platt["w_tok"], platt["platt_a"], platt["platt_b"]) for tr in test_list]
            test_f1s = [float(tr["f1"]) for tr in test_list]
            test_rels = [float(tr.get("max_ev_rel", 0.5)) for tr in test_list]

            # Existing Dev policy baseline + Split Conformal thresholds
            orig_pol = select_dev_risk_constrained_threshold(dev_tr, platt, abstain_threshold=0.05)
            orig_eval = evaluate_conformal_abstention(test_confs, test_f1s, float(orig_pol["adaptive_tau_conf"]), test_rels)

            harm_grid = {}
            for a in alphas_harmful:
                fit_sc = fit_split_conformal_abstention(dev_confs, dev_f1s, alpha_harmful=float(a))
                harm_grid[f"alpha_{int(round(a * 100))}"] = {
                    "conformal_fit": fit_sc,
                    "test_eval": evaluate_conformal_abstention(test_confs, test_f1s, float(fit_sc["tau_conf"]), test_rels),
                }

            risk_grid = {}
            for trisk in target_risks:
                fit_rc = fit_conformal_risk_control(dev_confs, dev_f1s, target_risk=float(trisk))
                risk_grid[f"target_risk_{int(round(trisk * 100))}"] = {
                    "risk_control_fit": fit_rc,
                    "test_eval": evaluate_conformal_abstention(test_confs, test_f1s, float(fit_rc["tau_conf"]), test_rels),
                }

            results["wp8_conformal"][ds] = {
                "n_dev": len(dev_tr),
                "n_test": len(test_list),
                "calibrated_platt": platt,
                "original_dev_adaptive": orig_eval,
                "split_conformal_harmful_control": harm_grid,
                "conformal_risk_control": risk_grid,
            }

    for p6 in wp6_ckpt_paths:
        p6_path = Path(p6)
        if not p6_path.exists():
            continue
        ckpt6 = Phase2CheckpointManager(p6_path, flush_every=100)
        model_label = ckpt6.metadata.get("model") or p6_path.stem
        per_ds: Dict[str, Any] = {}
        for ds in ("hotpotqa", "popqa", "2wikimultihop", "musique"):
            bundle6 = _collect_wp6_dataset_traces(ckpt6, ds, seed=42)
            dev_tr = bundle6["dev"]
            test_recs = bundle6["test_recs"]
            if not dev_tr or not test_recs:
                continue
            cal_b = bundle6["cal_bundle"]
            platt = cal_b["full_platt"] if cal_b and "full_platt" in cal_b else _fit_dev_calibration_configs(dev_tr)["full_platt"]
            dev_confs = [_score_trace_with_cal_params(tr, platt["T"], platt["w_tok"], platt["platt_a"], platt["platt_b"]) for tr in dev_tr]
            dev_f1s = [float(tr["f1"]) for tr in dev_tr]

            test_confs = [float(r["confidence"]) for r in test_recs]
            test_f1s = [float(r["raw_f1"]) for r in test_recs]

            orig_tau = float(cal_b["adaptive_tau_conf"]) if cal_b and "adaptive_tau_conf" in cal_b else float(
                select_dev_risk_constrained_threshold(dev_tr, platt)["adaptive_tau_conf"]
            )
            orig_eval = evaluate_conformal_abstention(test_confs, test_f1s, orig_tau)
            fit_sc15 = fit_split_conformal_abstention(dev_confs, dev_f1s, alpha_harmful=0.15)
            eval_sc15 = evaluate_conformal_abstention(test_confs, test_f1s, float(fit_sc15["tau_conf"]))
            fit_rc40 = fit_conformal_risk_control(dev_confs, dev_f1s, target_risk=0.40)
            eval_rc40 = evaluate_conformal_abstention(test_confs, test_f1s, float(fit_rc40["tau_conf"]))

            per_ds[ds] = {
                "n_dev": len(dev_tr),
                "n_test": len(test_recs),
                "original_dev_adaptive": orig_eval,
                "split_conformal_alpha15": {"fit": fit_sc15, "test_eval": eval_sc15},
                "conformal_risk_ucb40": {"fit": fit_rc40, "test_eval": eval_rc40},
            }
        if per_ds:
            results["wp6_conformal"][model_label] = per_ds

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_path / "conformal_posthoc_summary.json", results)
        md_lines = [
            "# Post-Hoc Split Conformal Risk Control Audit (Zero Extra GPU Calls)",
            "",
            "| Source | Dataset | Policy | Tau Conf | Coverage (%) | Sel Acc (%) | HarmAb (%) | AURC |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for ds, dv in results["wp8_conformal"].items():
            ov = dv["original_dev_adaptive"]
            md_lines.append(f"| WP8 | {ds} | Dev Adaptive Baseline | {ov['tau_conf']:.4f} | {ov['coverage_pct']:.2f} | {ov['selective_accuracy_pct']:.2f} | {ov['harmful_abstention_rate_pct']:.2f} | {ov['aurc']:.4f} |")
            for k, hv in dv["split_conformal_harmful_control"].items():
                ev = hv["test_eval"]
                md_lines.append(f"| WP8 | {ds} | Split Conformal ({k}) | {ev['tau_conf']:.4f} | {ev['coverage_pct']:.2f} | {ev['selective_accuracy_pct']:.2f} | {ev['harmful_abstention_rate_pct']:.2f} | {ev['aurc']:.4f} |")
        for mk, mv in results["wp6_conformal"].items():
            for ds, dv in mv.items():
                ov = dv["original_dev_adaptive"]
                sv = dv["split_conformal_alpha15"]["test_eval"]
                md_lines.append(f"| WP6-{mk} | {ds} | Dev Adaptive | {ov['tau_conf']:.4f} | {ov['coverage_pct']:.2f} | {ov['selective_accuracy_pct']:.2f} | {ov['harmful_abstention_rate_pct']:.2f} | {ov['aurc']:.4f} |")
                md_lines.append(f"| WP6-{mk} | {ds} | Split Conformal (alpha=0.15) | {sv['tau_conf']:.4f} | {sv['coverage_pct']:.2f} | {sv['selective_accuracy_pct']:.2f} | {sv['harmful_abstention_rate_pct']:.2f} | {sv['aurc']:.4f} |")
        (out_path / "CONFORMAL_POSTHOC_TABLE.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return results


def main() -> None:
    m3_dir = resolve_milestone3_dir()
    parser = argparse.ArgumentParser(description="Offline Split Conformal post-hoc upgrade over WP8/WP6 checkpoints.")
    parser.add_argument("--wp8-ckpt", type=str, default=str(m3_dir / "WP8_MethodValidity" / "ckpt_wp8_qwen3b_medium.json"))
    parser.add_argument("--wp6-ckpts", nargs="*", default=[
        str(m3_dir / "WP6_CrossFamily" / "ckpt_wp6_qwen3b_medium.json"),
        str(m3_dir / "WP6_CrossFamily" / "ckpt_wp6_qwen7b_medium.json"),
        str(m3_dir / "WP6_CrossFamily" / "ckpt_wp6_llama3_8b_medium.json"),
        str(m3_dir / "WP6_qwen7b" / "ckpt_wp6_qwen7b_medium.json"),
        str(m3_dir / "WP6_llama3" / "ckpt_wp6_llama3_8b_medium.json"),
    ])
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    args = parser.parse_args()

    res = run_posthoc_conformal_audit(
        wp8_ckpt_path=Path(args.wp8_ckpt),
        wp6_ckpt_paths=[Path(p) for p in args.wp6_ckpts],
        out_dir=Path(args.out_dir),
    )
    print(json.dumps({
        "wp8_datasets": list(res["wp8_conformal"].keys()),
        "wp6_models": list(res["wp6_conformal"].keys()),
        "out_dir": args.out_dir,
    }, indent=2))


if __name__ == "__main__":
    main()
