"""
wp11_build_paper_package.py
===========================
Milestone 3 Phase E (WP11): Reproducibility Manifest, Error Taxonomy Mining,
Publication Figures (Figs 2-7), LaTeX/Markdown Tables, and MILESTONE3_EXPERIMENT_REPORT.md.

Handles partial runs gracefully (generates all available figures/tables cleanly without crashing
if a subset of WP6-WP10 has finished). Writes strictly to `results/MileStone3/WP11_PaperPackage/`
and `results/MileStone3/MILESTONE3_EXPERIMENT_REPORT.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase3.phase3_common import (
    MILE3_DIR,
    WP3_DIRS,
    assert_safe_output_path,
    atomic_write_json,
    collect_hardware_manifest,
    setup_phase2_logger,
    verify_v5_frozen_checksums,
)

logger = setup_phase2_logger("phase3.wp11")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.8,
    "axes.titlesize": 11.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#fcfcfd",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def _load_opt(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def mine_qualitative_failure_cases(pred_dir: Path, max_per_category: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Scan JSONL prediction traces from WP6 for concrete error examples by taxonomy category."""
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "retrieval_miss": [],
        "selection_loss": [],
        "generation_or_reasoning_error": [],
        "harmful_abstention": [],
    }
    if not pred_dir.exists():
        return buckets
    for jp in sorted(pred_dir.glob("*Full_UGraphRAG*.jsonl")):
        tag = jp.stem
        with open(jp, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                cat = rec.get("failure_category")
                if cat in buckets and len(buckets[cat]) < max_per_category:
                    buckets[cat].append({
                        "source_run": tag,
                        "id": rec.get("id"),
                        "question": rec.get("question"),
                        "gold": rec.get("gold_answers", [])[:2],
                        "prediction": rec.get("prediction"),
                        "confidence": rec.get("confidence"),
                        "pool_gold_recall": rec.get("pool_gold_recall"),
                        "selected_sp_recall": rec.get("sp_recall"),
                    })
    return buckets


def _plot_fig2_m1(wp8: Dict[str, Any], fig_dir: Path) -> Optional[Path]:
    qmap = wp8.get("wp8_1_m1_validity", {}).get("quadrant_summary", {})
    if not qmap:
        return None
    quads = [q for q in ("Q1_known_consistent", "Q2_unknown_sufficient", "Q3_known_conflicting", "Q4_unknown_missing") if q in qmap]
    x = np.arange(len(quads))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - w, [qmap[q]["mean_v5_eu"] for q in quads], w, label=r"V5 Proxy ($U_{\mathrm{EU}}^{\mathrm{v5}}$)", color="#1b9e77")
    ax.bar(x, [qmap[q]["mean_para_eu"] for q in quads], w, label=r"Paraphrase Disagr. ($U_{\mathrm{EU}}^{\mathrm{para}}$)", color="#7570b3")
    ax.bar(x + w, [qmap[q]["v5_expand_rate_pct"] / 100.0 for q in quads], w, label="M1 Trigger Rate (frac)", color="#d95f02")
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace("_", "\n") for q in quads], fontsize=8.5)
    ax.set_ylabel("Uncertainty Score / Trigger Fraction")
    ax.set_title("Fig. 2: M1 Signal Separation & Adaptive Trigger Rate (WP8.1)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    out = assert_safe_output_path(fig_dir / "fig2_wp8_m1_uncertainty.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_fig3_m2(wp8: Dict[str, Any], fig_dir: Path) -> Optional[Path]:
    m2 = wp8.get("wp8_2_m2_validity", {})
    csum = m2.get("test_clean_summary", {})
    xsum = m2.get("test_conflict_summary", {})
    if not csum or not xsum:
        return None
    keys = [k for k in ("A0_top5", "A5_mmr_baseline", "G_entity_cooccur", "M2_orig_lc1.0", "M2_orig_lc5.0", "D_overlap_gated_js_lc5", "P_hard_conflict_prune") if k in csum]
    x = np.arange(len(keys))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.bar(x - w, [csum[k]["f1"] for k in keys], w, label="Clean F1 (%)", color="#2b5c8f")
    ax.bar(x, [xsum[k]["f1"] for k in keys], w, label="Conflict-Slice F1 (%)", color="#6baed6")
    ax.bar(x + w, [xsum[k]["conflict_retention_pct"] for k in keys], w, label="Conflict Retention (%)", color="#e34a33")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=7.8, rotation=16)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Fig. 3: M2 Evidence Optimization vs Baselines & Conflict Suppression (WP8.2)")
    ax.legend(frameon=False, fontsize=8.2)
    fig.tight_layout()
    out = assert_safe_output_path(fig_dir / "fig3_wp8_m2_optimization.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_fig4_cross_family(wp6: Dict[str, Any], fig_dir: Path) -> Optional[Path]:
    mat = wp6.get("matrix", {})
    models = [m for m, v in mat.items() if v.get("status") == "completed"]
    if not models:
        return None
    n = len(models)
    if n <= 3:
        n_rows, n_cols = 1, n
        figsize = (5.5 * n_cols, 4.6)
    else:
        n_cols = math.ceil(n / 2)
        n_rows = 2
        figsize = (5.5 * n_cols, 8.5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    for idx, mk in enumerate(models):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]
        ds_map = mat[mk].get("datasets", {})
        ds_names = list(ds_map.keys())
        x = np.arange(len(ds_names))
        w = 0.18
        for j, (meth, lbl, col) in enumerate([
            ("B0_StandardRAG", "StandardRAG", "#7f7f7f"),
            ("B1_BM25_RAG", "BM25", "#a6cee3"),
            ("B3_MMR_RAG", "MMR", "#fdbf6f"),
            ("Full_UGraphRAG", "Full U-GraphRAG", "#1b9e77"),
        ]):
            vals = [ds_map[d]["multiseed_aggregate"].get(meth, {}).get("mean", {}).get("f1_pct", 0.0) for d in ds_names]
            ax.bar(x + (j - 1.5) * w, vals, w, label=lbl, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_names, fontsize=8.2, rotation=12)
        ax.set_ylabel("Answer F1 (%)")
        ax.set_title(f"{mk} Cross-Dataset F1")
        ax.legend(frameon=False, fontsize=7.5)

    # Hide extra unused subplots in grid
    for idx in range(n, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].set_visible(False)

    fig.suptitle("Fig. 4: Cross-Family & Cross-Dataset Main Results (WP6)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = assert_safe_output_path(fig_dir / "fig4_wp6_cross_family_main.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_fig5_calibration(wp8: Dict[str, Any], fig_dir: Path) -> Optional[Path]:
    m3 = wp8.get("wp8_3_m3_validity", {})
    if not m3:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    covs = [0.4, 0.5, 0.6, 0.7, 0.8]
    for ds, col in [("hotpotqa", "#386cb0"), ("musique", "#e7298a")]:
        if ds not in m3:
            continue
        rc = m3[ds].get("risk_at_target_coverages", {})
        ys = [rc.get(f"risk_at_cov_{int(c*100)}", 0.0) for c in covs]
        ax.plot(covs, ys, "-o", color=col, linewidth=2.0, label=f"{ds} (Adaptive Cov={m3[ds]['dev_adaptive_tau']['coverage_pct']:.1f}%)")
    ax.set_xlabel("Target Coverage Fraction")
    ax.set_ylabel("Selective Risk (1 - F1, %)")
    ax.set_title("Fig. 5: Calibration & Selective Risk-Coverage Curves (WP8.3)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    out = assert_safe_output_path(fig_dir / "fig5_wp8_calibration_rc.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_fig6_stress(wp9: Dict[str, Any], fig_dir: Path) -> Optional[Path]:
    curves = wp9.get("stress", {}).get("curves", {})
    if not curves:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    panels = [
        ("axis_a1_noise", "noise_", "Noise Rate (%)", "(a) Retrieval Noise"),
        ("axis_b1_conflict_intensity", "conflict_", "Conflict Rate (%)", "(b) Factual Conflict"),
        ("axis_a3_gold_drop", "gold_drop_", "Gold Drop (%)", "(c) Gold Starvation (Brier)"),
    ]
    for ax, (ak, prefix, xl, ttl) in zip(axes, panels):
        sub = curves.get(ak, {})
        if not sub:
            continue
        lvls = list(sub.keys())
        xs = np.arange(len(lvls))
        metric = "brier_score" if ak == "axis_a3_gold_drop" else "f1_pct"
        for sys_k, lbl, col in [("B0_StandardRAG", "StandardRAG", "#7f7f7f"), ("B3_MMR_RAG", "MMR", "#fdbf6f"), ("Full_UGraphRAG", "Full U-GraphRAG", "#1b9e77")]:
            ax.plot(xs, [sub[l][sys_k][metric] for l in lvls], "-o", color=col, linewidth=2.0, label=lbl)
        ax.set_xticks(xs)
        ax.set_xticklabels([l.replace(prefix, "").replace("pct", "%") for l in lvls])
        ax.set_xlabel(xl)
        ax.set_ylabel("Brier Score" if metric == "brier_score" else "Answer F1 (%)")
        ax.set_title(ttl)
        ax.legend(frameon=False, fontsize=8.0)
    fig.suptitle(f"Fig. 6: Single-Variable Stress Curves ({wp9.get('stress', {}).get('model', 'LLM')}, WP9)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = assert_safe_output_path(fig_dir / "fig6_wp9_stress_curves.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_fig7_efficiency(wp10: Dict[str, Any], fig_dir: Path) -> Optional[Path]:
    models = {m: v.get("benchmarks", {}) for m, v in wp10.get("models", {}).items() if v.get("status") == "completed" and v.get("benchmarks")}
    if not models:
        return None
    n = len(models)
    if n <= 3:
        n_rows, n_cols = 1, n
        figsize = (6.0 * n_cols, 4.8)
    else:
        n_cols = math.ceil(n / 2)
        n_rows = 2
        figsize = (6.0 * n_cols, 8.8)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    cfg_colors = {
        "Standard_RAG": "#7f7f7f",
        "B3_MMR_RAG": "#fdbf6f",
        "M1_Only": "#386cb0",
        "M2_Greedy": "#a6cee3",
        "M2_Exact_Enum": "#b2df8a",
        "Full_UGraphRAG": "#1b9e77",
        "Full_UGraphRAG_Exact": "#33a02c",
    }

    for idx, (mk, bm) in enumerate(models.items()):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]
        cfgs = list(bm.keys())
        for cfg_name in cfgs:
            rec = bm[cfg_name]
            lat = rec.get("latency_ms", {}).get("e2e_mean", 0.0)
            f1 = rec.get("f1_pct", 0.0)
            vr = max(100.0, rec.get("vram_mb", {}).get("max_reserved_peak", 100.0))
            col = cfg_colors.get(cfg_name, "#4daf4a")
            ax.scatter(lat, f1, s=max(70.0, vr / 12.0), alpha=0.85, color=col, edgecolors="black", linewidth=0.6)
            ax.annotate(f"{cfg_name}\n({vr:.0f}MB)", (lat, f1), textcoords="offset points", xytext=(5, 4), fontsize=7.2)
        ax.set_xlabel("Mean End-to-End Latency (ms)", fontsize=8.8)
        ax.set_ylabel("Answer F1 (%)", fontsize=8.8)
        ax.set_title(f"{mk} Pareto Front", fontsize=9.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

    for idx in range(n, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].set_visible(False)

    fig.suptitle("Fig. 7: Quality-Latency-VRAM Tradeoff Across Models (WP10)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = assert_safe_output_path(fig_dir / "fig7_wp10_quality_latency_vram.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def build_latex_and_markdown_tables(
    wp6: Optional[Dict[str, Any]],
    wp7: Optional[Dict[str, Any]],
    wp10: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """Generate camera-ready LaTeX and Markdown tables strictly from completed JSON outputs."""
    md_lines = ["# Milestone 3 Camera-Ready Tables Summary\n"]
    tex_lines = ["% Auto-generated Milestone 3 LaTeX Tables\n"]

    if wp6:
        md_lines.append("## Table 1: WP6 Cross-Family Generalization (Mean F1% / ECE / AURC)\n")
        md_lines.append("| Backbone | Dataset | B0 Hybrid | B1 BM25 | B3 MMR | Full U-GraphRAG (F1 / ECE / AURC) |")
        md_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
        tex_lines.append("\\begin{table}[t]\n\\centering\n\\caption{WP6 Cross-Family Comparison}\n\\begin{tabular}{llcccc}\n\\toprule")
        tex_lines.append("Backbone & Dataset & Standard & BM25 & MMR & Full (F1 / ECE) \\\\\n\\midrule")
        for mk, mv in wp6.get("matrix", {}).items():
            if mv.get("status") != "completed":
                continue
            for ds, dv in mv.get("datasets", {}).items():
                ms = dv.get("multiseed_aggregate", {})
                f_b0 = ms.get("B0_StandardRAG", {}).get("mean", {}).get("f1_pct", 0.0)
                f_bm = ms.get("B1_BM25_RAG", {}).get("mean", {}).get("f1_pct", 0.0)
                f_mm = ms.get("B3_MMR_RAG", {}).get("mean", {}).get("f1_pct", 0.0)
                full_m = ms.get("Full_UGraphRAG", {}).get("mean", {})
                f_full = full_m.get("f1_pct", 0.0)
                ece_full = full_m.get("ece", 0.0)
                aurc_full = full_m.get("aurc", 0.0)
                md_lines.append(f"| {mk} | {ds} | {f_b0:.2f} | {f_bm:.2f} | {f_mm:.2f} | {f_full:.2f} / {ece_full:.4f} / {aurc_full:.4f} |")
                tex_lines.append(f"{mk} & {ds} & {f_b0:.2f} & {f_bm:.2f} & {f_mm:.2f} & {f_full:.2f} / {ece_full:.4f} \\\\")
        tex_lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    if wp7:
        md_lines.append("\n## Table 2: WP7 In-Domain vs Out-of-Domain Zero-Tuning Transfer\n")
        md_lines.append("| Target Dataset | Source Tuning | F1 (%) | ECE | Brier | Coverage (%) |")
        md_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
        for t_ds, tv in wp7.get("zero_tuning_transfer_matrix", {}).items():
            for src_lbl, sv in tv.get("evaluations", {}).items():
                md_lines.append(f"| {t_ds} | {src_lbl} | {sv['f1_pct']:.2f} | {sv['ece']:.4f} | {sv['brier_score']:.4f} | {sv['coverage_pct']:.2f} |")

    if wp10:
        md_lines.append("\n## Table 3: WP10 Hardware Efficiency & Exact vs Greedy M2 Comparison\n")
        md_lines.append("| Backbone | Variant | E2E (ms) | M2 (ms) | QPS | Peak Reserved (MB) | F1 (%) |")
        md_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
        for mk, mv in wp10.get("models", {}).items():
            if mv.get("status") != "completed":
                continue
            for cfg, cv in mv.get("benchmarks", {}).items():
                md_lines.append(
                    f"| {mk} | {cfg} | {cv['latency_ms']['e2e_mean']:.1f} | {cv['latency_ms']['m2_mean']:.3f} "
                    f"| {cv['throughput']['questions_per_sec']:.2f} | {cv['vram_mb']['max_reserved_peak']:.1f} | {cv['f1_pct']:.2f} |"
                )

    return "\n".join(md_lines) + "\n", "\n".join(tex_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="WP11: Paper Evidence Package, Figures 2-7 & Report Builder.")
    parser.add_argument("--wp6-dir", type=str, default=str(WP3_DIRS["WP6"]))
    parser.add_argument("--wp7-dir", type=str, default=str(WP3_DIRS["WP7"]))
    parser.add_argument("--wp8-dir", type=str, default=str(WP3_DIRS["WP8"]))
    parser.add_argument("--wp9-dir", type=str, default=str(WP3_DIRS["WP9"]))
    parser.add_argument("--wp10-dir", type=str, default=str(WP3_DIRS["WP10"]))
    parser.add_argument("--out-dir", type=str, default=str(WP3_DIRS["WP11"]))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    setup_phase2_logger("phase3.wp11", out_dir / "wp11_run.log")
    v5_info = verify_v5_frozen_checksums()

    wp6 = _load_opt(Path(args.wp6_dir) / "wp6_cross_family_matrix.json")
    wp7 = _load_opt(Path(args.wp7_dir) / "wp7_cross_dataset_transfer.json")
    wp8 = _load_opt(Path(args.wp8_dir) / "wp8_method_validity_results.json")
    wp9 = _load_opt(Path(args.wp9_dir) / "wp9_stress_curves.json")
    wp10 = _load_opt(Path(args.wp10_dir) / "wp10_efficiency_pareto.json")

    figs: List[Path] = []
    if wp8:
        for fn in (_plot_fig2_m1, _plot_fig3_m2, _plot_fig5_calibration):
            p = fn(wp8, fig_dir)
            if p:
                figs.append(p)
    if wp6:
        p = _plot_fig4_cross_family(wp6, fig_dir)
        if p:
            figs.append(p)
    if wp9:
        p = _plot_fig6_stress(wp9, fig_dir)
        if p:
            figs.append(p)
    if wp10:
        p = _plot_fig7_efficiency(wp10, fig_dir)
        if p:
            figs.append(p)

    md_tables, tex_tables = build_latex_and_markdown_tables(wp6, wp7, wp10)
    assert_safe_output_path(out_dir / "TABLES_PHASE3.md").write_text(md_tables, encoding="utf-8")
    assert_safe_output_path(out_dir / "tables_phase3.tex").write_text(tex_tables, encoding="utf-8")

    failures = mine_qualitative_failure_cases(Path(args.wp6_dir) / "predictions")
    atomic_write_json(out_dir / "qualitative_failure_cases.json", failures)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "v5_integrity": v5_info,
        "hardware_manifest": collect_hardware_manifest(),
        "completed_artifacts": {
            "wp6": wp6 is not None,
            "wp7": wp7 is not None,
            "wp8": wp8 is not None,
            "wp9": wp9 is not None,
            "wp10": wp10 is not None,
        },
        "generated_figures": [str(f) for f in figs],
    }
    atomic_write_json(out_dir / "reproducibility_manifest.json", manifest)

    report_lines = [
        "# Milestone 3 External Validation & Methodological Closure Report",
        "",
        f"> Generated: {manifest['generated_at']} | V5_FROZEN Verified: {v5_info['verified_count']}/61 | GPU: `{manifest['hardware_manifest'].get('gpu_name')}`",
        "",
    ]
    if wp8:
        dec = wp8.get("wp8_2_m2_validity", {}).get("validity_gate_decision", {})
        m1_v = wp8.get("wp8_1_m1_validity", {})
        report_lines.extend([
            "## Phase A (WP8): Methodological Validity Gate",
            f"- M1 Operational Claim Scope: `{m1_v.get('recommended_m1_claim_scope', 'operational_uncertainty_routing_signal')}` (V5 Q4>=Q1: `{m1_v.get('m1_monotonic_missing_gt_consistent')}`, Paraphrase Q4>=Q1: `{m1_v.get('paraphrase_monotonic_missing_gt_consistent')}`)",
            f"- M2 Validity Recommendation: `{dec.get('recommended_claim_scope')}` (graph mode: `{dec.get('recommended_m2_graph_mode')}`)",
            "",
        ])
    report_lines.append(md_tables)
    report_lines.append("## Figures Generated\n")
    for f in figs:
        report_lines.append(f"- `{f.name}`")
    report_lines.append("\n## Mined Failure Case Counts (Full U-GraphRAG)\n")
    for k, v in failures.items():
        report_lines.append(f"- `{k}`: {len(v)} sampled trace(s) saved in `qualitative_failure_cases.json`")

    report_path = assert_safe_output_path(out_dir / "MILESTONE3_EXPERIMENT_REPORT.md")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info(f"Built WP11 paper package ({len(figs)} figures + tables + report) -> {out_dir}")


if __name__ == "__main__":
    main()
