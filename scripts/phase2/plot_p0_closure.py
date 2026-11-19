"""
plot_p0_closure.py
==================
Generate publication-grade 300-DPI figures under
`results/MileStone2/P0_Closure_Fixes/figures/`
directly from verified real-GPU P0-1..P0-4 JSON artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.common import WP_DIRS, assert_safe_output_path, setup_phase2_logger

logger = setup_phase2_logger("phase2.p0_figures")

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


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_p0_1_figures(p0_dir: Path, fig_dir: Path) -> Path:
    d = _load(p0_dir / "p0_1_m2_conflict_and_scale.json")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    lambdas = [1.0, 2.5, 5.0, 10.0]

    # Panel A: Dev Pareto Tradeoff (Gold Recall vs Conflict Retention across lambda_conflict)
    ax = axes[0]
    for ds, label, col, marker in [
        ("hotpotqa", "HotpotQA Dev (N=80)", "#2b5c8f", "o"),
        ("2wikimultihop", "2Wiki Dev (N=80)", "#d95f02", "s"),
    ]:
        conf_sum = d["datasets"][ds]["dev_conflict_summary"]
        clean_sum = d["datasets"][ds]["dev_clean_summary"]
        c_ret = [conf_sum[f"M2_orig_lc{l:.1f}"]["conflict_retention_pct"] for l in lambdas]
        g_rec = [conf_sum[f"M2_orig_lc{l:.1f}"]["sp_recall"] for l in lambdas]
        c_rec = [clean_sum[f"M2_orig_lc{l:.1f}"]["sp_recall"] for l in lambdas]
        ax.plot(lambdas, g_rec, f"-{marker}", color=col, linewidth=2.0, label=f"{label}: Conflict Gold Recall")
        ax.plot(lambdas, c_ret, f"--{marker}", color=col, alpha=0.65, linewidth=1.6, label=f"{label}: Conflict Retention")
        ax.plot(lambdas, c_rec, f":{marker}", color=col, alpha=0.85, linewidth=1.6, label=f"{label}: Clean Gold Recall")
        for l_val, gr, cr in zip(lambdas, g_rec, c_ret):
            ax.annotate(f"{gr:.1f}%", (l_val, gr), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5, color=col)
    ax.set_xlabel(r"Conflict Penalty Weight $\lambda_{\mathrm{conflict}}$")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(a) Dev Sweep: Gold Recall vs Conflict Retention")
    ax.set_xticks(lambdas)
    ax.set_ylim(25, 108)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    # Panel B: Unseen Test Conflict-Slice Gold Recall & Conflict Retention
    ax = axes[1]
    vars_cmp = [
        ("A5_mmr_baseline", "MMR"),
        ("G_entity_cooccur", "EntityGraph"),
        ("M2_orig_lc1.0", r"M2 ($\lambda=1$)"),
        ("M2_orig_lc5.0", r"M2 ($\lambda=5$)"),
        ("P_hard_conflict_prune", "HardPrune"),
        ("D_overlap_gated_js_lc5", "OverlapGated"),
    ]
    x = np.arange(len(vars_cmp))
    w = 0.20
    hp_conf = d["datasets"]["hotpotqa"]["test_conflict_summary"]
    tw_conf = d["datasets"]["2wikimultihop"]["test_conflict_summary"]
    ax.bar(x - 1.5 * w, [hp_conf[k]["sp_recall"] for k, _ in vars_cmp], w, label="Hotpot Gold Rec", color="#386cb0")
    ax.bar(x - 0.5 * w, [hp_conf[k]["conflict_retention_pct"] for k, _ in vars_cmp], w, label="Hotpot Conf Ret", color="#f0027f")
    ax.bar(x + 0.5 * w, [tw_conf[k]["sp_recall"] for k, _ in vars_cmp], w, label="2Wiki Gold Rec", color="#7fc97f")
    ax.bar(x + 1.5 * w, [tw_conf[k]["conflict_retention_pct"] for k, _ in vars_cmp], w, label="2Wiki Conf Ret", color="#fdc086")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in vars_cmp], fontsize=8.0, rotation=15)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"(b) Unseen Test Conflict Slice (N={d['datasets']['hotpotqa']['n_test_unseen']}/{d['datasets']['2wikimultihop']['n_test_unseen']})")
    ax.set_ylim(0, 114)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)

    # Panel C: Unseen Test Clean vs Conflict Answer F1
    ax = axes[2]
    hp_clean = d["datasets"]["hotpotqa"]["test_clean_summary"]
    tw_clean = d["datasets"]["2wikimultihop"]["test_clean_summary"]
    ax.bar(x - 1.5 * w, [hp_clean[k]["f1"] for k, _ in vars_cmp], w, label="Hotpot Clean F1", color="#2b5c8f")
    ax.bar(x - 0.5 * w, [hp_conf[k]["f1"] for k, _ in vars_cmp], w, label="Hotpot Conflict F1", color="#6baed6")
    ax.bar(x + 0.5 * w, [tw_clean[k]["f1"] for k, _ in vars_cmp], w, label="2Wiki Clean F1", color="#d95f02")
    ax.bar(x + 1.5 * w, [tw_conf[k]["f1"] for k, _ in vars_cmp], w, label="2Wiki Conflict F1", color="#fdae6b")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in vars_cmp], fontsize=8.0, rotation=15)
    ax.set_ylabel("Answer F1 (%)")
    ax.set_title("(c) Expanded Unseen Test Answer F1")
    ax.set_ylim(25, 68)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)

    fig.suptitle("MileStone 2 P0-1: Conflict Weight Scan, Disentanglement Diagnostic & Expanded Test Evaluation", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = assert_safe_output_path(fig_dir / "fig_p0_1_lambda_pareto_and_scale.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_p0_2_figures(p0_dir: Path, fig_dir: Path) -> Path:
    d = _load(p0_dir / "p0_2_m1_signal_unification.json")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))

    # Panel A: Q1->Q4 Comparison of V5 Proxy vs Paraphrase Disagreement across 3B & 7B
    ax = axes[0]
    quads = ["Q1_known_consistent", "Q2_unknown_sufficient", "Q3_known_conflicting", "Q4_unknown_missing"]
    q_lbls = ["Q1\nKnown+\nConsist", "Q2\nUnknown+\nSuffic", "Q3\nKnown+\nConflict", "Q4\nUnknown+\nMissing"]
    x = np.arange(len(quads))
    w = 0.20
    q4_align = d.get("wp1_four_quadrant_signal_comparison", {})
    if "qwen3b" in q4_align and "qwen7b" in q4_align:
        v5_3 = [q4_align["qwen3b"]["quadrant_means"][q]["mean_v5_eu_proxy"] for q in quads]
        pa_3 = [q4_align["qwen3b"]["quadrant_means"][q]["mean_paraphrase_eu"] for q in quads]
        v5_7 = [q4_align["qwen7b"]["quadrant_means"][q]["mean_v5_eu_proxy"] for q in quads]
        pa_7 = [q4_align["qwen7b"]["quadrant_means"][q]["mean_paraphrase_eu"] for q in quads]
        ax.bar(x - 1.5 * w, v5_3, w, label="3B V5 EU Proxy (Prod)", color="#1b9e77")
        ax.bar(x - 0.5 * w, pa_3, w, label="3B Paraphrase Disagr.", color="#7570b3")
        ax.bar(x + 0.5 * w, v5_7, w, label="7B V5 EU Proxy (Prod)", color="#66c2a5")
        ax.bar(x + 1.5 * w, pa_7, w, label="7B Paraphrase Disagr.", color="#beaed4")
        ax.axhline(0.18, color="red", linestyle="--", linewidth=1.0, label=r"Prod Gate $\tau_{EU}=0.18$")
        ax.set_xticks(x)
        ax.set_xticklabels(q_lbls, fontsize=8.5)
        ax.set_ylabel("Uncertainty Score")
        ax.set_title("(a) Canonical Signal Taxonomy on 4-Quadrant Probe (N=160)")
        ax.legend(frameon=False, fontsize=7.8, loc="upper left")

    # Panel B: Expanded N=150 Paired Interventions (95% Bootstrap CI)
    ax = axes[1]
    runs = d["expanded_replacement_interventions"]
    bars_spec = []
    for mk, mlabel, c_base in [("qwen3b", "3B", ["#1b9e77", "#d95f02"]), ("qwen7b", "7B", ["#2b8cbe", "#b30000"])]:
        if mk not in runs:
            continue
        pi = runs[mk]["paired_interventions"]
        n_t = runs[mk]["n_tested"]
        bars_spec.extend([
            (f"{mlabel} V5 EU\n(Miss-Gold)", pi["v5_eu_missing_minus_gold"], c_base[0]),
            (f"{mlabel} Para EU\n(Miss-Gold)", pi["para_eu_missing_minus_gold"], c_base[0]),
            (f"{mlabel} Para AU\n(Conf-Gold)", pi["para_au_conflict_minus_gold"], c_base[1]),
        ])
    xr = np.arange(len(bars_spec))
    means = [b[1]["mean_delta"] for b in bars_spec]
    yerr_lo = [max(0.0, b[1]["mean_delta"] - b[1]["ci_95_low"]) for b in bars_spec]
    yerr_hi = [max(0.0, b[1]["ci_95_high"] - b[1]["mean_delta"]) for b in bars_spec]
    cols = [b[2] for b in bars_spec]
    ax.bar(xr, means, yerr=[yerr_lo, yerr_hi], capsize=3.5, color=cols, alpha=0.90, width=0.58)
    ax.axhline(0.0, color="black", linewidth=0.9)
    for i, (_, e, _) in enumerate(bars_spec):
        p = e.get("wilcoxon_p_value_holm", e.get("wilcoxon_p_value", 1.0))
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        top = max(e["ci_95_high"], e["mean_delta"]) + 0.008
        ax.text(i, top, f"{e['mean_delta']:+.3f}\n({stars})", ha="center", va="bottom", fontsize=7.6)
    ax.set_xticks(xr)
    ax.set_xticklabels([b[0] for b in bars_spec], fontsize=7.8)
    ax.set_ylabel("Paired Mean Delta (95% Bootstrap CI)")
    n_rep = next(iter(runs.values()))["n_tested"] if runs else 0
    ax.set_title(f"(b) Expanded Causal Replacement Interventions (N={n_rep}, Holm p)")
    max_y = max((b[1]["ci_95_high"] for b in bars_spec), default=0.3)
    min_y = min((b[1]["ci_95_low"] for b in bars_spec), default=0.0)
    ax.set_ylim(min(-0.02, min_y * 1.2), max(0.05, max_y * 1.28))

    fig.suptitle("MileStone 2 P0-2: Unified M1 Signal Alignment & Expanded Causal Dissociation", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = assert_safe_output_path(fig_dir / "fig_p0_2_m1_signal_unification.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_p0_3_figures(p0_dir: Path, fig_dir: Path) -> Path:
    d = _load(p0_dir / "p0_3_m3_adaptive_gating.json")
    model_key = next(iter(d["models"].keys()))
    ds_map = d["models"][model_key]["datasets"]
    datasets = [(k, lbl) for k, lbl in [("popqa", "PopQA"), ("hotpotqa", "HotpotQA"), ("2wikimultihop", "2Wiki"), ("musique", "MuSiQue")] if k in ds_map]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    x = np.arange(len(datasets))
    w = 0.25

    # Panel A: Answer F1 Fixed_B0 vs Fixed_B012 vs Gated_Beta (V5 EU)
    ax = axes[0]
    f0 = [ds_map[k]["test_policies"]["Fixed_B0__DevAdaptiveTau"]["raw_f1_pct"] for k, _ in datasets]
    f12 = [ds_map[k]["test_policies"]["Fixed_B012__DevAdaptiveTau"]["raw_f1_pct"] for k, _ in datasets]
    fg = [ds_map[k]["test_policies"]["Gated_Beta__DevAdaptiveTau"]["raw_f1_pct"] for k, _ in datasets]
    ax.bar(x - w, f0, w, label=r"Fixed $\beta=0.0$", color="#7f7f7f")
    ax.bar(x, f12, w, label=r"Fixed $\beta=0.12$", color="#dd8452")
    ax.bar(x + w, fg, w, label=r"Uncertainty-Gated $\beta(q)$", color="#1b9e77")
    for xi, a, b, c in zip(x, f0, f12, fg):
        ax.text(xi + w, c + 1.0, f"{c:.1f}%", ha="center", va="bottom", fontsize=7.6, color="#0d5e45", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(N={ds_map[k]['n_test_unseen']})" for k, lbl in datasets], fontsize=8.5)
    ax.set_ylabel("Unseen Test Answer F1 (%)")
    ax.set_title("(a) Uncertainty-Gated CD Avoids Easy-Task Regression")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=8)

    # Panel B: Coverage & Harmful Abstention: Legacy Tau=0.38 vs Dev Adaptive Tau
    ax = axes[1]
    cov_38 = [ds_map[k]["test_policies"]["Gated_Beta__Tau038"]["coverage_pct"] for k, _ in datasets]
    cov_ad = [ds_map[k]["test_policies"]["Gated_Beta__DevAdaptiveTau"]["coverage_pct"] for k, _ in datasets]
    harm_38 = [ds_map[k]["test_policies"]["Gated_Beta__Tau038"]["harmful_abstention_overall_pct"] if "harmful_abstention_overall_pct" in ds_map[k]["test_policies"]["Gated_Beta__Tau038"] else ds_map[k]["test_policies"]["Gated_Beta__Tau038"]["harmful_abstention_rate_pct"] for k, _ in datasets]
    harm_ad = [ds_map[k]["test_policies"]["Gated_Beta__DevAdaptiveTau"]["harmful_abstention_rate_pct"] for k, _ in datasets]
    w2 = 0.20
    ax.bar(x - 1.5 * w2, cov_38, w2, label=r"Cov % ($\tau=0.38$)", color="#c994c7")
    ax.bar(x - 0.5 * w2, cov_ad, w2, label="Cov % (Dev Adaptive)", color="#7a0177")
    ax.bar(x + 0.5 * w2, harm_38, w2, label=r"Harmful Ab % ($\tau=0.38$)", color="#fc9272")
    ax.bar(x + 1.5 * w2, harm_ad, w2, label="Harmful Ab % (Dev Adaptive)", color="#cb181d")
    for xi, c1, c2 in zip(x, cov_38, cov_ad):
        ax.text(xi - 1.5 * w2, c1 + 1.2, f"{c1:.0f}", ha="center", va="bottom", fontsize=7.2)
        ax.text(xi - 0.5 * w2, c2 + 1.2, f"{c2:.0f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in datasets], fontsize=8.5)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(b) Coverage & Harmful Abstention: Static vs Adaptive Threshold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)

    # Panel C: Risk vs Target Coverage {0.4, 0.5, 0.6, 0.7, 0.8}
    ax = axes[2]
    cov_grid = [40, 50, 60, 70, 80]
    for k, lbl, col in [("popqa", "PopQA", "#1b9e77"), ("hotpotqa", "HotpotQA", "#386cb0"), ("2wikimultihop", "2Wiki", "#d95f02"), ("musique", "MuSiQue", "#e7298a")]:
        if k not in ds_map:
            continue
        rc = ds_map[k]["exact_risk_at_target_coverages"]["Gated_Beta"]
        ys = [rc[f"risk_at_cov_{c}"] for c in cov_grid]
        ax.plot([c / 100.0 for c in cov_grid], ys, "-o", color=col, linewidth=2.0, label=f"{lbl} (AURC={ds_map[k]['test_policies']['Gated_Beta__DevAdaptiveTau']['aurc']:.3f})")
    ax.set_xlabel("Target Coverage Fraction")
    ax.set_ylabel("Selective Risk (1 - F1, %)")
    ax.set_title("(c) Risk-Coverage Trajectory of Uncertainty-Gated M3")
    ax.set_xticks([c / 100.0 for c in cov_grid])
    ax.legend(frameon=False, fontsize=7.8)

    fig.suptitle(f"MileStone 2 P0-3: Uncertainty-Gated CD & Dev Risk-Constrained Coverage ({model_key})", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = assert_safe_output_path(fig_dir / "fig_p0_3_m3_adaptive_gating.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_p0_4_figures(p0_dir: Path, fig_dir: Path, model_key: str = "qwen3b") -> Path:
    d = _load(p0_dir / f"p0_4_wp5_expanded_stress_{model_key}.json")
    ss = d["stress_suites"]
    n_t = d["metadata"]["n_tested"]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))

    configs = [
        ("retrieval_noise_suite", ["noise_0pct", "noise_40pct", "noise_60pct"], ["0%", "40%", "60%"], "Noise Injection Ratio", "(a) Retrieval Noise Stress"),
        ("conflict_injection_suite", ["conflict_0pct", "conflict_20pct", "conflict_40pct"], ["0%", "20%", "40%"], "Conflict Injection Ratio", "(b) Factual Conflict Stress"),
        ("gold_evidence_drop_suite", ["gold_drop_0pct", "gold_drop_50pct", "gold_drop_100pct"], ["0%", "50%", "100%"], "Gold Evidence Drop Ratio", "(c) Gold Starvation (Brier ↓)"),
    ]
    for ax, (suite_k, lvls, x_lbls, xlabel, title), use_brier in zip(axes, configs, [False, False, True]):
        smap = ss[suite_k]
        xs = np.arange(len(lvls))
        metric = "brier_score" if use_brier else "f1_pct"
        for sys_k, label, col, marker in [
            ("B0_StandardRAG", "B0 Standard RAG", "#7f7f7f", "s"),
            ("Full_UGraphRAG", "Full U-GraphRAG (V5 EU)", "#1b9e77", "o"),
            ("Full_HardPrune", "Hard Conflict Prune", "#d95f02", "^"),
        ]:
            vals = [smap[l][sys_k][metric] for l in lvls]
            ax.plot(xs, vals, f"-{marker}", color=col, linewidth=2.0, markersize=6.5, label=label)
            for xi, v in zip(xs, vals):
                fmt = f"{v:.3f}" if use_brier else f"{v:.1f}"
                ax.annotate(fmt, (xi, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.6, color=col)
        ax.set_xticks(xs)
        ax.set_xticklabels(x_lbls)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Brier Score (lower better)" if use_brier else "Answer F1 (%)")
        ax.set_title(f"{title} (N={n_t})")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"MileStone 2 P0-4: Expanded N={n_t} Three-Axis Stress Evaluation (Unified V5 Proxy)", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = assert_safe_output_path(fig_dir / "fig_p0_4_wp5_expanded_stress.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p0_dir = WP_DIRS["P0"]
    fig_dir = p0_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    if (p0_dir / "p0_1_m2_conflict_and_scale.json").exists():
        generated.append(plot_p0_1_figures(p0_dir, fig_dir))
    if (p0_dir / "p0_2_m1_signal_unification.json").exists():
        generated.append(plot_p0_2_figures(p0_dir, fig_dir))
    if (p0_dir / "p0_3_m3_adaptive_gating.json").exists():
        generated.append(plot_p0_3_figures(p0_dir, fig_dir))
    if (p0_dir / "p0_4_wp5_expanded_stress_qwen3b.json").exists():
        generated.append(plot_p0_4_figures(p0_dir, fig_dir))
    logger.info(f"Generated {len(generated)} P0 closure figures -> {[str(p) for p in generated]}")


if __name__ == "__main__":
    main()
