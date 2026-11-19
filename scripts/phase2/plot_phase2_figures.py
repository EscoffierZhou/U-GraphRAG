"""
plot_phase2_figures.py
======================
Generate publication-grade 300-DPI figures under results/MileStone2/figures/
directly from the verified real-GPU JSON outputs of WP1 through WP5.
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

from scripts.phase2.common import MILE2_DIR, WP_DIRS, assert_safe_output_path, setup_phase2_logger

logger = setup_phase2_logger("phase2.figures")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
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


def plot_wp1_figures(fig_dir: Path) -> Path:
    d = _load(WP_DIRS["WP1"] / "wp1_eu_validation_results.json")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel A: Four-Quadrant BALD EU Progression (3B vs 7B)
    ax = axes[0]
    quads = ["Q1_known_consistent", "Q2_unknown_sufficient", "Q3_known_conflicting", "Q4_unknown_missing"]
    q_labels = ["Q1\nKnown+\nConsistent", "Q2\nUnknown+\nSufficient", "Q3\nKnown+\nConflicting", "Q4\nUnknown+\nMissing"]
    x = np.arange(len(quads))
    w = 0.36
    vals_3b = [d["runs"]["qwen3b"]["four_quadrant_summary"][q]["mean_bald_u_eu"] for q in quads]
    vals_7b = [d["runs"]["qwen7b"]["four_quadrant_summary"][q]["mean_bald_u_eu"] for q in quads]
    b1 = ax.bar(x - w / 2, vals_3b, w, label="Qwen2.5-3B", color="#4c72b0")
    b2 = ax.bar(x + w / 2, vals_7b, w, label="Qwen2.5-7B", color="#dd8452")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=8.5)
    ax.set_ylabel("Mean Paraphrase BALD EU")
    ax.set_title("(a) Four-Quadrant EU Separation")
    ax.set_ylim(0, max(vals_3b + vals_7b) * 1.25)
    ax.legend(frameon=False, fontsize=9)

    # Panel B: Double Dissociation in Evidence Replacement Sensitivity (N=60)
    ax = axes[1]
    labels_rep = ["EU (Miss - Gold)\nQwen2.5-3B", "AU (Conf - Gold)\nQwen2.5-3B", "EU (Miss - Gold)\nQwen2.5-7B", "AU (Conf - Gold)\nQwen2.5-7B"]
    entries = [
        d["runs"]["qwen3b"]["replacement_sensitivity"]["eu_missing_minus_gold"],
        d["runs"]["qwen3b"]["replacement_sensitivity"]["au_conflict_minus_gold"],
        d["runs"]["qwen7b"]["replacement_sensitivity"]["eu_missing_minus_gold"],
        d["runs"]["qwen7b"]["replacement_sensitivity"]["au_conflict_minus_gold"],
    ]
    means = [e["mean_delta"] for e in entries]
    yerr_low = [e["mean_delta"] - e["ci_95_low"] for e in entries]
    yerr_hi = [e["ci_95_high"] - e["mean_delta"] for e in entries]
    colors = ["#2b8cbe", "#e34a33", "#045a8d", "#b30000"]
    xr = np.arange(len(entries))
    bars = ax.bar(xr, means, yerr=[yerr_low, yerr_hi], capsize=4, color=colors, width=0.56, alpha=0.92)
    ax.axhline(0.0, color="black", linewidth=0.9)
    for i, e in enumerate(entries):
        p = e["sign_flip_p_value"]
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        top = e["ci_95_high"] + 0.012
        ax.text(i, top, f"+{e['mean_delta']:.3f}\n({stars}, p={p:.3g})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xr)
    ax.set_xticklabels(labels_rep, fontsize=8.5)
    ax.set_ylabel("Paired Mean Delta (95% Bootstrap CI)")
    ax.set_title("(b) Controlled Replacement Dissociation (N=60)")
    ax.set_ylim(-0.02, max(e["ci_95_high"] for e in entries) * 1.28)

    # Panel C: Closed-Book Error AUROC Across Baselines
    ax = axes[2]
    m_keys = ["bald_u_eu", "v5_u_eu_proxy", "raw_entropy_h_total", "max_prob_uncertainty", "self_consistency_5x"]
    m_names = ["BALD MI", "V5 Proxy", "Entropy", "1 - MaxP", "SC (5x)"]
    xm = np.arange(len(m_keys))
    w3 = 0.20
    hp3 = [d["runs"]["qwen3b"]["datasets"]["hotpotqa"]["closed_error_detection"][k]["auroc"] for k in m_keys]
    pp3 = [d["runs"]["qwen3b"]["datasets"]["popqa"]["closed_error_detection"][k]["auroc"] for k in m_keys]
    hp7 = [d["runs"]["qwen7b"]["datasets"]["hotpotqa"]["closed_error_detection"][k]["auroc"] for k in m_keys]
    pp7 = [d["runs"]["qwen7b"]["datasets"]["popqa"]["closed_error_detection"][k]["auroc"] for k in m_keys]
    ax.bar(xm - 1.5 * w3, hp3, w3, label="3B HotpotQA", color="#74add1")
    ax.bar(xm - 0.5 * w3, pp3, w3, label="3B PopQA", color="#313695")
    ax.bar(xm + 0.5 * w3, hp7, w3, label="7B HotpotQA", color="#fdae61")
    ax.bar(xm + 1.5 * w3, pp7, w3, label="7B PopQA", color="#d73027")
    ax.set_xticks(xm)
    ax.set_xticklabels(m_names, fontsize=8.5)
    ax.set_ylabel("Closed-Book Error AUROC")
    ax.set_title("(c) Closed-Book Hallucination Detection (N=300)")
    ax.set_ylim(0.50, 0.96)
    ax.legend(frameon=False, ncol=2, fontsize=8)

    fig.suptitle("WP1: Epistemic & Aleatoric Uncertainty Empirical Validation (Qwen2.5-3B & 7B)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = assert_safe_output_path(fig_dir / "fig1_wp1_eu_validation.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_wp2_figures(fig_dir: Path) -> Path:
    hp = _load(WP_DIRS["WP2"] / "wp2_m2_ablation_hotpotqa_test.json")
    tw = _load(WP_DIRS["WP2"] / "wp2_m2_ablation_2wikimultihop_test.json")
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.8))

    # Panel A: Standard Ablation F1 across HotpotQA & 2WikiMultiHopQA
    ax = axes[0]
    variants = [
        ("A0_top3", "Top-3"),
        ("A0_top5", "Top-5"),
        ("A1_exact_no_graph", "Exact NoGraph"),
        ("A2_exact_minhash_no_js", "MinHash NoJS"),
        ("A3_exact_full_orig_order", "Full OrigOrd"),
        ("A4_full_m2_mst_dfs", "Full M2 (DFS)"),
        ("A5_mmr_baseline", "MMR"),
        ("G_entity_cooccur", "Entity Graph"),
        ("G_dense_cosine", "Dense Graph"),
        ("G_random", "Random Graph"),
        ("P_hard_conflict_prune", "Hard Prune"),
        ("O_reverse_dfs", "Reverse DFS"),
    ]
    x = np.arange(len(variants))
    w = 0.38
    hp_f1 = [hp["standard_ablation_summary"][k]["f1"] for k, _ in variants]
    tw_f1 = [tw["standard_ablation_summary"][k]["f1"] for k, _ in variants]
    b1 = ax.bar(x - w / 2, hp_f1, w, label="HotpotQA Test (N=120)", color="#4c72b0")
    b2 = ax.bar(x + w / 2, tw_f1, w, label="2WikiMultiHopQA Test (N=100)", color="#55a868")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.4, f"{h:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in variants], rotation=32, ha="right", fontsize=8.5)
    ax.set_ylabel("QA F1 (%)")
    ax.set_title("(a) M2 Component, Graph-Signal & Ordering Ablation")
    ax.set_ylim(30, 66)
    ax.legend(frameon=False, fontsize=9)

    # Panel B: Conflict-Injected Slice F1 vs Sp-Recall
    ax = axes[1]
    diag_vars = [
        ("A0_top5", "Top-5"),
        ("A1_exact_no_graph", "NoGraph"),
        ("A2_exact_minhash_no_js", "NoJS"),
        ("A4_full_m2_mst_dfs", "Full M2"),
        ("A5_mmr_baseline", "MMR"),
        ("P_hard_conflict_prune", "Hard Prune"),
    ]
    xd = np.arange(len(diag_vars))
    hp_cf1 = [hp["conflict_injected_diagnostic_summary"][k]["f1"] for k, _ in diag_vars]
    tw_cf1 = [tw["conflict_injected_diagnostic_summary"][k]["f1"] for k, _ in diag_vars]
    hp_spr = [hp["conflict_injected_diagnostic_summary"][k]["sp_recall"] for k, _ in diag_vars]
    ax.bar(xd - w / 2, hp_cf1, w, label="HotpotQA Conflict F1 (N=60)", color="#c44e52")
    ax.bar(xd + w / 2, tw_cf1, w, label="2Wiki Conflict F1 (N=50)", color="#dd8452")
    ax2 = ax.twinx()
    ax2.plot(xd, hp_spr, marker="o", color="#2f4b7c", linewidth=2.0, label="HotpotQA Sp-Recall (%)")
    ax2.set_ylabel("Supporting Fact Recall (%)", color="#2f4b7c")
    ax2.set_ylim(40, 100)
    ax.set_xticks(xd)
    ax.set_xticklabels([lbl for _, lbl in diag_vars], fontsize=9)
    ax.set_ylabel("QA F1 (%)")
    ax.set_title("(b) Adversarial Conflict Injection & Pruning Boundary")
    ax.set_ylim(28, 62)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower left", fontsize=8.5)

    fig.suptitle("WP2: M2 Structural Optimization, Ordering & Conflict Boundary (Qwen2.5-3B)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = assert_safe_output_path(fig_dir / "fig2_wp2_m2_ablation.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_wp3_figures(fig_dir: Path) -> Path:
    datasets = [("hotpotqa", "HotpotQA"), ("popqa", "PopQA"), ("2wikimultihop", "2WikiMultiHopQA"), ("musique", "MuSiQue")]
    loaded = {k: _load(WP_DIRS["WP3"] / f"wp3_m3_calibration_{k}.json") for k, _ in datasets}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel A: Uncalibrated vs Temp / Platt / Contrastive ECE across 4 datasets
    ax = axes[0]
    x = np.arange(len(datasets))
    w = 0.19
    m_specs = [
        ("M3_0_uncalibrated", "Uncalibrated", "#d73027"),
        ("M3_1_temp_only", "Temp Scaling", "#4575b4"),
        ("M3_3_full_platt", "Full Platt", "#1a9850"),
        ("M3_6_contrastive_b012", "Contrastive b=0.12", "#8c510a"),
    ]
    for idx, (mkey, mlbl, col) in enumerate(m_specs):
        vals = [loaded[dk]["unseen_test_variants"][mkey]["metrics"]["ece"] for dk, _ in datasets]
        bars = ax.bar(x + (idx - 1.5) * w, vals, w, label=mlbl, color=col)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.008, f"{h:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in datasets], fontsize=9)
    ax.set_ylabel("Unseen Test ECE (Lower is Better)")
    ax.set_title("(a) Zero-Leakage Calibration Across 4 Corpora")
    ax.set_ylim(0, 0.80)
    ax.legend(frameon=False, fontsize=8)

    # Panel B: Brier Score Comparison across 4 datasets
    ax = axes[1]
    for idx, (mkey, mlbl, col) in enumerate(m_specs):
        vals = [loaded[dk]["unseen_test_variants"][mkey]["metrics"]["brier_score"] for dk, _ in datasets]
        bars = ax.bar(x + (idx - 1.5) * w, vals, w, label=mlbl, color=col)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.008, f"{h:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in datasets], fontsize=9)
    ax.set_ylabel("Unseen Test Brier Score (Lower is Better)")
    ax.set_title("(b) Brier Score Reduction on Unseen Test Sets")
    ax.set_ylim(0, 0.72)

    # Panel C: Risk-Coverage Selective Accuracy vs Coverage on HotpotQA & 2Wiki
    ax = axes[2]
    for dk, lbl, col, ls in [
        ("hotpotqa", "HotpotQA Full Platt", "#1b9e77", "-"),
        ("hotpotqa", "HotpotQA Contrastive b=0.12", "#66a61e", "--"),
        ("2wikimultihop", "2Wiki Full Platt", "#7570b3", "-"),
        ("2wikimultihop", "2Wiki Contrastive b=0.12", "#e7298a", "--"),
    ]:
        vkey = "M3_6_contrastive_b012" if "Contrastive" in lbl else "M3_3_full_platt"
        sweep = loaded[dk]["unseen_test_variants"][vkey]["tau_conf_sweep"]
        cov = [p["coverage_pct"] for p in sweep]
        sacc = [p["selective_accuracy_pct"] for p in sweep]
        ax.plot(cov, sacc, marker="o", linestyle=ls, color=col, linewidth=1.8, markersize=4.5, label=lbl)
    ax.set_xlabel("Selective Coverage (%)")
    ax.set_ylabel("Selective QA F1 Accuracy (%)")
    ax.set_title("(c) Risk-Coverage Selective Accuracy Curves")
    ax.invert_xaxis()
    ax.set_ylim(0, 95)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle("WP3: M3 Dev-to-Test Confidence Calibration & Selective Abstention (Qwen2.5-3B, N=120/set)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = assert_safe_output_path(fig_dir / "fig3_wp3_calibration_aurc.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_wp4_figures(fig_dir: Path) -> Path:
    d = _load(WP_DIRS["WP4"] / "wp4_generalization_matrix.json")
    datasets = [("hotpotqa", "HotpotQA"), ("popqa", "PopQA"), ("2wikimultihop", "2WikiMH"), ("musique", "MuSiQue")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    x = np.arange(len(datasets))
    w = 0.19

    # Panel A: Qwen2.5-7B QA F1 across 4 unseen corpora
    ax = axes[0]
    modes = [
        ("ClosedBook", "ClosedBook", "#999999"),
        ("B0_StandardRAG", "B0 Standard RAG", "#e66101"),
        ("M1_NoM2", "M1 Adaptive", "#fdb863"),
        ("Full_UGraphRAG", "Full U-GraphRAG", "#2c7bb6"),
    ]
    for idx, (mkey, mlbl, col) in enumerate(modes):
        vals = [d["matrix"]["qwen7b"]["datasets"][dk]["modes"][mkey]["f1_pct"] for dk, _ in datasets]
        bars = ax.bar(x + (idx - 1.5) * w, vals, w, label=mlbl, color=col)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in datasets], fontsize=9)
    ax.set_ylabel("QA F1 (%)")
    ax.set_title("(a) Qwen2.5-7B Unseen Test F1 (N=120/set)")
    ax.set_ylim(0, 114)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # Panel B: Brier Score Comparison (B0 vs Full_UGraphRAG across 3B & 7B)
    ax = axes[1]
    b0_3 = [d["matrix"]["qwen3b"]["datasets"][dk]["modes"]["B0_StandardRAG"]["calibration_and_risk"]["brier_score"] for dk, _ in datasets]
    fu_3 = [d["matrix"]["qwen3b"]["datasets"][dk]["modes"]["Full_UGraphRAG"]["calibration_and_risk"]["brier_score"] for dk, _ in datasets]
    b0_7 = [d["matrix"]["qwen7b"]["datasets"][dk]["modes"]["B0_StandardRAG"]["calibration_and_risk"]["brier_score"] for dk, _ in datasets]
    fu_7 = [d["matrix"]["qwen7b"]["datasets"][dk]["modes"]["Full_UGraphRAG"]["calibration_and_risk"]["brier_score"] for dk, _ in datasets]
    ax.bar(x - 1.5 * w, b0_3, w, label="3B B0 RAG", color="#fdae6b")
    ax.bar(x - 0.5 * w, fu_3, w, label="3B Full U-GraphRAG", color="#6baed6")
    ax.bar(x + 0.5 * w, b0_7, w, label="7B B0 RAG", color="#d94801")
    ax.bar(x + 1.5 * w, fu_7, w, label="7B Full U-GraphRAG", color="#08519c")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in datasets], fontsize=9)
    ax.set_ylabel("Brier Score (Lower is Better)")
    ax.set_title("(b) Cross-Backbone Calibration Transfer (V5 Frozen)")
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=False, fontsize=8)

    # Panel C: Task-Adaptive M1 Early Exit Rate across Task Difficulty
    ax = axes[2]
    diff_order = [("popqa", "PopQA\n(1-hop)"), ("hotpotqa", "HotpotQA\n(2-hop)"), ("2wikimultihop", "2WikiMH\n(2-4 hop)"), ("musique", "MuSiQue\n(comp. 2-4 hop)")]
    xd = np.arange(len(diff_order))
    ee3 = [d["matrix"]["qwen3b"]["datasets"][dk]["modes"]["Full_UGraphRAG"]["early_exit_rate_pct"] for dk, _ in diff_order]
    ee7 = [d["matrix"]["qwen7b"]["datasets"][dk]["modes"]["Full_UGraphRAG"]["early_exit_rate_pct"] for dk, _ in diff_order]
    ax.plot(xd, ee3, marker="s", linewidth=2.2, color="#4c72b0", label="Qwen2.5-3B Early Exit (%)")
    ax.plot(xd, ee7, marker="o", linewidth=2.2, color="#d73027", label="Qwen2.5-7B Early Exit (%)")
    for i, (v3, v7) in enumerate(zip(ee3, ee7)):
        ax.text(i, v3 - 4.5, f"{v3:.1f}%", ha="center", va="top", fontsize=8.5, color="#4c72b0")
        ax.text(i, v7 + 2.2, f"{v7:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#d73027")
    ax.set_xticks(xd)
    ax.set_xticklabels([lbl for _, lbl in diff_order], fontsize=8.5)
    ax.set_ylabel("Early-Exit Rate (%)")
    ax.set_title("(c) Difficulty-Adaptive Retrieval Gating")
    ax.set_ylim(45, 108)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    fig.suptitle("WP4: 4 Datasets x 2 Backbones Zero-Tuning Generalization Matrix (V5 Frozen)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = assert_safe_output_path(fig_dir / "fig4_wp4_generalization_matrix.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_wp5_figures(fig_dir: Path) -> Path:
    d = _load(WP_DIRS["WP5"] / "wp5_stress_and_profile_qwen3b.json")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel A: 3-Axis Stress Brier Score Degradation (Noise / Conflict / Gold Drop)
    ax = axes[0]
    n_lvls = ["noise_0pct", "noise_20pct", "noise_40pct", "noise_60pct"]
    c_lvls = ["conflict_0pct", "conflict_20pct", "conflict_40pct"]
    g_lvls = ["gold_drop_0pct", "gold_drop_50pct", "gold_drop_100pct"]
    # Plot Gold Drop & Noise Brier curves for B0 vs Full
    gx = [0, 50, 100]
    b0_g = [d["stress_suites"]["gold_evidence_drop_suite"][l]["B0_StandardRAG"]["brier_score"] for l in g_lvls]
    fu_g = [d["stress_suites"]["gold_evidence_drop_suite"][l]["Full_UGraphRAG"]["brier_score"] for l in g_lvls]
    nx = [0, 20, 40, 60]
    b0_n = [d["stress_suites"]["retrieval_noise_suite"][l]["B0_StandardRAG"]["brier_score"] for l in n_lvls]
    fu_n = [d["stress_suites"]["retrieval_noise_suite"][l]["Full_UGraphRAG"]["brier_score"] for l in n_lvls]
    ax.plot(gx, b0_g, marker="o", linestyle="-", color="#d73027", linewidth=2, label="GoldDrop: B0 RAG")
    ax.plot(gx, fu_g, marker="o", linestyle="-", color="#1a9850", linewidth=2, label="GoldDrop: Full U-GraphRAG")
    ax.plot(nx, b0_n, marker="s", linestyle="--", color="#f46d43", linewidth=1.8, label="Noise: B0 RAG")
    ax.plot(nx, fu_n, marker="s", linestyle="--", color="#2b8cbe", linewidth=1.8, label="Noise: Full U-GraphRAG")
    ax.set_xlabel("Perturbation Severity (%)")
    ax.set_ylabel("Brier Score (Lower is Better)")
    ax.set_title("(a) Stress Calibration Resilience (N=60)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel B: Conflict Injection Stress (F1 & AUROC at 0/20/40%)
    ax = axes[1]
    cx = np.arange(3)
    b0_cf1 = [d["stress_suites"]["conflict_injection_suite"][l]["B0_StandardRAG"]["f1_pct"] for l in c_lvls]
    fu_cf1 = [d["stress_suites"]["conflict_injection_suite"][l]["Full_UGraphRAG"]["f1_pct"] for l in c_lvls]
    hp_cf1 = [d["stress_suites"]["conflict_injection_suite"][l]["Full_HardPrune"]["f1_pct"] for l in c_lvls]
    ax.plot(cx, b0_cf1, marker="o", linewidth=2.2, color="#e66101", label="B0 Standard RAG")
    ax.plot(cx, fu_cf1, marker="s", linewidth=2.2, color="#2c7bb6", label="Full U-GraphRAG")
    ax.plot(cx, hp_cf1, marker="^", linewidth=2.2, color="#7b3294", label="Full Hard Prune")
    ax.set_xticks(cx)
    ax.set_xticklabels(["0% Conflict", "20% Conflict", "40% Conflict"])
    ax.set_ylabel("QA F1 (%)")
    ax.set_title("(b) Conflict Injection F1 Comparison")
    ax.set_ylim(34, 60)
    ax.legend(frameon=False, fontsize=8.5, loc="center left")

    # Panel C: Stage Latency Breakdown + Scaling Matrix
    ax = axes[2]
    sc = d["efficiency_profile"]["scaling_matrix"]
    s_labels = [f"k={r['pool_k']},M={r['max_k']}" for r in sc]
    sx = np.arange(len(sc))
    ex_ms = [r["mean_exact_m2_ms"] for r in sc]
    gr_ms = [r["mean_greedy_m2_ms"] for r in sc]
    ax.plot(sx, ex_ms, marker="o", linewidth=2.2, color="#d73027", label="M2 Exact Enum (ms)")
    ax.plot(sx, gr_ms, marker="s", linewidth=2.2, color="#1a9850", label="M2 Greedy (ms)")
    for i, v in enumerate(ex_ms):
        ax.text(i, v + 0.28, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#d73027")
    ax.set_xticks(sx)
    ax.set_xticklabels(s_labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("M2 Optimization Latency (ms)")
    st = d["efficiency_profile"]["stage_latency_ms"]
    ax.set_title(f"(c) M2 Scaling (Mean M2={st['m2_selection_mst']['mean']:.2f}ms / {st['total_end_to_end']['mean']:.0f}ms)")
    ax.set_ylim(0, max(ex_ms) * 1.25)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.suptitle("WP5: System Efficiency Profiling, Scaling & 3-Axis Stress Suites (Qwen2.5-3B)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = assert_safe_output_path(fig_dir / "fig5_wp5_stress_and_scaling.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_all_phase2_figures() -> List[Path]:
    fig_dir = MILE2_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    outs = [
        plot_wp1_figures(fig_dir),
        plot_wp2_figures(fig_dir),
        plot_wp3_figures(fig_dir),
        plot_wp4_figures(fig_dir),
        plot_wp5_figures(fig_dir),
    ]
    for p in outs:
        logger.info(f"Generated figure -> {p}")
    return outs


if __name__ == "__main__":
    generate_all_phase2_figures()
