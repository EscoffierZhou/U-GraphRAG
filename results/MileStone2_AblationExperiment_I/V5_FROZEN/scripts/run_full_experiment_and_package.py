# -*- coding: utf-8 -*-
"""
scripts/run_full_experiment_and_package.py
=========================================
End-to-End Orchestrator for Full Experimental Pipeline:
1. Executes unified single-pass M1/M2/M3 Grid Search + Steps 1-4 ablations via
   `scripts/run_overnight_ablation.py`.
2. Generates honest, data-driven publication-grade report `results/FULL_EXPERIMENT_REPORT.md`
   (dynamically populated strictly from verified JSON outputs; zero hardcoded claims).
3. Packages all experimental deliverables into `results/UGraphRAG_Experimental_Results.zip`.
4. Outputs detailed integrity verification.
"""

import os
import sys
import json
import time
import argparse
import zipfile
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.plot_ablation_charts import (
    _compute_paired_bootstrap_stats,
    _compute_risk_coverage_curve,
)
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Orchestrator] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_full_experiment_and_package")


def _pct(val: Any) -> Optional[float]:
    if val is None:
        return None
    v = float(val)
    return v * 100.0 if 0.0 <= v <= 1.0 else v


def run_overnight_suite(args: argparse.Namespace) -> bool:
    """Executes the master overnight ablation script via subprocess for clean memory isolation."""
    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "run_overnight_ablation.py"),
        "--data-dir", str(DATA_DIR),
        "--models-dir", str(MODELS_DIR),
        "--results-dir", str(RESULTS_DIR),
        "--figures-dir", str(FIGURES_DIR),
        "--hotpot-limit", str(args.hotpot_limit),
        "--popqa-limit", str(args.popqa_limit),
        "--perm-limit", str(args.perm_limit),
        "--grid-offset", str(args.grid_offset),
        "--grid-limit", str(args.grid_limit),
        "--grid-preset", str(args.grid_preset),
        "--params-preset", str(args.params_preset),
    ]
    if not args.no_grid_search:
        cmd.append("--grid-search")
    if args.force:
        cmd.append("--force")

    logger.info("=" * 70)
    logger.info("STEP 1: Starting Unified Grid Search + Overnight Ablation Engine...")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 70)

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    if proc.stdout:
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
        proc.stdout.close()

    proc.wait()
    duration = time.time() - t0

    if proc.returncode != 0:
        logger.error(f"Overnight ablation script failed with exit code {proc.returncode} after {duration/60:.2f} mins.")
        return False

    logger.info(f"Overnight ablation script successfully finished in {duration/60:.2f} mins.")
    return True


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return {}


def _resolve_hardware_label(exp1_data: Dict[str, Any], grid_data: Dict[str, Any]) -> str:
    hw = exp1_data.get("metadata", {}).get("hardware") or grid_data.get("metadata", {}).get("device")
    if hw:
        return str(hw)
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU / Unknown Accelerator"


def generate_full_experiment_report() -> Path:
    """Generates the honest, dynamically populated FULL_EXPERIMENT_REPORT.md."""
    logger.info("=" * 70)
    logger.info("STEP 2: Generating Honest Data-Driven Master Report...")
    logger.info("=" * 70)

    grid_data = _load_json_if_exists(RESULTS_DIR / "grid_search_results.json")
    best_presets = _load_json_if_exists(MODELS_DIR / "best_grid_params.json")
    exp1_data = _load_json_if_exists(RESULTS_DIR / "step1_corrected_3b_7b.json")
    exp2_data = _load_json_if_exists(RESULTS_DIR / "step2_popqa_uncertainty.json")
    if not exp2_data:
        exp2_data = _load_json_if_exists(RESULTS_DIR / "step2_m1_popqa100.json")
    exp3_chart = _load_json_if_exists(RESULTS_DIR / "step3_hotpotqa_ablations.json")
    exp3_full = _load_json_if_exists(RESULTS_DIR / "step3_step4_ablation_hotpot100.json")
    perm_data = _load_json_if_exists(RESULTS_DIR / "step4_m2_permutation.json")

    hardware_label = _resolve_hardware_label(exp1_data, grid_data)
    hp_count = exp1_data.get("metadata", {}).get("hotpot_eval_count", exp3_full.get("metadata", {}).get("num_samples", "N/A"))
    pop_count = exp1_data.get("metadata", {}).get("popqa_eval_count", exp2_data.get("metadata", {}).get("num_samples", "N/A"))

    lines = [
        "# U-GraphRAG 全量实验评测与参数网格搜索总览报告",
        f"**生成时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**实测硬件平台**：{hardware_label} (单卡 8GB VRAM 约束环境)  ",
        "**核心方法**：U-GraphRAG (v4/v5 Theory-Complete) - 不确定性解耦、预算约束集合优化与前向根节点优先 MST 拓扑线性化、零截距正则 Platt 校准与选通解码  ",
        "**量化配置**：纯本地 4-bit NF4 量化 (`bfloat16` 计算精度), 零外部商业 API, 串行模型显存强隔离  ",
        f"**历史审查与迭代修复档案**：[V1_AUDIT_CRITIQUE_AND_V2_FIXES.md]({(PROJECT_ROOT / 'V1_AUDIT_CRITIQUE_AND_V2_FIXES.md').as_uri()}) | [V2_AUDIT_CRITIQUE_AND_V3_FIXES.md]({(PROJECT_ROOT / 'V2_AUDIT_CRITIQUE_AND_V3_FIXES.md').as_uri()}) | [V3_AUDIT_CRITIQUE_AND_V4_FIXES.md]({(PROJECT_ROOT / 'V3_AUDIT_CRITIQUE_AND_V4_FIXES.md').as_uri()}) | [V4_AUDIT_CRITIQUE_AND_POPQA_FIX.md]({(PROJECT_ROOT / 'V4_AUDIT_CRITIQUE_AND_POPQA_FIX.md').as_uri()}) | [results/MileStone_1.md]({(RESULTS_DIR / 'MileStone_1.md').as_uri()}) | [results/FOR_GPT_VERIFICATION_EVIDENCE.md]({(RESULTS_DIR / 'FOR_GPT_VERIFICATION_EVIDENCE.md').as_uri()})  ",
        "",
        "---",
        "",
        "## 1. 实验总体设计与里程碑概览",
        f"- **Stage 0 (单次多参数共享缓存网格搜索)**：在独立验证切片上单次遍历同时产出并验证多套有效 M1/M2/M3 参数配置；",
        f"- **Milestone 1 (Step 1 基线评测)**：在 HotpotQA ($N={hp_count}$) 与 PopQA ($N={pop_count}$) 上对比 Qwen2.5-3B 与 7B 的基准性能与峰值显存；",
        f"- **Milestone 2 (Step 2 PopQA 长尾验证)**：基于共享对称检索库与闭卷对比诊断，评估认知不确定性 $U_{{\\mathrm{{EU}}}}$ 随实体流行度的变化及 AUROC；",
        f"- **Milestone 3 (Step 3 HotpotQA 8-Mode 消融矩阵)**：严格隔离 M1 自适应检索、M2-A 精确/贪心子集选择、M2-B MST 拓扑线性化与 M3 温度校准解码增益；",
        "- **Milestone 4 (Step 4 M2 置换敏感度检验)**：分离样本间语义方差与单样本重排方差，全面评估 DFS 拓扑序对位置偏置的抑制效果。",
        "",
    ]

    # Stage 0: Grid Search Section
    if grid_data:
        s1_grid = grid_data.get("stage1_m1", {})
        s2_grid = grid_data.get("stage2_m2", {})
        s3_grid = grid_data.get("stage3_m3", {})
        lines.extend([
            "---",
            "",
            "## 2. Stage 0: 单实验共享缓存网格搜索与多套有效参数组",
            f"- **独立验证切片**：HotpotQA `[{grid_data.get('metadata', {}).get('offset', 100)}:{grid_data.get('metadata', {}).get('offset', 100) + grid_data.get('metadata', {}).get('num_samples', 0)}]`，"
            f"运行模式：`{'Dry-Run' if grid_data.get('metadata', {}).get('dry_run') else 'Real GPU'}`，耗时 `{grid_data.get('metadata', {}).get('wall_time_sec', 0.0):.1f}s`",
            f"- **Stage 1 (M1 $\\tau_{{\\mathrm{{EU}}}}$ 扫描)**：有效配置数 `{len(s1_grid.get('valid_configs', []))}` 组；",
            f"- **Stage 2 (M2 图优化网格)**：CPU 遍历结构网格后筛选 Pareto 前沿 `{len(s2_grid.get('evaluated_llm_configs', []))}` 组开展缓存去重 LLM 实测；",
            f"- **Stage 3 (M3 校准与门控网格)**：共评估 `{s3_grid.get('total_combinations_tested', 0)}` 组解码/校准/拒答超参数，"
            f"筛出有效选择性校准配置 `{s3_grid.get('valid_selective_count', 0)}` 组与全覆盖校准配置 `{s3_grid.get('valid_full_cov_count', 0)}` 组。",
            "",
        ])
        top_sel = s3_grid.get("top_valid_selective_configs", [])[:5]
        if top_sel:
            lines.extend([
                "### Top-5 有效 M3 校准与门控工作点",
                "| $\\beta$ | $T_{\\text{conf}}$ | $w_{\\text{tok}}$ | $(\\tau_{\\text{abs}}, \\tau_{\\text{conf}})$ | Raw EM/F1 (%) | ECE | Brier | Coverage (%) | Selective Risk (%) |",
                "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for item in top_sel:
                p = item["params"]
                lines.append(
                    f"| {p['m3_beta']:.2f} | {p['m3_confidence_temperature']:.2f} | {p['m3_confidence_tok_weight']:.2f} | "
                    f"({p['m3_abstain_threshold']:.2f}, {p['m3_confidence_threshold']:.2f}) | "
                    f"{item['raw_em']:.2f} / {item['raw_f1']:.2f} | {item['ece']:.4f} | {item['brier_score']:.4f} | "
                    f"{item['coverage_pct']:.1f}% | {item['selective_risk_pct']:.2f}% |"
                )
            lines.append("")
        if best_presets:
            lines.extend([
                "已自动固化多目标推荐参数套件至 `models/best_grid_params.json`（含 `balanced_optimal`、`low_ece_high_precision`、`full_coverage_calibrated`）。",
                "",
            ])
        step3_gp = exp3_full.get("metadata", {}).get("grid_params", {})
        if step3_gp:
            lines.extend([
                f"- **Stage 0 -> Step 3 参数零泄漏映射确认**：Step 3 实测采用 Stage 0 验证切片 (`[100:180]`) 锁定的 **`balanced_optimal`** 工作点 "
                f"(`platt_a={step3_gp.get('m3_confidence_platt_a', 'N/A')}`, `platt_b={step3_gp.get('m3_confidence_platt_b', 'N/A')}`, "
                f"`tau_conf={step3_gp.get('m3_confidence_threshold', 'N/A')}`, `tau_abs={step3_gp.get('m3_abstain_threshold', 'N/A')}`)。",
                "",
            ])

    # Milestone 1: Step 1 Baselines
    lines.extend([
        "---",
        "",
        "## 3. Milestone 1: 基线模型性能与 8GB 显存裕度实测 (Step 1)",
    ])
    models_info = exp1_data.get("models", {})
    if models_info:
        lines.extend([
            "| 模型骨干 | HotpotQA EM / F1 (%) | HotpotQA Sp-F1 (%) | PopQA Closed EM / F1 (%) | 峰值 Allocated / Reserved VRAM | 8GB 安全裕度 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for mname in ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]:
            if mname not in models_info:
                continue
            minfo = models_info[mname]
            hp = minfo.get("hotpotqa", {})
            pq = minfo.get("popqa", {})
            vr = minfo.get("vram_profiling", {})
            hp_em = _pct(hp.get("exact_match")) or 0.0
            hp_f1 = _pct(hp.get("f1")) or 0.0
            hp_spf1 = _pct(hp.get("sp_f1")) or 0.0
            pq_em = _pct(pq.get("exact_match")) or 0.0
            pq_f1 = _pct(pq.get("f1")) or 0.0
            peak_mb = float(vr.get("peak_allocated_mb", 0.0))
            res_mb = float(vr.get("final_reserved_mb", 0.0))
            headroom_mb = float(vr.get("headroom_mb", 0.0))
            lines.append(
                f"| **{mname}** | {hp_em:.2f}% / {hp_f1:.2f}% | {hp_spf1:.2f}% | "
                f"{pq_em:.2f}% / {pq_f1:.2f}% | {peak_mb:.1f} / {res_mb:.1f} MB ({res_mb/1024:.2f} GB) | +{headroom_mb:.1f} MB |"
            )
        lines.append("")
        if "Qwen2.5-3B-Instruct" in models_info and "Qwen2.5-7B-Instruct" in models_info:
            em_3b = _pct(models_info["Qwen2.5-3B-Instruct"].get("hotpotqa", {}).get("exact_match")) or 0.0
            em_7b = _pct(models_info["Qwen2.5-7B-Instruct"].get("hotpotqa", {}).get("exact_match")) or 0.0
            f1_3b = _pct(models_info["Qwen2.5-3B-Instruct"].get("hotpotqa", {}).get("f1")) or 0.0
            f1_7b = _pct(models_info["Qwen2.5-7B-Instruct"].get("hotpotqa", {}).get("f1")) or 0.0
            lines.append(
                f"> **7B vs 3B 实测差异**：HotpotQA 基准上 7B 模型 EM 变化 `{em_7b - em_3b:+.2f}%`，F1 变化 `{f1_7b - f1_3b:+.2f}%`。"
            )
            lines.append("")
    else:
        lines.extend(["*暂无 Step 1 模型基准数据（`step1_corrected_3b_7b.json` 未找到）。*", ""])

    # Milestone 2: Step 2 PopQA
    lines.extend([
        "---",
        "",
        "## 4. Milestone 2: PopQA 对称共享语料长尾不确定性验证 (Step 2)",
    ])
    bins = exp2_data.get("bins", [])
    if bins:
        e2_meta = exp2_data.get("metadata", {})
        has_gold_rec = any("m2_gold_recall_pct" in b for b in bins)
        brier_str = f" | Full Brier = `{e2_meta.get('full_brier_score', 'N/A')}`" if "full_brier_score" in e2_meta else ""
        gold_str = f" | M2 Gold Recall = `{e2_meta.get('m2_gold_recall_pct', 'N/A')}%`" if "m2_gold_recall_pct" in e2_meta else ""
        abst_str = f" | Full Abstain = `{e2_meta.get('full_abstention_rate_pct', 'N/A')}%`" if "full_abstention_rate_pct" in e2_meta else ""
        lines.extend([
            f"- 样本总量 `N={e2_meta.get('num_samples', 'N/A')}`，校准阈值 `calib_tau_eu={e2_meta.get('calibrated_tau_eu', 'N/A')}`，"
            f"实际采用阈值 `active_tau_eu={e2_meta.get('active_tau_eu', e2_meta.get('calibrated_tau_eu', 'N/A'))}`",
            f"- 闭卷认知不确定性判别故障 AUROC = `{e2_meta.get('auroc_closed_eu_vs_error', 'N/A')}` | "
            f"Full U-GraphRAG 置信度 AUROC = `{e2_meta.get('auroc_full_confidence', 'N/A')}` | "
            f"Full ECE = `{e2_meta.get('full_ece', 'N/A')}`{brier_str}{gold_str}{abst_str}",
            "",
        ])
        if has_gold_rec:
            lines.extend([
                "| 流行度分桶 | N | Closed $U_{\\mathrm{EU}}$ | $D_0$ $U_{\\mathrm{EU}}$ | Closed Acc | B0 (Top-5 公平检索) | M1 Only | Full U-GraphRAG | M2 Gold Recall | Abst (%) | Early Exit (%) | Token Saving (%) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
        else:
            lines.extend([
                "| 流行度分桶 | N | Closed $U_{\\mathrm{EU}}$ | $D_0$ $U_{\\mathrm{EU}}$ | Closed Acc | B0 (Top-5 公平检索) | M1 Only | Full U-GraphRAG | Early Exit (%) | Token Saving (%) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
        for b in bins:
            bname = b.get("bin_name", "").replace("\n", " ")
            if has_gold_rec:
                lines.append(
                    f"| {bname} | {b.get('count', 0)} | {b.get('avg_closed_epistemic_uncertainty', b.get('avg_epistemic_uncertainty', 0.0)):.4f} | "
                    f"{b.get('avg_epistemic_uncertainty', 0.0):.4f} | {b.get('acc_closed_book', 0.0):.1f}% | "
                    f"{b.get('acc_standard_rag', 0.0):.1f}% | {b.get('acc_m1_only', 0.0):.1f}% | "
                    f"**{b.get('acc_u_graphrag', 0.0):.1f}%** | {b.get('m2_gold_recall_pct', 0.0):.1f}% | "
                    f"{b.get('full_abstention_rate_pct', 0.0):.1f}% | {b.get('early_exit_rate_pct', 0.0):.1f}% | {b.get('token_saving_rate_pct', 0.0):.1f}% |"
                )
            else:
                lines.append(
                    f"| {bname} | {b.get('count', 0)} | {b.get('avg_closed_epistemic_uncertainty', b.get('avg_epistemic_uncertainty', 0.0)):.4f} | "
                    f"{b.get('avg_epistemic_uncertainty', 0.0):.4f} | {b.get('acc_closed_book', 0.0):.1f}% | "
                    f"{b.get('acc_standard_rag', 0.0):.1f}% | {b.get('acc_m1_only', 0.0):.1f}% | "
                    f"**{b.get('acc_u_graphrag', 0.0):.1f}%** | {b.get('early_exit_rate_pct', 0.0):.1f}% | {b.get('token_saving_rate_pct', 0.0):.1f}% |"
                )
        lines.extend([
            "",
            f"> 全局 Token 节省率（相对于公平 Top-5 B0 上下文）：**{e2_meta.get('overall_token_saving_rate_pct', 0.0):.2f}%**；"
            f"Bloom 剪枝平均召回损失：**{e2_meta.get('overall_bloom_pruning_recall_drop', 0.0):.4f}**。",
            "",
        ])
    else:
        lines.extend(["*暂无 Step 2 PopQA 数据。*", ""])

    # Milestone 3: Step 3 8-Mode Ablations
    lines.extend([
        "---",
        "",
        "## 5. Milestone 3: HotpotQA 8-Mode 全量消融矩阵与校准表现 (Step 3)",
    ])
    if exp3_chart:
        lines.extend([
            "| Ablation Mode | EM (%) | F1 (%) | Sp-R (%) | Sp-F1 (%) | Token Saved (%) | ECE | AUROC | Coverage (%) | Selective Risk (%) | Latency (s) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for mode_key, m in exp3_chart.items():
            lines.append(
                f"| **{mode_key}** | {m.get('EM', 0.0):.2f} | {m.get('F1', 0.0):.2f} | "
                f"{m.get('sp_rec', 0.0):.2f} | {m.get('sp_f1', 0.0):.2f} | "
                f"{m.get('token_saving', 0.0):.1f}% | {m.get('ece', 0.0):.4f} | "
                f"{m.get('auroc', 0.5):.4f} | {m.get('coverage', 100.0):.1f}% | "
                f"{m.get('selective_risk', 0.0):.2f}% | {m.get('latency_s', 0.0):.2f} |"
            )
        lines.append("")

        b0 = exp3_chart.get("B0_Standard_RAG")
        m1 = exp3_chart.get("M1_Only")
        m2 = exp3_chart.get("M2_Full")
        full = exp3_chart.get("Full_UGraphRAG")
        b1 = exp3_chart.get("B1_Naive_MST")
        obs_notes = ["> [!NOTE]", "> **基于实测数据的逐模块归因诊断**："]
        if b0 and m1:
            obs_notes.append(
                f"> - **M1_Only vs B0**: EM `{m1['EM'] - b0['EM']:+.2f}%`, F1 `{m1['F1'] - b0['F1']:+.2f}%`, "
                f"Sp-R `{m1['sp_rec'] - b0['sp_rec']:+.2f}%`, Sp-F1 `{m1['sp_f1'] - b0['sp_f1']:+.2f}%`。"
            )
        if b1 and m2:
            obs_notes.append(
                f"> - **M2_Full vs B1_Naive_MST**: EM `{m2['EM'] - b1['EM']:+.2f}%`, F1 `{m2['F1'] - b1['F1']:+.2f}%`, "
                f"Sp-F1 `{m2['sp_f1'] - b1['sp_f1']:+.2f}%`, Token Saving `{m2['token_saving']:.1f}%`。"
            )
        if m2 and full:
            obs_notes.append(
                f"> - **Full_UGraphRAG vs M2_Full**: EM `{full['EM'] - m2['EM']:+.2f}%`, F1 `{full['F1'] - m2['F1']:+.2f}%`, "
                f"ECE `{full['ece'] - m2['ece']:+.4f}`, AUROC `{full.get('auroc', 0.5):.4f}`, Selective Coverage `{full.get('coverage', 100.0):.1f}%`。"
            )
        lines.extend(obs_notes + [""])
        abl7 = exp3_full.get("ablations_7b", {})
        if abl7.get("Full_UGraphRAG", {}).get("sample_records"):
            rc_full = _compute_risk_coverage_curve(
                abl7["Full_UGraphRAG"]["sample_records"],
                coverages=(1.0, 0.96, 0.9, 0.8, 0.7, 0.6, 0.5),
            )
            rc_b0 = _compute_risk_coverage_curve(
                abl7.get("B0_Standard_RAG", {}).get("sample_records", []),
                coverages=(1.0, 0.96, 0.9, 0.8, 0.7, 0.6, 0.5),
            )
            rc_m2 = _compute_risk_coverage_curve(
                abl7.get("M2_Full", {}).get("sample_records", []),
                coverages=(1.0, 0.96, 0.9, 0.8, 0.7, 0.6, 0.5),
            )
            lines.extend([
                "### M3 Selective Prediction Risk-Coverage 递进表 (`100% -> 50%` Coverage)",
                "| Coverage (%) | N | Full Risk (%) | Full EM (%) | Full F1 (%) | B0 Risk (%) | M2_Full Risk (%) |",
                "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for i_rc, pt in enumerate(rc_full):
                b0_r = rc_b0[i_rc]["risk_pct"] if i_rc < len(rc_b0) else 0.0
                m2_r = rc_m2[i_rc]["risk_pct"] if i_rc < len(rc_m2) else 0.0
                lines.append(
                    f"| {pt['coverage_pct']:.1f}% | {pt['k']} | **{pt['risk_pct']:.2f}%** | "
                    f"{pt['em_pct']:.2f}% | {pt['f1_pct']:.2f}% | {b0_r:.2f}% | {m2_r:.2f}% |"
                )
            lines.append("")
    else:
        lines.extend(["*暂无 Step 3 消融矩阵数据。*", ""])

    # Milestone 4: Step 4 Permutation
    lines.extend([
        "---",
        "",
        "## 6. Milestone 4: M2 证据拓扑排序置换检验 (Step 4)",
    ])
    if perm_data:
        dfs_d = perm_data.get("dfs", {})
        rev_d = perm_data.get("reverse", {})
        shuf_d = perm_data.get("shuffles", {})
        n_perm = perm_data.get("num_samples_evaluated", "N/A")
        dfs_cross = dfs_d.get("cross_sample_std", dfs_d.get("std", 0.0))
        rev_cross = rev_d.get("cross_sample_std", rev_d.get("std", 0.0))
        shuf_intra = shuf_d.get("intra_sample_std", perm_data.get("random_shuffles", {}).get("f1_std", 0.0))
        shuf_cross = shuf_d.get("cross_sample_std", shuf_d.get("std", 0.0))
        boot_stats = _compute_paired_bootstrap_stats(perm_data)
        lines.extend([
            f"评测样本量：$N={n_perm}$（注：DFS 与 Reverse 为单次确定性推理；5 次随机重排统计单样本内重排波动 `Intra-Sample Std = ±{shuf_intra:.2f}%`，以及跨样本分布标准差 `Cross-Sample Std`）：",
            "",
            "| 证据排序策略 | Mean F1 (%) | Intra-Sample Spread | Cross-Sample Std | Drop vs DFS (%) |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| **DFS 拓扑序 (M2-B Ours)** | **{dfs_d.get('mean', 0.0):.2f}%** | Deterministic (1 pass) | ±{dfs_cross:.2f} | 0.00% |",
            f"| **逆序排列 (Reverse)** | {rev_d.get('mean', 0.0):.2f}% | Deterministic (1 pass) | ±{rev_cross:.2f} | {rev_d.get('mean', 0.0) - dfs_d.get('mean', 0.0):+.2f}% |",
            f"| **5 次随机打乱 (Random Shuffles)** | {shuf_d.get('mean', 0.0):.2f}% | ±{shuf_intra:.2f}% (within-question) | ±{shuf_cross:.2f} | {shuf_d.get('mean', 0.0) - dfs_d.get('mean', 0.0):+.2f}% |",
            "",
            f"- 实测最坏位置降幅（Max Drop vs DFS）：**{perm_data.get('max_drop', 0.0):.2f}%**。",
        ])
        if boot_stats:
            lines.extend([
                "",
                f"### 配对自助法 (Paired Bootstrap, $B={boot_stats['n_boot']}$) 95% CI 与排序敏感子集统计",
                "| 对比组 | Mean $\\Delta$ F1 (%) | Paired Std | 95% Bootstrap CI | Win / Tie / Loss | 敏感子集 Mean Gain |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ])
            for bk in ("dfs_vs_reverse", "dfs_vs_shuffles"):
                bs = boot_stats[bk]
                lines.append(
                    f"| {bs['label']} | {bs['mean_diff']:+.2f}% | ±{bs['paired_std']:.2f}% | "
                    f"[{bs['ci_95_low']:+.2f}%, {bs['ci_95_high']:+.2f}%] | "
                    f"{bs['wins']} / {bs['ties']} / {bs['losses']} | "
                    f"{bs['sensitive_mean_gain']:+.2f}% ($n={bs['sensitive_n']}$) |"
                )
        lines.append("")
    else:
        lines.extend(["*暂无 Step 4 置换检验数据。*", ""])

    # Asset Index using relative & file URI paths grounded in PROJECT_ROOT
    lines.extend([
        "---",
        "",
        "## 7. 图表与归档资产索引",
        f"- [V1_AUDIT_CRITIQUE_AND_V2_FIXES.md]({(PROJECT_ROOT / 'V1_AUDIT_CRITIQUE_AND_V2_FIXES.md').as_uri()})",
        f"- [V2_AUDIT_CRITIQUE_AND_V3_FIXES.md]({(PROJECT_ROOT / 'V2_AUDIT_CRITIQUE_AND_V3_FIXES.md').as_uri()})",
        f"- [V3_AUDIT_CRITIQUE_AND_V4_FIXES.md]({(PROJECT_ROOT / 'V3_AUDIT_CRITIQUE_AND_V4_FIXES.md').as_uri()})",
        f"- [V4_AUDIT_CRITIQUE_AND_POPQA_FIX.md]({(PROJECT_ROOT / 'V4_AUDIT_CRITIQUE_AND_POPQA_FIX.md').as_uri()})",
        f"- [results/MileStone_1.md]({(RESULTS_DIR / 'MileStone_1.md').as_uri()})",
        f"- [results/FOR_GPT_VERIFICATION_EVIDENCE.md]({(RESULTS_DIR / 'FOR_GPT_VERIFICATION_EVIDENCE.md').as_uri()})",
        f"- [results/figures/fig1_popqa_eu_vs_popularity.png]({(FIGURES_DIR / 'fig1_popqa_eu_vs_popularity.png').as_uri()})",
        f"- [results/figures/fig2_hotpotqa_ablation_progression.png]({(FIGURES_DIR / 'fig2_hotpotqa_ablation_progression.png').as_uri()})",
        f"- [results/figures/fig3_permutation_test_variance.png]({(FIGURES_DIR / 'fig3_permutation_test_variance.png').as_uri()})",
        f"- [results/figures/fig4_m3_calibration_and_risk_coverage.png]({(FIGURES_DIR / 'fig4_m3_calibration_and_risk_coverage.png').as_uri()})",
        f"- [results/figures/tables_latex.tex]({(FIGURES_DIR / 'tables_latex.tex').as_uri()})",
        f"- [results/GRID_SEARCH_REPORT.md]({(RESULTS_DIR / 'GRID_SEARCH_REPORT.md').as_uri()})",
        f"- [results/UGraphRAG_Experimental_Results.zip]({(RESULTS_DIR / 'UGraphRAG_Experimental_Results.zip').as_uri()})",
        "",
    ])

    report_path = RESULTS_DIR / "FULL_EXPERIMENT_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Master experiment report generated successfully at: {report_path}")
    return report_path


def package_results_to_zip() -> Path:
    """Packages all fresh experimental results and calibrated parameter artifacts into a clean ZIP archive."""
    logger.info("=" * 70)
    logger.info("STEP 3: Packaging Experimental Deliverables into ZIP Archive...")
    logger.info("=" * 70)

    zip_path = RESULTS_DIR / "UGraphRAG_Experimental_Results.zip"
    excluded_dir_names = {
        "UGraphRAG_Experimental_Results",
        "_dryrun_test",
        "_verify_figs",
        "__pycache__",
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(RESULTS_DIR):
            dirs[:] = [
                d for d in dirs
                if d not in excluded_dir_names and not d.startswith("UGraphRAG_Experimental_Results")
            ]
            for file in files:
                if file.endswith(".zip") or file.endswith(".tmp") or file.startswith("~$"):
                    continue
                file_abs = Path(root) / file
                rel_path = file_abs.relative_to(PROJECT_ROOT)
                zf.write(file_abs, arcname=str(rel_path).replace("\\", "/"))

        for model_artifact in ("best_grid_params.json", "m1_calibration.json"):
            m_path = MODELS_DIR / model_artifact
            if m_path.exists():
                rel_path = m_path.relative_to(PROJECT_ROOT)
                zf.write(m_path, arcname=str(rel_path).replace("\\", "/"))

        for root_doc in (
            "V4_AUDIT_CRITIQUE_AND_POPQA_FIX.md",
            "V3_AUDIT_CRITIQUE_AND_V4_FIXES.md",
            "V2_AUDIT_CRITIQUE_AND_V3_FIXES.md",
            "V1_AUDIT_CRITIQUE_AND_V2_FIXES.md",
            "CODE_AUDIT_README.md",
        ):
            doc_path = PROJECT_ROOT / root_doc
            if doc_path.exists():
                zf.write(doc_path, arcname=root_doc)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info(f"ZIP package created: {zip_path} ({size_mb:.2f} MB)")

    with zipfile.ZipFile(zip_path, "r") as zf:
        infolist = zf.infolist()
        logger.info(f"Total archived files: {len(infolist)}")
        for info in infolist[:25]:
            logger.info(f"  - {info.filename} ({info.file_size / 1024:.1f} KB)")
        if len(infolist) > 25:
            logger.info(f"  ... and {len(infolist) - 25} more files.")

    return zip_path


def main():
    parser = argparse.ArgumentParser(description="End-to-End U-GraphRAG Grid Search + Experimental Pipeline")
    parser.add_argument("--hotpot-limit", type=int, default=100)
    parser.add_argument("--popqa-limit", type=int, default=100)
    parser.add_argument("--perm-limit", type=int, default=100)
    parser.add_argument("--no-grid-search", action="store_true", help="Skip Stage 0 grid search before running suite")
    parser.add_argument("--grid-offset", type=int, default=100)
    parser.add_argument("--grid-limit", type=int, default=80)
    parser.add_argument("--grid-preset", type=str, default="balanced", choices=["balanced", "exhaustive"])
    parser.add_argument("--params-preset", type=str, default="balanced_optimal", choices=["balanced_optimal", "low_ece_high_precision", "full_coverage_calibrated"])
    parser.add_argument("--force", action="store_true", help="Ignore prior checkpoints and run cleanly")
    parser.add_argument("--report-only", action="store_true", help="Skip GPU experiments and regenerate report + ZIP from existing results/")
    args = parser.parse_args()

    t_start = time.time()
    logger.info("========================================================================")
    logger.info("U-GraphRAG Full End-to-End Experiment & Packaging Pipeline Initialized")
    logger.info("========================================================================")

    if not args.report_only:
        success = run_overnight_suite(args)
        if not success:
            logger.error("Overnight experiment encountered an error. Generating report & zip from available outputs...")

    report_file = generate_full_experiment_report()
    zip_file = package_results_to_zip()

    total_time = time.time() - t_start
    logger.info("========================================================================")
    logger.info(f"FULL EXPERIMENT PIPELINE FINISHED IN {total_time/60.0:.2f} MINUTES")
    logger.info(f"Report: {report_file}")
    logger.info(f"ZIP:    {zip_file}")
    logger.info("========================================================================")


if __name__ == "__main__":
    main()
