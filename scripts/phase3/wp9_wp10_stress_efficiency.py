"""
wp9_wp10_stress_efficiency.py
==============================
Milestone 3 Phase D (WP9 & WP10): Controlled Single-Variable Adversarial Stress Curves
and Cross-Family Hardware Efficiency / Quality-Latency-VRAM Benchmarking.

WP9 Single-Variable Controlled Perturbation Axes (AboutMileStone_3.md Sec 9):
  - Axis A1 (Irrelevant Distractor Noise):      {0, 20, 40, 60}%
  - Axis A2 (Near-Miss Entity Distractors):     {0, 30, 60}% (entity-similar wrong passages)
  - Axis A3 (Gold Evidence Starvation):         {0, 50, 100}%
  - Axis B1 (Single-Source Conflict Intensity): {0, 20, 40, 60}%
  - Axis B2 (Multi-Source Contradiction Count): {0, 1, 2} competing conflicting docs/item
  - Axis C  (Resource Budget Constraints):      pool_k / max_k in {(8,3), (8,5), (12,5), (16,5), (16,8)}
Compares Standard_RAG, B3_MMR_RAG, M2_Only, and Full_UGraphRAG and identifies inflection points.

WP10 Hardware Efficiency Across Backbones (AboutMileStone_3.md Sec 10):
  Measures real Peak/Mean VRAM (allocated & reserved MB), stage latencies
  (Retrieval, M1, M2, M3, E2E Mean/P50/P95), QPS & output Tokens/s throughput, Token cost,
  and M2 Exact vs Greedy Quality-Latency-VRAM Pareto across available models.

Writes strictly to `results/MileStone3/WP9_Robustness/` and `results/MileStone3/WP10_Efficiency/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cache_builder import _load_cpu_dense_model, inject_numeric_or_entity_conflict
from scripts.phase2.wp2_m2_ablation import _select_mmr
from scripts.phase2.wp3_m3_calibration import _collect_decode_traces, _fit_dev_calibration_configs, _score_trace_with_cal_params
from scripts.phase3.phase3_common import (
    WP3_DIRS,
    Phase2CheckpointManager,
    atomic_write_json,
    calibrate_dev_m1_tau_eu,
    cleanup_cached_model_weights,
    collect_hardware_manifest,
    compute_calibration_and_risk_suite,
    compute_next_token_distribution,
    compute_token_count,
    compute_unified_m1_signal,
    evaluate_qa_prediction,
    get_vram_snapshot_mb,
    is_phase3_backbone_available,
    load_phase3_dataset_cache,
    load_v5_balanced_params,
    reset_cuda_peak,
    resolve_phase3_model_list,
    resolve_wp8_recommended_graph_mode,
    run_open_decode_cached,
    select_dev_risk_constrained_threshold,
    select_m2_by_graph_mode,
    sequential_load_phase3_generator,
    setup_phase2_logger,
    should_bypass_m1_early_exit,
    teardown_generator,
    verify_v5_frozen_checksums,
)
from src.baseline_rag import HybridRetriever
from src.m1_adaptive_retriever import UncertaintyDecoupler
from src.m3_calibrated_decoder import CalibratedDecoderConfig, M3CalibratedDecoder

logger = setup_phase2_logger("phase3.wp9_10")


def _build_second_conflict_variant(anchor: Dict[str, Any], seed_key: str, avoid_answers: Sequence[str]) -> Dict[str, Any]:
    """Create a second independent conflicting passage differing from both gold and conflict #1."""
    c1 = inject_numeric_or_entity_conflict(anchor, seed_key=f"{seed_key}_v1", avoid_answers=avoid_answers)
    c2 = inject_numeric_or_entity_conflict(c1, seed_key=f"{seed_key}_v2_multi", avoid_answers=avoid_answers)
    c2["doc_id"] = f"{anchor.get('doc_id', 'doc')}__conflict2"
    c2["title"] = f"{anchor.get('title', 'Doc')} (Conflicting Source 2)"
    c2["is_synthetic_conflict"] = True
    return c2


def _eval_wp9_slice(
    axis_tag: str,
    level_label: str,
    perturbed_items: Sequence[Dict[str, Any]],
    generator: Any,
    params: Dict[str, Any],
    m2_graph_mode: str,
    ckpt: Phase2CheckpointManager,
    pool_k: int = 8,
    max_k: int = 5,
    m2_lambda_conflict: Optional[float] = None,
    decode_mem: Optional[Dict[str, Dict[str, Any]]] = None,
    dist_mem: Optional[Dict[str, Dict[str, float]]] = None,
    tau_eu_override: Optional[float] = None,
    cal_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate Standard_RAG, B3_MMR_RAG, and Full_UGraphRAG cleanly on a single-axis perturbation slice."""
    decoupler = UncertaintyDecoupler()
    tau_eu = float(tau_eu_override if tau_eu_override is not None else params.get("m1_tau_eu", 0.18))
    rel_off = float(params.get("m2_reliability_offset", 0.22))
    ab_tau = float(params.get("m3_abstain_threshold", 0.05))
    if cal_bundle and "full_platt" in cal_bundle:
        platt = cal_bundle["full_platt"]
        cal_temp = float(platt.get("T", params.get("m3_confidence_temperature", 2.0)))
        cal_wtok = float(platt.get("w_tok", params.get("m3_confidence_tok_weight", 0.6)))
        cal_pa = float(platt.get("platt_a", params.get("m3_confidence_platt_a", 0.8174)))
        cal_pb = float(platt.get("platt_b", params.get("m3_confidence_platt_b", -0.2053)))
        conf_tau = float(cal_bundle.get("adaptive_tau_conf", params.get("m3_confidence_threshold", 0.35)))
    else:
        cal_temp = float(params.get("m3_confidence_temperature", 2.0))
        cal_wtok = float(params.get("m3_confidence_tok_weight", 0.6))
        cal_pa = float(params.get("m3_confidence_platt_a", 0.8174))
        cal_pb = float(params.get("m3_confidence_platt_b", -0.2053))
        conf_tau = float(params.get("m3_confidence_threshold", 0.35))

    systems = ("B0_StandardRAG", "B3_MMR_RAG", "Full_UGraphRAG")
    sys_recs: Dict[str, List[Dict[str, Any]]] = {s: [] for s in systems}

    for it in perturbed_items:
        iid = str(it["id"])
        pool = list(it["perturbed_pool"])[:pool_k]
        golds = it["possible_answers"]

        for sname in systems:
            ckey = f"wp9_v2::{axis_tag}::{level_label}::{sname}::{iid}"
            if ckpt.has(ckey):
                sys_recs[sname].append(ckpt.get(ckey))
                continue

            eff_beta = 0.0
            if sname == "B0_StandardRAG":
                chosen = [{**d, "calibrated_reliability": float(d.get("score", 0.5))} for d in pool[: min(3, len(pool))]]
            elif sname == "B3_MMR_RAG":
                chosen = [{**d, "calibrated_reliability": float(d.get("score", 0.5))} for d in _select_mmr(pool, k_target=min(max_k, len(pool)))]
            else:
                d0 = pool[: min(3, len(pool))]
                eu = float(compute_unified_m1_signal(generator, it["question"], d0, eu_mode="v5_proxy", decoupler=decoupler)["u_eu"])
                conflict_bypass = should_bypass_m1_early_exit(pool[: min(5, len(pool))], m2_graph_mode)
                early_exit = bool(eu <= tau_eu and not conflict_bypass)
                if early_exit:
                    chosen = [{**d, "calibrated_reliability": float(min(1.0, float(d.get("score", 0.5)) + rel_off))} for d in d0]
                    eff_beta = 0.0
                else:
                    pdists: Dict[str, Dict[str, float]] = {}
                    for d in pool:
                        did = str(d["doc_id"])
                        txt = str(d.get("text", ""))
                        t_hash = hashlib.md5(txt.encode("utf-8")).hexdigest()[:12]
                        d_mkey = f"{iid}::{did}::{t_hash}"
                        if dist_mem is not None and d_mkey in dist_mem:
                            pdists[did] = dist_mem[d_mkey]
                        else:
                            d_ckey = f"wp9_dist::{d_mkey}"
                            if ckpt.has(d_ckey):
                                d_val = ckpt.get(d_ckey)
                            else:
                                d_val = compute_next_token_distribution(generator, it["question"], [txt], variant_seed=did)["dist"]
                                ckpt.record(d_ckey, d_val)
                            pdists[did] = d_val
                            if dist_mem is not None:
                                dist_mem[d_mkey] = d_val
                    chosen, _ = select_m2_by_graph_mode(
                        it["question"], pool, pdists, params, graph_mode=m2_graph_mode, pool_k=pool_k, max_k=max_k, lambda_conflict=m2_lambda_conflict
                    )
                    eff_beta = 0.12

            trace = run_open_decode_cached(
                generator=generator,
                question=it["question"],
                evidence_docs=chosen,
                eff_beta=eff_beta,
                params=params,
                ckpt=ckpt,
                cache_prefix=f"wp9_gen_v2::{iid}",
                memory_cache=decode_mem,
                max_new_tokens=48,
                calib_from_b0=True,
            )
            raw_answer = trace["raw_answer"]
            if sname == "Full_UGraphRAG":
                conf = float(_score_trace_with_cal_params(trace, cal_temp, cal_wtok, cal_pa, cal_pb))
                is_ab = (not chosen) or (float(trace["max_ev_rel"]) < ab_tau) or (conf < conf_tau)
            else:
                conf = float(trace["raw_confidence"])
                is_ab = not chosen

            qa = evaluate_qa_prediction(raw_answer, golds, is_abstained=is_ab)
            rec = {
                "id": iid,
                "raw_f1": qa["f1"],
                "raw_em": qa["em"],
                "effective_em": qa["effective_em"],
                "effective_f1": qa["effective_f1"],
                "confidence": round(conf, 4),
                "is_abstained": bool(is_ab),
                "conflict_retained": int(any(bool(d.get("is_synthetic_conflict")) for d in chosen)),
                "selected_tokens": sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in chosen),
            }
            ckpt.record(ckey, rec)
            sys_recs[sname].append(rec)

    ckpt.flush(force=True)
    summary = {}
    for sname, recs in sys_recs.items():
        n = max(1, len(recs))
        suite = compute_calibration_and_risk_suite(
            [r["confidence"] for r in recs], [r["raw_f1"] for r in recs], abstained_flags=[r["is_abstained"] for r in recs]
        )
        summary[sname] = {
            "n": len(recs),
            "em_pct": round(100.0 * sum(r["raw_em"] for r in recs) / n, 2),
            "f1_pct": round(100.0 * sum(r["raw_f1"] for r in recs) / n, 2),
            "effective_f1_pct": round(100.0 * sum(r.get("effective_f1", 0.0 if r["is_abstained"] else r["raw_f1"]) for r in recs) / n, 2),
            "ece": suite["ece"],
            "brier_score": suite["brier_score"],
            "auroc_confidence": suite["auroc_confidence"],
            "aurc": suite["aurc"],
            "selective_risk_pct": suite["selective_risk_pct"],
            "abstention_rate_pct": suite["abstention_rate_pct"],
            "harmful_abstention_rate_pct": suite["harmful_abstention_rate_pct"],
            "conflict_retention_pct": round(100.0 * sum(r["conflict_retained"] for r in recs) / n, 2),
            "mean_tokens": round(sum(r["selected_tokens"] for r in recs) / n, 1),
        }
    return summary


def run_wp9_stress_curves(
    items: Sequence[Dict[str, Any]],
    generator: Any,
    params: Dict[str, Any],
    m2_graph_mode: str,
    ckpt: Phase2CheckpointManager,
    m2_lambda_conflict: Optional[float] = None,
    tau_eu_override: Optional[float] = None,
    cal_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute controlled single-variable perturbation axes cleanly."""
    decode_mem: Dict[str, Dict[str, Any]] = {}
    dist_mem: Dict[str, Dict[str, float]] = {}
    donor_distractors = [d for it in items for d in it.get("pool", []) if not d.get("is_gold")]
    if not donor_distractors:
        donor_distractors = [d for it in items for d in it.get("pool", [])]

    # Axis A1: Irrelevant Distractor Noise
    noise_curve = {}
    for rate in (0.0, 0.2, 0.4, 0.6):
        lbl = f"noise_{int(rate*100)}pct"
        p_items = []
        for idx, it in enumerate(items):
            bp = [dict(d) for d in it["pool"][:8]]
            if rate > 0 and donor_distractors:
                rng = random.Random(int(hashlib.md5(f"wp9a1|{lbl}|{it['id']}".encode()).hexdigest()[:8], 16))
                n_rep = int(round(rate * max(0, len(bp) - 1)))
                slots = list(range(1, len(bp)))
                rng.shuffle(slots)
                for s in slots[:n_rep]:
                    dn = dict(donor_distractors[(idx * 7 + s) % len(donor_distractors)])
                    dn["doc_id"] = f"wp9_n_{it['id']}_{s}"
                    dn["is_gold"] = False
                    dn["score"] = bp[s]["score"]
                    bp[s] = dn
            p_items.append({**it, "perturbed_pool": bp})
        noise_curve[lbl] = _eval_wp9_slice(
            "axis_a1_noise", lbl, p_items, generator, params, m2_graph_mode, ckpt,
            m2_lambda_conflict=m2_lambda_conflict, decode_mem=decode_mem, dist_mem=dist_mem,
            tau_eu_override=tau_eu_override, cal_bundle=cal_bundle,
        )

    # Axis A2: Near-Miss Entity-Matched Confounders
    nearmiss_curve = {}
    for rate in (0.0, 0.3, 0.6):
        lbl = f"nearmiss_{int(rate*100)}pct"
        cutoff = int(round(rate * len(items)))
        p_items = []
        for idx, it in enumerate(items):
            bp = [dict(d) for d in it["pool"][:8]]
            if idx < cutoff and len(bp) >= 3:
                # Duplicate entity anchor from non-gold slot with altered misleading claim at high score
                non_gold = next((d for d in bp if not d.get("is_gold")), bp[-1])
                nm = inject_numeric_or_entity_conflict(non_gold, seed_key=f"nm_{it['id']}", avoid_answers=it["possible_answers"])
                nm["doc_id"] = f"nm_{it['id']}"
                nm["score"] = round(float(bp[0].get("score", 0.9)) * 0.96, 4)
                nm["is_synthetic_conflict"] = False
                bp = [bp[0], nm] + bp[1:7]
            p_items.append({**it, "perturbed_pool": bp})
        nearmiss_curve[lbl] = _eval_wp9_slice(
            "axis_a2_nearmiss", lbl, p_items, generator, params, m2_graph_mode, ckpt,
            m2_lambda_conflict=m2_lambda_conflict, decode_mem=decode_mem, dist_mem=dist_mem,
            tau_eu_override=tau_eu_override, cal_bundle=cal_bundle,
        )

    # Axis A3: Gold Evidence Starvation
    drop_curve = {}
    for rate in (0.0, 0.5, 1.0):
        lbl = f"gold_drop_{int(rate*100)}pct"
        p_items = []
        for it in items:
            bp = [dict(d) for d in it["pool"][:8]]
            g_idx = [i for i, d in enumerate(bp) if d.get("is_gold")]
            drop_s = set(g_idx) if rate >= 1.0 else (set(g_idx[: max(1, int(round(rate * len(g_idx))))]) if (rate > 0 and g_idx) else set())
            filt = [d for i, d in enumerate(bp) if i not in drop_s] or [{**bp[-1], "is_gold": False, "score": 0.2}]
            if rate >= 1.0:
                filt = [{**d, "score": round(min(0.45, float(d.get("score", 0.4)) * 0.55), 4)} for d in filt]
            p_items.append({**it, "perturbed_pool": filt})
        drop_curve[lbl] = _eval_wp9_slice(
            "axis_a3_drop", lbl, p_items, generator, params, m2_graph_mode, ckpt,
            m2_lambda_conflict=m2_lambda_conflict, decode_mem=decode_mem, dist_mem=dist_mem,
            tau_eu_override=tau_eu_override, cal_bundle=cal_bundle,
        )

    # Axis B1: Conflict Intensity
    conf_curve = {}
    for rate in (0.0, 0.2, 0.4, 0.6):
        lbl = f"conflict_{int(rate*100)}pct"
        cutoff = int(round(rate * len(items)))
        p_items = []
        for idx, it in enumerate(items):
            bp = [dict(d) for d in it["pool"][:8]]
            cd = it.get("synthetic_conflict_doc")
            if idx < cutoff and cd and len(bp) >= 2:
                cc = dict(cd)
                cc["score"] = round(float(bp[0].get("score", 0.9)) * 0.98, 4)
                bp = [bp[0], cc] + bp[1:7]
            p_items.append({**it, "perturbed_pool": bp})
        conf_curve[lbl] = _eval_wp9_slice(
            "axis_b1_conflict", lbl, p_items, generator, params, m2_graph_mode, ckpt,
            m2_lambda_conflict=m2_lambda_conflict, decode_mem=decode_mem, dist_mem=dist_mem,
            tau_eu_override=tau_eu_override, cal_bundle=cal_bundle,
        )

    # Axis B2: Multi-Source Contradictions {0, 1, 2}
    multisrc_curve = {}
    for n_src in (0, 1, 2):
        lbl = f"sources_{n_src}"
        p_items = []
        for it in items:
            bp = [dict(d) for d in it["pool"][:8]]
            cd1 = it.get("synthetic_conflict_doc")
            if n_src >= 1 and cd1 and len(bp) >= 2:
                cc1 = dict(cd1)
                cc1["score"] = round(float(bp[0].get("score", 0.9)) * 0.98, 4)
                extra = [cc1]
                if n_src >= 2:
                    anchor = next((d for d in bp if d.get("is_gold")), bp[0])
                    cc2 = _build_second_conflict_variant(anchor, str(it["id"]), it["possible_answers"])
                    cc2["score"] = round(float(bp[0].get("score", 0.9)) * 0.96, 4)
                    extra.append(cc2)
                bp = [bp[0]] + extra + bp[1 : max(1, 8 - len(extra))]
            p_items.append({**it, "perturbed_pool": bp})
        multisrc_curve[lbl] = _eval_wp9_slice(
            "axis_b2_multisrc", lbl, p_items, generator, params, m2_graph_mode, ckpt,
            m2_lambda_conflict=m2_lambda_conflict, decode_mem=decode_mem, dist_mem=dist_mem,
            tau_eu_override=tau_eu_override, cal_bundle=cal_bundle,
        )

    # Axis C: Resource Budget Sweep (pool_k, max_k)
    budget_curve = {}
    for pk, mk in ((8, 3), (8, 5), (12, 5), (16, 5), (16, 8)):
        lbl = f"pk{pk}_mk{mk}"
        p_items = [{**it, "perturbed_pool": list(it["pool"][:pk])} for it in items]
        budget_curve[lbl] = _eval_wp9_slice(
            "axis_c_budget", lbl, p_items, generator, params, m2_graph_mode, ckpt,
            pool_k=pk, max_k=mk, m2_lambda_conflict=m2_lambda_conflict, decode_mem=decode_mem, dist_mem=dist_mem,
            tau_eu_override=tau_eu_override, cal_bundle=cal_bundle,
        )

    # Detect inflection point where Full_UGraphRAG drops > 5 F1 points vs 0% baseline
    base_f1 = noise_curve["noise_0pct"]["Full_UGraphRAG"]["f1_pct"]
    noise_inflection = next(
        (lbl for lbl, v in noise_curve.items() if base_f1 - v["Full_UGraphRAG"]["f1_pct"] > 5.0),
        "none_above_5pp_drop",
    )

    return {
        "axis_a1_noise": noise_curve,
        "axis_a2_near_miss_entity": nearmiss_curve,
        "axis_a3_gold_drop": drop_curve,
        "axis_b1_conflict_intensity": conf_curve,
        "axis_b2_multi_source_conflict": multisrc_curve,
        "axis_c_resource_budget": budget_curve,
        "inflection_summary": {"noise_5pp_drop_point": noise_inflection},
    }


def run_wp10_efficiency_for_model(
    model_key: str,
    items: Sequence[Dict[str, Any]],
    generator: Any,
    params: Dict[str, Any],
    m2_graph_mode: str,
    dry_run: bool,
    m2_lambda_conflict: Optional[float] = None,
    tau_eu_override: Optional[float] = None,
    cal_bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Benchmark all 7 required WP10 configurations on real Latency, VRAM, Throughput, Cost, and Quality."""
    shared_dense = _load_cpu_dense_model(dry_run=dry_run)
    decoupler = UncertaintyDecoupler()
    tau_eu = float(tau_eu_override if tau_eu_override is not None else params.get("m1_tau_eu", 0.18))
    rel_off = float(params.get("m2_reliability_offset", 0.22))
    if cal_bundle and "full_platt" in cal_bundle:
        platt = cal_bundle["full_platt"]
        cal_temp = float(platt.get("T", params.get("m3_confidence_temperature", 2.0)))
        cal_wtok = float(platt.get("w_tok", params.get("m3_confidence_tok_weight", 0.6)))
        cal_pa = float(platt.get("platt_a", params.get("m3_confidence_platt_a", 0.8174)))
        cal_pb = float(platt.get("platt_b", params.get("m3_confidence_platt_b", -0.2053)))
        conf_tau = float(cal_bundle.get("adaptive_tau_conf", params.get("m3_confidence_threshold", 0.35)))
    else:
        cal_temp = float(params.get("m3_confidence_temperature", 2.0))
        cal_wtok = float(params.get("m3_confidence_tok_weight", 0.6))
        cal_pa = float(params.get("m3_confidence_platt_a", 0.8174))
        cal_pb = float(params.get("m3_confidence_platt_b", -0.2053))
        conf_tau = float(params.get("m3_confidence_threshold", 0.35))

    uncal = M3CalibratedDecoder(
        CalibratedDecoderConfig(
            abstain_threshold=0.0, confidence_threshold=0.0, beta=0.0, enable_contrastive=False,
            preserve_evidence_order=True, max_new_tokens=48, confidence_temperature=1.0, confidence_tok_weight=1.0,
        )
    )
    full_dec_b12 = M3CalibratedDecoder(
        CalibratedDecoderConfig(
            abstain_threshold=float(params.get("m3_abstain_threshold", 0.05)),
            confidence_threshold=conf_tau, beta=0.12, enable_contrastive=True,
            preserve_evidence_order=True, max_new_tokens=48,
            confidence_temperature=cal_temp,
            confidence_tok_weight=cal_wtok,
            confidence_platt_a=cal_pa,
            confidence_platt_b=cal_pb,
            contrastive_conf_guard=0.90, protect_eos=True,
        )
    )
    full_dec_b0 = M3CalibratedDecoder(
        CalibratedDecoderConfig(
            abstain_threshold=float(params.get("m3_abstain_threshold", 0.05)),
            confidence_threshold=conf_tau, beta=0.0, enable_contrastive=False,
            preserve_evidence_order=True, max_new_tokens=48,
            confidence_temperature=cal_temp,
            confidence_tok_weight=cal_wtok,
            confidence_platt_a=cal_pa,
            confidence_platt_b=cal_pb,
            contrastive_conf_guard=0.90, protect_eos=True,
        )
    )

    configs = ("Standard_RAG", "MMR_RAG", "M1_Only", "M1_M2", "M2_Exact_Enum", "M2_Greedy", "Full_UGraphRAG")
    out: Dict[str, Any] = {}

    for cfg in configs:
        lat_ret, lat_m1, lat_m2, lat_m3, lat_tot = [], [], [], [], []
        alloc_peaks, resv_peaks, tok_used, gen_tok_counts = [], [], [], []
        recs = []

        for it in items:
            pool = list(it["pool"])
            t_s = time.perf_counter()

            # Stage 1: Retrieval telemetry
            t0 = time.perf_counter()
            lret = HybridRetriever([{"id": d["doc_id"], "title": d["title"], "text": d["text"]} for d in pool], shared_dense_model=shared_dense)
            _ = lret.retrieve(it["question"], top_k=min(8, len(pool)))
            t_ret = (time.perf_counter() - t0) * 1000.0

            # Stage 2: M1 Probe
            t1 = time.perf_counter()
            early_exit_full = False
            if cfg in ("Standard_RAG", "MMR_RAG", "M2_Exact_Enum", "M2_Greedy"):
                eu = 1.0
                pdists = (
                    {str(d["doc_id"]): compute_next_token_distribution(generator, it["question"], [d["text"]], variant_seed=str(d["doc_id"]))["dist"] for d in pool[:8]}
                    if cfg in ("M2_Exact_Enum", "M2_Greedy")
                    else {}
                )
            else:
                eu = float(compute_unified_m1_signal(generator, it["question"], pool[:3], eu_mode="v5_proxy", decoupler=decoupler)["u_eu"])
                conflict_bypass = False if cfg == "M1_Only" else should_bypass_m1_early_exit(pool[: min(5, len(pool))], m2_graph_mode)
                early_exit_full = bool(eu <= tau_eu and not conflict_bypass)
                pdists = (
                    {str(d["doc_id"]): compute_next_token_distribution(generator, it["question"], [d["text"]], variant_seed=str(d["doc_id"]))["dist"] for d in pool[:8]}
                    if (not early_exit_full and cfg != "M1_Only")
                    else {}
                )
            t_m1 = (time.perf_counter() - t1) * 1000.0

            # Stage 3: M2 Selection / Organization
            t2 = time.perf_counter()
            if cfg == "Standard_RAG":
                chosen = [{**d, "calibrated_reliability": float(d["score"])} for d in pool[:3]]
            elif cfg == "MMR_RAG":
                chosen = [{**d, "calibrated_reliability": float(d["score"])} for d in _select_mmr(pool[:8], k_target=min(5, len(pool)))]
            elif cfg == "M1_Only":
                chosen = [{**d, "calibrated_reliability": float(d["score"])} for d in (pool[:3] if eu <= tau_eu else pool[:5])]
            elif cfg == "M2_Greedy":
                chosen, _ = select_m2_by_graph_mode(it["question"], pool, pdists, params, graph_mode=m2_graph_mode, algorithm="greedy", lambda_conflict=m2_lambda_conflict)
            elif cfg == "M2_Exact_Enum":
                chosen, _ = select_m2_by_graph_mode(it["question"], pool, pdists, params, graph_mode=m2_graph_mode, algorithm="exact_enum", lambda_conflict=m2_lambda_conflict)
            else:
                if early_exit_full:
                    chosen = [{**d, "calibrated_reliability": float(min(1.0, float(d["score"]) + rel_off))} for d in pool[:3]]
                else:
                    chosen, _ = select_m2_by_graph_mode(it["question"], pool, pdists, params, graph_mode=m2_graph_mode, algorithm="exact_enum", lambda_conflict=m2_lambda_conflict)
            t_m2 = (time.perf_counter() - t2) * 1000.0

            # Stage 4: Generation & Peak VRAM
            t3 = time.perf_counter()
            reset_cuda_peak()
            if cfg == "Full_UGraphRAG":
                dec_to_use = full_dec_b0 if early_exit_full else full_dec_b12
                res = dec_to_use.generate_calibrated(generator, it["question"], chosen)
            else:
                res = uncal.generate_calibrated(generator, it["question"], chosen)
            t_m3 = (time.perf_counter() - t3) * 1000.0
            t_tot = (time.perf_counter() - t_s) * 1000.0
            vsnap = get_vram_snapshot_mb()

            qa = evaluate_qa_prediction(res.raw_answer, it["possible_answers"], is_abstained=res.is_abstained)
            lat_ret.append(t_ret)
            lat_m1.append(t_m1)
            lat_m2.append(t_m2)
            lat_m3.append(t_m3)
            lat_tot.append(t_tot)
            alloc_peaks.append(vsnap["max_allocated_mb"])
            resv_peaks.append(vsnap["max_reserved_mb"])
            tok_used.append(sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in chosen))
            gen_tok_counts.append(max(1, len(res.raw_answer.split())))
            recs.append({"em": qa["em"], "f1": qa["f1"], "conf": float(res.confidence), "ab": bool(res.is_abstained)})

        total_sec = sum(lat_tot) / 1000.0
        cal = compute_calibration_and_risk_suite([r["conf"] for r in recs], [r["f1"] for r in recs], abstained_flags=[r["ab"] for r in recs])
        out[cfg] = {
            "n": len(recs),
            "em_pct": round(100.0 * sum(r["em"] for r in recs) / max(1, len(recs)), 2),
            "f1_pct": round(100.0 * sum(r["f1"] for r in recs) / max(1, len(recs)), 2),
            "ece": cal["ece"],
            "selective_risk_pct": cal["selective_risk_pct"],
            "latency_ms": {
                "retrieval_mean": round(float(np.mean(lat_ret)), 2),
                "m1_mean": round(float(np.mean(lat_m1)), 2),
                "m2_mean": round(float(np.mean(lat_m2)), 3),
                "m3_mean": round(float(np.mean(lat_m3)), 2),
                "e2e_mean": round(float(np.mean(lat_tot)), 2),
                "e2e_p50": round(float(np.percentile(lat_tot, 50)), 2),
                "e2e_p95": round(float(np.percentile(lat_tot, 95)), 2),
            },
            "throughput": {
                "questions_per_sec": round(len(recs) / max(1e-6, total_sec), 3),
                "output_tokens_per_sec": round(sum(gen_tok_counts) / max(1e-6, total_sec), 2),
            },
            "vram_mb": {
                "mean_allocated_peak": round(float(np.mean(alloc_peaks)), 2),
                "max_allocated_peak": round(float(max(alloc_peaks)), 2),
                "max_reserved_peak": round(float(max(resv_peaks)), 2),
            },
            "mean_prompt_tokens": round(float(np.mean(tok_used)), 1),
        }
    return out


DEFAULT_PHASE3_MODELS = ["qwen3b", "qwen7b", "llama3_8b", "mistral7b", "gemma2"]


def main() -> None:
    parser = argparse.ArgumentParser(description="WP9 & WP10: Stress Curves & Cross-Family Efficiency Pareto.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_PHASE3_MODELS, help="Backbone models to evaluate")
    parser.add_argument("--model-group", type=str, default=None, choices=["core3", "extra2", "all5"], help="3+2 Kaggle anti-disk-overflow model group")
    parser.add_argument("--stress-model", type=str, default=None, help="Defaults to last available requested model (e.g. llama3_8b or qwen3b)")
    parser.add_argument("--scale", type=str, default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--m2-graph-mode", type=str, default="auto", choices=["auto", "entity_js", "entity_cooccur", "overlap_gated_js"])
    parser.add_argument("--m2-lambda-conflict", type=float, default=None, help="Explicit M2 lambda_conflict weight; overrides default/learned setting")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "bf16"])
    parser.add_argument("--allow-hf-online", action="store_true")
    parser.add_argument("--cleanup-after-model", action="store_true", help="Delete cached model weights after teardown to free disk space")
    parser.add_argument("--skip-stress", action="store_true", help="Skip WP9 stress curves (useful when running Group 2 to only collect WP10 efficiency)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output JSONs instead of merging models")
    parser.add_argument("--merge-from", nargs="+", default=[], help="External Phase 3 JSON files to merge efficiency pareto models from")
    parser.add_argument("--wp9-out-dir", type=str, default=str(WP3_DIRS["WP9"]))
    parser.add_argument("--wp10-out-dir", type=str, default=str(WP3_DIRS["WP10"]))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wp9_dir = Path(args.wp9_out_dir)
    wp10_dir = Path(args.wp10_out_dir)
    setup_phase2_logger("phase3.wp9_10", wp9_dir / "wp9_wp10_run.log")
    v5_info = verify_v5_frozen_checksums()
    params = load_v5_balanced_params()
    t0 = time.time()

    # Determine actual model list supporting 3+2 Kaggle anti-disk-overflow
    models_arg = args.models
    if args.model_group is not None and args.models == DEFAULT_PHASE3_MODELS:
        models_arg = None
    actual_models = resolve_phase3_model_list(models_arg, args.model_group)

    # Resolve M2 graph mode
    if args.m2_graph_mode == "auto":
        effective_graph_mode, graph_mode_source = resolve_wp8_recommended_graph_mode()
    else:
        effective_graph_mode = args.m2_graph_mode
        graph_mode_source = "cli_override"
    logger.info(
        f"WP9/WP10 setup: models={actual_models} (group={args.model_group}), "
        f"m2_graph_mode={effective_graph_mode} (source={graph_mode_source}, lambda_conflict={args.m2_lambda_conflict}), "
        f"skip_stress={args.skip_stress}"
    )

    hp_dev, hp_test = load_phase3_dataset_cache("hotpotqa", scale=args.scale, allow_synthetic_fallback=args.dry_run)
    stress_cap = 10 if args.scale == "smoke" else (120 if args.scale == "medium" else 200)
    prof_cap = 8 if args.scale == "smoke" else (40 if args.scale == "medium" else 60)
    stress_items = hp_test[:stress_cap]
    prof_items = hp_test[:prof_cap]
    ab_tau = float(params.get("m3_abstain_threshold", 0.05))

    avail_models = [mk for mk in actual_models if args.dry_run or is_phase3_backbone_available(mk, allow_hf_online=args.allow_hf_online)]
    stress_target = args.stress_model or (avail_models[-1] if avail_models else actual_models[0])

    wp9_results: Dict[str, Any] = {}
    wp10_results: Dict[str, Any] = {}

    for mk in actual_models:
        if mk not in avail_models:
            wp10_results[mk] = {"status": "skipped_unavailable"}
            continue
        reset_cuda_peak()
        generator = sequential_load_phase3_generator(mk, dry_run=args.dry_run, max_new_tokens=48, quantization=args.quantization)
        try:
            dev_ckpt = Phase2CheckpointManager(wp10_dir / f"ckpt_dev_cal_{mk}_{args.scale}.json", flush_every=25)
            dev_ckpt.metadata.update({
                "model": mk,
                "scale": args.scale,
                "quantization": args.quantization,
                "schema_version": "wp9_10_dev_v2",
            })
            model_tau_eu = calibrate_dev_m1_tau_eu(
                hp_dev,
                generator,
                params,
                ckpt=dev_ckpt,
                target_quantile=0.60,
                model_tag=mk,
                dataset_name="hotpotqa",
            )
            cal_ckey = f"wp9_10_dev_cal_v2::{mk}::hotpotqa"
            if dev_ckpt.has(cal_ckey):
                cal_bundle = dev_ckpt.get(cal_ckey)
            else:
                class _CalNs:
                    def has(self, k: str) -> bool:
                        return dev_ckpt.has(f"dev_b0_v2::{k}")
                    def get(self, k: str, d: Any = None) -> Any:
                        return dev_ckpt.get(f"dev_b0_v2::{k}", d)
                    def record(self, k: str, v: Any) -> None:
                        dev_ckpt.record(f"dev_b0_v2::{k}", v)
                    def flush(self, force: bool = True) -> None:
                        dev_ckpt.flush(force=force)

                dev_b0 = _collect_decode_traces(hp_dev, generator, params, beta=0.0, reliability_prompt=False, tag="dev_b0_v2", ckpt=_CalNs())
                full_platt = _fit_dev_calibration_configs(dev_b0)["full_platt"]
                thresh_info = select_dev_risk_constrained_threshold(dev_b0, full_platt, abstain_threshold=ab_tau)
                cal_bundle = {"full_platt": full_platt, "adaptive_tau_conf": float(thresh_info["adaptive_tau_conf"])}
                dev_ckpt.record(cal_ckey, cal_bundle)
            dev_ckpt.metadata["calibrated_tau_eu"] = model_tau_eu
            dev_ckpt.metadata["cal_bundle"] = cal_bundle
            dev_ckpt.flush(force=True)

            logger.info(f"[WP10] Benchmarking efficiency on model={mk} (N={len(prof_items)}, tau_eu={model_tau_eu:.4f})...")
            wp10_results[mk] = {
                "status": "completed",
                "calibrated_tau_eu": model_tau_eu,
                "cal_bundle": cal_bundle,
                "benchmarks": run_wp10_efficiency_for_model(
                    mk,
                    prof_items,
                    generator,
                    params,
                    effective_graph_mode,
                    args.dry_run,
                    m2_lambda_conflict=args.m2_lambda_conflict,
                    tau_eu_override=model_tau_eu,
                    cal_bundle=cal_bundle,
                ),
            }
            if not args.skip_stress and mk == stress_target:
                logger.info(f"[WP9] Running controlled 6-subaxis stress curves on model={mk} (N={len(stress_items)}, tau_eu={model_tau_eu:.4f})...")
                ckpt = Phase2CheckpointManager(wp9_dir / f"ckpt_wp9_{mk}_{args.scale}.json", flush_every=25)
                ckpt.metadata.update({
                    "model": mk,
                    "scale": args.scale,
                    "quantization": args.quantization,
                    "m2_graph_mode": effective_graph_mode,
                    "calibrated_tau_eu": model_tau_eu,
                    "cal_bundle": cal_bundle,
                    "schema_version": "wp9_v2",
                })
                ckpt.flush(force=True)
                wp9_results = {
                    "model": mk,
                    "n_tested": len(stress_items),
                    "calibrated_tau_eu": model_tau_eu,
                    "cal_bundle": cal_bundle,
                    "curves": run_wp9_stress_curves(
                        stress_items,
                        generator,
                        params,
                        effective_graph_mode,
                        ckpt,
                        m2_lambda_conflict=args.m2_lambda_conflict,
                        tau_eu_override=model_tau_eu,
                        cal_bundle=cal_bundle,
                    ),
                }
        finally:
            teardown_generator(generator)
            if args.cleanup_after_model:
                cleanup_cached_model_weights(mk)

    hw = collect_hardware_manifest()

    # Handle WP9 stress curves persistence and preservation
    wp9_file = wp9_dir / "wp9_stress_curves.json"
    final_stress_results = wp9_results
    if (args.skip_stress or not wp9_results) and not args.overwrite and wp9_file.exists():
        try:
            with open(wp9_file, "r", encoding="utf-8") as f:
                old_wp9 = json.load(f)
            if "stress" in old_wp9 and old_wp9["stress"]:
                final_stress_results = old_wp9["stress"]
                logger.info(f"[WP9] Preserved existing stress curves from {wp9_file}")
        except Exception as e:
            logger.warning(f"Failed to read existing WP9 file for preservation: {e}")

    if final_stress_results or not args.skip_stress:
        atomic_write_json(wp9_file, {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scale": args.scale,
                "m2_graph_mode": effective_graph_mode,
                "m2_graph_mode_source": graph_mode_source,
                "m2_lambda_conflict": args.m2_lambda_conflict,
                "stress_model": stress_target,
                "skip_stress": args.skip_stress,
                "dry_run": args.dry_run,
                "elapsed_sec": round(time.time() - t0, 2),
            },
            "hardware_manifest": hw,
            "v5_integrity": v5_info,
            "stress": final_stress_results,
        })

    # Handle WP10 efficiency pareto merging
    wp10_file = wp10_dir / "wp10_efficiency_pareto.json"
    merged_wp10_models: Dict[str, Any] = {}

    if not args.overwrite and wp10_file.exists():
        try:
            with open(wp10_file, "r", encoding="utf-8") as f:
                existing_wp10 = json.load(f)
            for mk_k, mk_v in existing_wp10.get("models", {}).items():
                merged_wp10_models[mk_k] = mk_v
        except Exception as e:
            logger.warning(f"Failed to read existing WP10 file for merging: {e}")

    for mf in args.merge_from:
        mf_path = Path(mf)
        if mf_path.exists():
            try:
                with open(mf_path, "r", encoding="utf-8") as f:
                    ext_wp10 = json.load(f)
                ext_models = ext_wp10.get("models", {})
                for mk_k, mk_v in ext_models.items():
                    merged_wp10_models[mk_k] = mk_v
            except Exception as e:
                logger.warning(f"Failed to merge external WP10 file {mf_path}: {e}")

    merged_wp10_models.update(wp10_results)

    atomic_write_json(wp10_file, {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scale": args.scale,
            "models": list(merged_wp10_models.keys()),
            "active_models": actual_models,
            "model_group": args.model_group,
            "m2_graph_mode": effective_graph_mode,
            "m2_graph_mode_source": graph_mode_source,
            "m2_lambda_conflict": args.m2_lambda_conflict,
            "quantization": args.quantization,
            "dry_run": args.dry_run,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "hardware_manifest": hw,
        "v5_integrity": v5_info,
        "models": merged_wp10_models,
    })
    logger.info(f"Saved WP9 & WP10 results -> {wp9_dir} and {wp10_dir}")


if __name__ == "__main__":
    main()
