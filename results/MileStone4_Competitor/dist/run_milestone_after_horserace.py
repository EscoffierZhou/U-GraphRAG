"""
run_milestone_after_horserace.py
================================
MileStone_After Single-RTX-4090 Top-3 SOTA Horse-Race & Mechanism Fix Suite.

Closes the two remaining gaps identified in the 2024-2026 top-venue survey without
invalidating or re-running Milestone 3 WP6-WP11:
  1. Puzzle Piece 1: M1 Query-Evidence Relevance Lower-Bound Gate blocking blind
     low-entropy Early-Exit on S4 All-Irrelevant pools + Split Conformal Risk Control.
  2. Puzzle Piece 2: Controlled head-to-head comparison on public conflict benchmarks
     (`RAMDocs` 500-item mixed conflict/ambiguity & `NQ-Swap` counterfactual conflict)
     and HotpotQA slices (`Clean`, `Conflict`, `S4 All-Irrelevant`) against zero-training SOTA:
       - B0_StandardRAG
       - BGE_Reranker_Top5  (`BAAI/bge-reranker-v2-m3` 568M Cross-Encoder)
       - BGE_Reranker_MMR   (Cross-Encoder + MMR diversity)
       - CK_PLUG            (ICLR 2026 Confidence Gain logit control)
       - InstructRAG_ICL    (ICLR 2025 training-free rationale denoising)
       - CAD_a05            (NAACL 2024 Context-Aware Decoding alpha=0.5)
       - U_GraphRAG_Plus    (M1 RelevanceGate + M2 Overlap-Gated JS + M3 b0-decoupled Platt/Conformal)

Writes strictly to `results/MileStone4_Competitor/dist/outputs/` (leaving `V5_FROZEN` untouched).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results.MileStone4_Competitor.reuse import load_v5_params_compat, resolve_v5_frozen_dir
from results.MileStone4_Competitor.reuse.datasets.dataset_adapters import (
    build_all_irrelevant_s4_slice,
    load_nqswap_dataset,
    load_ramdocs_dataset,
)
from results.MileStone4_Competitor.reuse.operators.bge_reranker_operator import IsolatedBGEReranker
from results.MileStone4_Competitor.reuse.operators.cad_operator import generate_with_cad
from results.MileStone4_Competitor.reuse.operators.ck_plug_operator import generate_with_ck_plug
from results.MileStone4_Competitor.reuse.operators.conformal_risk_operator import (
    fit_split_conformal_abstention,
)
from results.MileStone4_Competitor.reuse.operators.instruct_rag_icl_operator import generate_with_instructrag_icl
from results.MileStone4_Competitor.reuse.operators.relevance_gate_operator import check_query_evidence_relevance_gate
from scripts.phase2.cache_builder import inject_numeric_or_entity_conflict
from scripts.phase2.wp3_m3_calibration import _collect_decode_traces, _fit_dev_calibration_configs, _score_trace_with_cal_params
from scripts.phase3.phase3_common import (
    Phase2CheckpointManager,
    apply_multiple_comparison_correction,
    atomic_write_json,
    calibrate_dev_m1_tau_eu,
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
    paired_bootstrap_with_tests,
    reset_cuda_peak,
    run_open_decode_cached,
    select_dev_risk_constrained_threshold,
    select_m2_by_graph_mode,
    sequential_load_phase3_generator,
    setup_phase2_logger,
    should_bypass_m1_early_exit,
    teardown_generator,
    verify_v5_frozen_checksums,
)
from src.m1_adaptive_retriever import UncertaintyDecoupler
from src.m3_calibrated_decoder import compute_calibrated_confidence

logger = setup_phase2_logger("phase4.milestone_after")

ALL_HORSERACE_METHODS: Tuple[str, ...] = (
    "B0_StandardRAG",
    "BGE_Reranker_Top5",
    "BGE_Reranker_MMR",
    "CK_PLUG",
    "InstructRAG_ICL",
    "CAD_a05",
    "U_GraphRAG_Plus",
)


def _prepare_benchmarks(scale: str, dry_run: bool, seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    cap_ram = 8 if scale == "smoke" else (200 if scale == "medium" else 500)
    cap_nq = 8 if scale == "smoke" else (200 if scale == "medium" else 500)
    cap_hp = 8 if scale == "smoke" else (120 if scale == "medium" else 200)

    hp_dev, hp_test = load_phase3_dataset_cache("hotpotqa", scale=scale, allow_synthetic_fallback=dry_run)
    hp_clean = list(hp_test[:cap_hp])
    hp_conf: List[Dict[str, Any]] = []
    for it in hp_clean:
        bp = [dict(d) for d in it.get("pool", [])[:16]]
        cd = it.get("synthetic_conflict_doc")
        if cd is None and bp:
            anchor = next((d for d in bp if d.get("is_gold")), bp[0])
            cd = inject_numeric_or_entity_conflict(anchor, seed_key=f"ma_{it['id']}", avoid_answers=it.get("possible_answers", []))
            cd["is_synthetic_conflict"] = True
        if cd and len(bp) >= 2:
            cc = dict(cd)
            cc["score"] = round(float(bp[0].get("score", 0.9)) * 0.98, 4)
            bp = [bp[0], cc] + bp[1:15]
        hp_conf.append({**it, "id": f"hpconf_{it['id']}", "pool": bp, "benchmark": "hotpotqa_conflict"})

    hp_s4 = build_all_irrelevant_s4_slice(hp_clean, max_items=cap_hp)
    ramdocs = load_ramdocs_dataset(max_items=cap_ram, seed=seed)
    nqswap = load_nqswap_dataset(max_items=cap_nq, mode="dual_conflict", seed=seed)

    return list(hp_dev), {
        "ramdocs": ramdocs,
        "nqswap_dual": nqswap,
        "hotpotqa_conflict": hp_conf,
        "hotpotqa_s4_irrelevant": hp_s4,
        "hotpotqa_clean": hp_clean,
    }


def _stage1_cache_bge_reranker_scores(
    benchmarks: Dict[str, List[Dict[str, Any]]],
    ckpt: Phase2CheckpointManager,
    dry_run: bool,
) -> None:
    """Load 568M BGE-Reranker-v2-m3 once, score all candidate pools, flush, and release VRAM."""
    reranker = IsolatedBGEReranker(dry_run=dry_run)
    loaded = False
    try:
        for bname, items in benchmarks.items():
            for it in items:
                pool = list(it.get("pool", []))[:16]
                rkey = f"bge_scores::{bname}::{it['id']}"
                if ckpt.has(rkey) or not pool:
                    continue
                if not loaded:
                    reranker.load()
                    loaded = True
                scores = reranker.score_pool(it["question"], pool)
                ckpt.record(rkey, scores)
        ckpt.flush(force=True)
    finally:
        reranker.teardown()


def run_horserace(
    model_key: str,
    scale: str,
    out_dir: Path,
    quantization: str = "4bit",
    allow_hf_online: bool = False,
    dry_run: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_phase2_logger("phase4.milestone_after", out_dir / "milestone_after.log")
    v5_info = verify_v5_frozen_checksums(v5_dir=resolve_v5_frozen_dir())
    params = load_v5_params_compat()
    ab_tau = float(params.get("m3_abstain_threshold", 0.05))
    rel_off = float(params.get("m2_reliability_offset", 0.22))
    t0 = time.time()

    if not dry_run and not is_phase3_backbone_available(model_key, allow_hf_online=allow_hf_online):
        raise RuntimeError(f"Backbone '{model_key}' not found locally; download first or pass --allow-hf-online.")

    ckpt = Phase2CheckpointManager(out_dir / f"ckpt_milestone_after_{model_key}_{scale}.json", flush_every=20)
    ckpt.metadata.update({
        "model": model_key,
        "scale": scale,
        "quantization": quantization,
        "dry_run": dry_run,
        "schema_version": "milestone_after_v1",
    })

    hp_dev, benchmarks = _prepare_benchmarks(scale=scale, dry_run=dry_run, seed=seed)

    # Stage 1: Score all pools with BGE-Reranker-v2-m3 in isolation before loading LLM
    logger.info(f"[Stage 1] Caching BGE-Reranker-v2-m3 scores across {len(benchmarks)} benchmarks...")
    _stage1_cache_bge_reranker_scores(benchmarks, ckpt, dry_run=dry_run)

    # Stage 2: Load LLM generator cleanly and calibrate anchor M1 + M3 Dev calibration
    reset_cuda_peak()
    generator = sequential_load_phase3_generator(model_key, dry_run=dry_run, max_new_tokens=48, quantization=quantization)
    decoupler = UncertaintyDecoupler()
    reranker_cpu_helper = IsolatedBGEReranker(dry_run=True)

    summary_by_benchmark: Dict[str, Any] = {}
    sig_by_benchmark: Dict[str, Any] = {}
    m1_s4_diagnostic: Dict[str, Any] = {}

    try:
        tau_eu = calibrate_dev_m1_tau_eu(hp_dev, generator, params, ckpt=ckpt, target_quantile=0.60, model_tag=model_key, dataset_name="hotpotqa")
        cal_key = f"after_dev_cal::{model_key}"
        if ckpt.has(cal_key):
            cal_bundle = ckpt.get(cal_key)
        else:
            class _DevNs:
                def has(self, k: str) -> bool: return ckpt.has(f"after_dev::{k}")
                def get(self, k: str, d: Any = None) -> Any: return ckpt.get(f"after_dev::{k}", d)
                def record(self, k: str, v: Any) -> None: ckpt.record(f"after_dev::{k}", v)
                def flush(self, force: bool = True) -> None: ckpt.flush(force=force)

            dev_b0 = _collect_decode_traces(hp_dev, generator, params, beta=0.0, reliability_prompt=False, tag="dev_b0", ckpt=_DevNs())
            platt = _fit_dev_calibration_configs(dev_b0)["full_platt"]
            pol = select_dev_risk_constrained_threshold(dev_b0, platt, abstain_threshold=ab_tau)
            dev_confs = [_score_trace_with_cal_params(tr, platt["T"], platt["w_tok"], platt["platt_a"], platt["platt_b"]) for tr in dev_b0]
            conf_fit = fit_split_conformal_abstention(dev_confs, [float(tr["f1"]) for tr in dev_b0], alpha_harmful=0.15)
            cal_bundle = {
                "full_platt": platt,
                "adaptive_tau_conf": float(pol["adaptive_tau_conf"]),
                "conformal_alpha15_tau": float(conf_fit["tau_conf"]),
            }
            ckpt.record(cal_key, cal_bundle)
            ckpt.flush(force=True)

        platt = cal_bundle["full_platt"]
        adaptive_tau = float(cal_bundle["adaptive_tau_conf"])
        ckpt.metadata["calibrated_tau_eu"] = tau_eu
        ckpt.metadata["cal_bundle"] = cal_bundle
        ckpt.flush(force=True)

        decode_mem: Dict[str, Dict[str, Any]] = {}

        for bname, items in benchmarks.items():
            logger.info(f"[Stage 2] Evaluating {bname} (N={len(items)}, model={model_key})...")
            recs_by_method: Dict[str, List[Dict[str, Any]]] = {m: [] for m in ALL_HORSERACE_METHODS}
            raw_ee_hits, gated_ee_hits, rel_gate_pass_hits = 0, 0, 0

            for it in items:
                iid = str(it["id"])
                pool = list(it.get("pool", []))[:16]
                if not pool:
                    continue
                golds = it["possible_answers"]
                d0_docs = pool[: min(3, len(pool))]

                bge_scores = ckpt.get(f"bge_scores::{bname}::{iid}")
                eu_ckey = f"after_eu::{bname}::{iid}"
                if ckpt.has(eu_ckey):
                    u_eu = float(ckpt.get(eu_ckey)["u_eu"])
                else:
                    u_eu = float(compute_unified_m1_signal(generator, it["question"], d0_docs, eu_mode="v5_proxy", decoupler=decoupler)["u_eu"])
                    ckpt.record(eu_ckey, {"u_eu": u_eu})

                raw_ee = bool(u_eu <= tau_eu)
                rel_gate = check_query_evidence_relevance_gate(it["question"], d0_docs, min_lexical_overlap=0.25, min_retriever_score=0.70)
                conf_bypass = should_bypass_m1_early_exit(pool[: min(5, len(pool))], "overlap_gated_js")
                eff_ee = bool(raw_ee and rel_gate["passes_relevance_gate"] and not conf_bypass)

                if raw_ee:
                    raw_ee_hits += 1
                if eff_ee:
                    gated_ee_hits += 1
                if rel_gate["passes_relevance_gate"]:
                    rel_gate_pass_hits += 1

                # Lazy M2 selection cache for U_GraphRAG_Plus
                m2_ev_cache: Optional[List[Dict[str, Any]]] = None

                def _get_m2_docs() -> List[Dict[str, Any]]:
                    nonlocal m2_ev_cache
                    if m2_ev_cache is not None:
                        return m2_ev_cache
                    pdists: Dict[str, Dict[str, float]] = {}
                    for d in pool:
                        did = str(d["doc_id"])
                        dkey = f"after_dist::{bname}::{iid}::{did}"
                        if ckpt.has(dkey):
                            pdists[did] = ckpt.get(dkey)
                        else:
                            dv = compute_next_token_distribution(generator, it["question"], [d["text"]], variant_seed=did)["dist"]
                            pdists[did] = dv
                            ckpt.record(dkey, dv)
                    m2_ev_cache, _ = select_m2_by_graph_mode(
                        it["question"], pool, pdists, params, graph_mode="overlap_gated_js", pool_k=len(pool), max_k=5
                    )
                    return m2_ev_cache

                for method in ALL_HORSERACE_METHODS:
                    rkey = f"after_res::{bname}::{method}::{iid}"
                    if ckpt.has(rkey):
                        recs_by_method[method].append(ckpt.get(rkey))
                        continue

                    t_s = time.perf_counter()
                    conf_modulated = None
                    if method == "B0_StandardRAG":
                        ev = [{**d, "calibrated_reliability": float(d.get("score", 0.5))} for d in d0_docs]
                        tr = run_open_decode_cached(generator, it["question"], ev, 0.0, params, ckpt=ckpt, cache_prefix=f"after_gen::{bname}", memory_cache=decode_mem, max_new_tokens=48)
                        raw_ans, conf, is_ab = tr["raw_answer"], float(tr["raw_confidence"]), False
                    elif method == "BGE_Reranker_Top5":
                        ev = reranker_cpu_helper.select_reranked(it["question"], pool, top_k=5, use_mmr=False, precomputed_scores=bge_scores)
                        tr = run_open_decode_cached(generator, it["question"], ev, 0.0, params, ckpt=ckpt, cache_prefix=f"after_gen::{bname}", memory_cache=decode_mem, max_new_tokens=48)
                        raw_ans, conf, is_ab = tr["raw_answer"], float(tr["raw_confidence"]), False
                    elif method == "BGE_Reranker_MMR":
                        ev = reranker_cpu_helper.select_reranked(it["question"], pool, top_k=5, use_mmr=True, precomputed_scores=bge_scores)
                        tr = run_open_decode_cached(generator, it["question"], ev, 0.0, params, ckpt=ckpt, cache_prefix=f"after_gen::{bname}", memory_cache=decode_mem, max_new_tokens=48)
                        raw_ans, conf, is_ab = tr["raw_answer"], float(tr["raw_confidence"]), False
                    elif method == "CK_PLUG":
                        ev = [{**d, "calibrated_reliability": float(d.get("score", 0.5))} for d in pool[: min(5, len(pool))]]
                        ckr = generate_with_ck_plug(generator, it["question"], ev, alpha=0.5, adaptive=True, max_new_tokens=48)
                        raw_ans = ckr.raw_answer
                        # Standard uncalibrated modulated probability vs unadjusted
                        conf = float(ckr.raw_confidence_modulated)
                        conf_modulated = float(ckr.raw_confidence_unadjusted)
                        is_ab = False
                    elif method == "InstructRAG_ICL":
                        ev = [{**d, "calibrated_reliability": float(d.get("score", 0.5))} for d in pool[: min(5, len(pool))]]
                        ir = generate_with_instructrag_icl(generator, it["question"], ev, max_new_tokens=96)
                        raw_ans, conf, is_ab = ir.raw_answer, float(ir.raw_confidence), False
                    elif method == "CAD_a05":
                        ev = [{**d, "calibrated_reliability": float(d.get("score", 0.5))} for d in pool[: min(5, len(pool))]]
                        cadr = generate_with_cad(generator, it["question"], ev, alpha=0.5, max_new_tokens=48)
                        raw_ans = cadr.raw_answer
                        conf = float(cadr.raw_confidence_modulated)
                        conf_modulated = float(cadr.raw_confidence_unadjusted)
                        is_ab = False
                    elif method == "U_GraphRAG_Plus":
                        if eff_ee:
                            ev = [{**d, "calibrated_reliability": float(min(1.0, float(d.get("score", 0.5)) + rel_off))} for d in d0_docs]
                            eff_beta = 0.0
                        else:
                            ev = _get_m2_docs()
                            eff_beta = 0.12
                        tr = run_open_decode_cached(generator, it["question"], ev, eff_beta, params, ckpt=ckpt, cache_prefix=f"after_gen::{bname}", memory_cache=decode_mem, max_new_tokens=48, calib_from_b0=True)
                        raw_ans = tr["raw_answer"]
                        conf = float(_score_trace_with_cal_params(tr, platt["T"], platt["w_tok"], platt["platt_a"], platt["platt_b"]))
                        is_ab = (float(tr["max_ev_rel"]) < ab_tau) or (not rel_gate["passes_relevance_gate"]) or (conf < adaptive_tau)
                    else:
                        raise ValueError(method)

                    lat_ms = (time.perf_counter() - t_s) * 1000.0
                    qa = evaluate_qa_prediction(raw_ans, golds, is_abstained=is_ab)
                    rec = {
                        "id": iid,
                        "prediction": raw_ans,
                        "raw_em": qa["em"],
                        "raw_f1": qa["f1"],
                        "effective_em": qa["effective_em"],
                        "effective_f1": qa["effective_f1"],
                        "confidence": round(conf, 4),
                        "alt_unadjusted_conf": round(conf_modulated, 4) if conf_modulated is not None else None,
                        "is_abstained": bool(is_ab),
                        "conflict_retained": int(any(bool(d.get("is_synthetic_conflict")) for d in ev)),
                        "gold_retained": int(any(bool(d.get("is_gold")) for d in ev)),
                        "selected_tokens": sum(int(d.get("token_count", compute_token_count(d.get("text", "")))) for d in ev),
                        "latency_ms": round(lat_ms, 2),
                    }
                    ckpt.record(rkey, rec)
                    recs_by_method[method].append(rec)

            ckpt.flush(force=True)
            if bname == "hotpotqa_s4_irrelevant":
                n_it = max(1, len(items))
                m1_s4_diagnostic = {
                    "n": len(items),
                    "raw_v5_early_exit_rate_pct": round(100.0 * raw_ee_hits / n_it, 2),
                    "relevance_gated_early_exit_rate_pct": round(100.0 * gated_ee_hits / n_it, 2),
                    "blind_ee_blocked_pp": round(100.0 * (raw_ee_hits - gated_ee_hits) / n_it, 2),
                    "relevance_gate_pass_rate_pct": round(100.0 * rel_gate_pass_hits / n_it, 2),
                }

            bsum: Dict[str, Any] = {}
            for m, recs in recs_by_method.items():
                n = max(1, len(recs))
                suite = compute_calibration_and_risk_suite(
                    [r["confidence"] for r in recs], [r["raw_f1"] for r in recs], abstained_flags=[r["is_abstained"] for r in recs]
                )
                bsum[m] = {
                    "n": len(recs),
                    "raw_em_pct": round(100.0 * sum(r["raw_em"] for r in recs) / n, 2),
                    "raw_f1_pct": round(100.0 * sum(r["raw_f1"] for r in recs) / n, 2),
                    "effective_f1_pct": round(100.0 * sum(r["effective_f1"] for r in recs) / n, 2),
                    "conflict_retention_pct": round(100.0 * sum(r["conflict_retained"] for r in recs) / n, 2),
                    "gold_retention_pct": round(100.0 * sum(r["gold_retained"] for r in recs) / n, 2),
                    "ece": suite["ece"],
                    "brier_score": suite["brier_score"],
                    "auroc_confidence": suite["auroc_confidence"],
                    "aurc": suite["aurc"],
                    "selective_accuracy_pct": suite["selective_accuracy_pct"],
                    "abstention_rate_pct": suite["abstention_rate_pct"],
                    "harmful_abstention_rate_pct": suite["harmful_abstention_rate_pct"],
                    "mean_tokens": round(sum(r["selected_tokens"] for r in recs) / n, 1),
                    "mean_latency_ms": round(sum(r["latency_ms"] for r in recs) / n, 1),
                }
            summary_by_benchmark[bname] = bsum

            # Paired bootstrap statistical comparison vs strong competitors
            full_recs = recs_by_method["U_GraphRAG_Plus"]
            raw_tests = {}
            for cmp_m in ("BGE_Reranker_MMR", "CK_PLUG", "InstructRAG_ICL", "CAD_a05"):
                crecs = recs_by_method[cmp_m]
                if full_recs and len(full_recs) == len(crecs):
                    raw_tests[f"U_GraphRAG_Plus_vs_{cmp_m}_F1"] = paired_bootstrap_with_tests(
                        [r["raw_f1"] * 100.0 for r in full_recs], [r["raw_f1"] * 100.0 for r in crecs]
                    )
                    raw_tests[f"U_GraphRAG_Plus_vs_{cmp_m}_Brier"] = paired_bootstrap_with_tests(
                        [(r["confidence"] - r["raw_f1"]) ** 2 for r in full_recs],
                        [(r["confidence"] - r["raw_f1"]) ** 2 for r in crecs],
                    )
            sig_by_benchmark[bname] = apply_multiple_comparison_correction(raw_tests)
    finally:
        teardown_generator(generator)

    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_key,
            "scale": scale,
            "quantization": quantization,
            "dry_run": dry_run,
            "calibrated_tau_eu": tau_eu,
            "cal_bundle": cal_bundle,
            "elapsed_sec": round(time.time() - t0, 2),
            "vram_peak_mb": get_vram_snapshot_mb(),
        },
        "hardware_manifest": collect_hardware_manifest(),
        "v5_integrity": v5_info,
        "m1_s4_relevance_gate_diagnostic": m1_s4_diagnostic,
        "horserace_results": summary_by_benchmark,
        "paired_significance": sig_by_benchmark,
    }
    atomic_write_json(out_dir / "milestone_after_horserace_results.json", payload)

    # Render Markdown & LaTeX summary tables
    md = [
        "# MileStone_After: SOTA Horse-Race & Mechanism Extension Tables",
        "",
        f"> Model: `{model_key}` | Scale: `{scale}` | Quantization: `{quantization}` | M1 tau_eu: `{tau_eu:.4f}` | Conformal tau (alpha=0.15): `{cal_bundle['conformal_alpha15_tau']:.4f}`",
        "",
    ]
    if m1_s4_diagnostic:
        md.extend([
            "## S4 All-Irrelevant M1 Relevance Gate Audit",
            f"- Un-gated `V5 Proxy` Early-Exit Rate on S4: `{m1_s4_diagnostic['raw_v5_early_exit_rate_pct']}%`",
            f"- With `RelevanceLowerBoundGate`: `{m1_s4_diagnostic['relevance_gated_early_exit_rate_pct']}%` (blocked **`{m1_s4_diagnostic['blind_ee_blocked_pp']} pp`** of blind off-topic early exits)",
            "",
        ])
    md.extend([
        "## Cross-Method Horse-Race Comparison",
        "",
        "| Benchmark | Method | Raw EM (%) | Raw F1 (%) | Eff F1 (%) | ConfRet (%) | AUROC | ECE | AURC | SelAcc (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    tex = [
        "% Auto-generated MileStone_After SOTA Horse-Race Table",
        "\\begin{table*}[t]\n\\centering\n\\caption{MileStone\\_After Top-Venue Horse-Race Comparison}\n\\begin{tabular}{llcccccc}",
        "\\toprule\nBenchmark & Method & Raw F1 & ConfRet & AUROC & ECE & AURC & SelAcc \\\\\n\\midrule",
    ]
    for bname, bmap in summary_by_benchmark.items():
        for m in ALL_HORSERACE_METHODS:
            mv = bmap[m]
            md.append(
                f"| {bname} | {m} | {mv['raw_em_pct']:.2f} | {mv['raw_f1_pct']:.2f} | {mv['effective_f1_pct']:.2f} "
                f"| {mv['conflict_retention_pct']:.2f} | {mv['auroc_confidence']:.4f} | {mv['ece']:.4f} | {mv['aurc']:.4f} | {mv['selective_accuracy_pct']:.2f} |"
            )
            tex.append(
                f"{bname.replace('_', '\\_')} & {m.replace('_', '\\_')} & {mv['raw_f1_pct']:.2f} & {mv['conflict_retention_pct']:.2f} "
                f"& {mv['auroc_confidence']:.4f} & {mv['ece']:.4f} & {mv['aurc']:.4f} & {mv['selective_accuracy_pct']:.2f} \\\\"
            )
    tex.append("\\bottomrule\n\\end{tabular}\n\\end{table*}")

    (out_dir / "TABLES_MILESTONE_AFTER.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "tables_milestone_after.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    logger.info(f"Saved MileStone_After Horse-Race outputs -> {out_dir}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="MileStone_After: SOTA Horse-Race & M1/M3 Mechanism Extension.")
    parser.add_argument("--model", type=str, default="llama3_8b", choices=["qwen3b", "qwen7b", "llama3_8b", "mistral7b", "gemma2"])
    parser.add_argument("--scale", type=str, default="medium", choices=["smoke", "medium", "full"])
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-hf-online", action="store_true")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_horserace(
        model_key=args.model,
        scale=args.scale,
        out_dir=Path(args.out_dir),
        quantization=args.quantization,
        allow_hf_online=args.allow_hf_online,
        dry_run=args.dry_run,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
