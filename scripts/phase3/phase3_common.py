"""
phase3_common.py
================
Shared infrastructure for MileStone 3 (WP6-WP11) external validation & AutoDL runs.
Reuses Phase II statistics, V5_FROZEN integrity guards, and M1/M2/M3 primitives
without duplicating core logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.cache_builder import (
    _assign_split,
    _extract_local_context_docs,
    _fetch_hf_parquet_rows,
    _generate_synthetic_multihop,
    _load_cpu_dense_model,
    _normalize_2wiki_row,
    _normalize_musique_row,
    _normalize_ranked_docs,
    build_popqa_neutral_corpus,
    inject_numeric_or_entity_conflict,
    load_or_build_cache as load_phase2_cache,
)
from scripts.phase2.common import (
    BACKBONE_ALIASES,
    DATA_DIR,
    M1_SIGNAL_TAXONOMY,
    MODELS_DIR,
    PHASE2_DATA_DIR,
    RESULTS_DIR,
    Phase2CheckpointManager,
    adjust_p_values_multiple_comparisons,
    apply_multiple_comparison_correction,
    assert_safe_output_path,
    atomic_write_json,
    build_interaction_matrix_for_docs,
    build_prompt,
    compute_auprc,
    compute_calibration_and_risk_suite,
    compute_next_token_distribution,
    compute_paraphrase_disagreement,
    compute_token_count,
    compute_unified_m1_signal,
    dict_docs_to_evidence_candidates,
    evaluate_qa_prediction,
    extract_entities,
    extract_gold_answers,
    extract_gold_titles,
    get_vram_snapshot_mb,
    jaccard_sim,
    load_json_file,
    load_v5_balanced_params,
    paired_bootstrap_with_tests,
    reset_cuda_peak,
    resolve_model_path,
    select_m2_evidence_from_pool,
    sequential_load_generator,
    setup_phase2_logger,
    teardown_generator,
    text_shingles,
    verify_v5_frozen_checksums,
)
from scripts.phase2.p0_3_m3_adaptive_gating import (
    TARGET_COVERAGES,
    compute_exact_risk_at_target_coverages,
    select_dev_risk_constrained_threshold,
)
from scripts.phase2.wp2_m2_ablation import _candidates_to_doc_dicts, _select_mmr
from scripts.phase2.wp3_m3_calibration import _fit_dev_calibration_configs, _score_trace_with_cal_params
from src.baseline_rag import BM25Retriever, DenseRetriever, HybridRetriever
from src.m1_adaptive_retriever import UncertaintyDecoupler
from src.m2_evidence_optimizer import TwoStageM2Optimizer
from src.m3_calibrated_decoder import CalibratedDecoderConfig, M3CalibratedDecoder, compute_calibrated_confidence
from src.u_graphrag_pipeline import compute_supporting_facts_metrics

MILE3_DIR: Path = RESULTS_DIR / "MileStone3"
PHASE3_DATA_DIR: Path = DATA_DIR / "phase3"
PHASE3_CACHE_DIR: Path = PHASE3_DATA_DIR / "caches"

WP3_DIRS: Dict[str, Path] = {
    "WP6": MILE3_DIR / "WP6_CrossFamily",
    "WP7": MILE3_DIR / "WP7_CrossDataset",
    "WP8": MILE3_DIR / "WP8_MethodValidity",
    "WP9": MILE3_DIR / "WP9_Robustness",
    "WP10": MILE3_DIR / "WP10_Efficiency",
    "WP11": MILE3_DIR / "WP11_PaperPackage",
}

PHASE3_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen3b": {
        "display_name": "Qwen2.5-3B-Instruct",
        "local_dirs": ["Qwen2.5-3B-Instruct"],
        "hf_repo": "Qwen/Qwen2.5-3B-Instruct",
        "modelscope_repo": "Qwen/Qwen2.5-3B-Instruct",
        "gated_hf": False,
        "family": "Qwen2.5",
    },
    "qwen7b": {
        "display_name": "Qwen2.5-7B-Instruct",
        "local_dirs": ["Qwen2.5-7B-Instruct"],
        "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
        "modelscope_repo": "Qwen/Qwen2.5-7B-Instruct",
        "gated_hf": False,
        "family": "Qwen2.5",
    },
    "llama3_8b": {
        "display_name": "Llama-3.1-8B-Instruct",
        "local_dirs": ["Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"],
        # Ungated mirror on HF avoids 401/manual-gate blocks and slow overseas ModelScope throttling
        "hf_repo": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "official_hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
        "modelscope_repo": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "gated_hf": False,
        "family": "Llama-3.1",
    },
    "mistral7b": {
        "display_name": "Mistral-7B-Instruct-v0.3",
        "local_dirs": ["Mistral-7B-Instruct-v0.3", "mistral-7b-instruct-v0.3"],
        "hf_repo": "unsloth/mistral-7b-instruct-v0.3",
        "official_hf_repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "modelscope_repo": "LLM-Research/Mistral-7B-Instruct-v0.3",
        "gated_hf": False,
        "family": "Mistral-v0.3",
    },
    "gemma2": {
        "display_name": "Gemma-2-9B-IT",
        "local_dirs": ["gemma-2-9b-it", "Gemma-2-9B-IT"],
        "hf_repo": "unsloth/gemma-2-9b-it",
        "modelscope_repo": "LLM-Research/gemma-2-9b-it",
        "gated_hf": False,
        "family": "Gemma-2",
    },
}

PHASE3_MODEL_GROUPS: Dict[str, List[str]] = {
    "core3": ["qwen3b", "qwen7b", "llama3_8b"],
    "extra2": ["mistral7b", "gemma2"],
    "all5": ["qwen3b", "qwen7b", "llama3_8b", "mistral7b", "gemma2"],
}


def resolve_phase3_model_list(
    models: Optional[Sequence[str]] = None,
    model_group: Optional[str] = None,
) -> List[str]:
    """
    Resolve Phase III model keys, supporting 3+2 Kaggle anti-disk-overflow groups
    ('core3', 'extra2', 'all5') and model alias mapping (e.g. 'gemma2_9b' -> 'gemma2'),
    preserving order and deduplicating.
    """
    selected: List[str] = []
    if model_group:
        grp = str(model_group).strip().lower()
        if grp not in PHASE3_MODEL_GROUPS:
            raise ValueError(f"Unknown model_group '{model_group}', valid groups: {list(PHASE3_MODEL_GROUPS.keys())}")
        selected.extend(PHASE3_MODEL_GROUPS[grp])
    if models:
        for m in models:
            if m:
                selected.append(str(m).strip())
    if not selected:
        selected = list(PHASE3_MODEL_GROUPS["core3"])

    alias_map = {
        "gemma2_9b": "gemma2",
        "gemma-2-9b": "gemma2",
        "gemma-2-9b-it": "gemma2",
    }
    resolved: List[str] = []
    for item in selected:
        cand = alias_map.get(item.lower(), item.lower())
        if cand not in resolved:
            resolved.append(cand)
    return resolved

SCALE_LIMITS: Dict[str, Dict[str, Tuple[int, int]]] = {
    # (dev_limit, test_limit) per scale preset
    "smoke": {"hotpotqa": (12, 20), "popqa": (12, 20), "2wikimultihop": (12, 20), "musique": (12, 20)},
    "medium": {"hotpotqa": (80, 320), "popqa": (80, 120), "2wikimultihop": (80, 220), "musique": (80, 220)},
    "full": {"hotpotqa": (150, 1000), "popqa": (150, 1000), "2wikimultihop": (150, 1000), "musique": (150, 1000)},
}


def _candidate_model_roots() -> List[Path]:
    """Ordered model cache roots checking UGRAPHRAG_MODELS_DIR override first."""
    roots: List[Path] = []
    custom_models_dir = os.getenv("UGRAPHRAG_MODELS_DIR")
    if custom_models_dir:
        roots.append(Path(custom_models_dir))
    roots.extend([MODELS_DIR, Path("/root/autodl-tmp/models"), Path("/tmp/models")])
    return roots


def resolve_phase3_model_path(model_key_or_path: str) -> str:
    """Resolve model key checking all Phase III / AutoDL / Colab / Kaggle candidate folders first."""
    key = str(model_key_or_path).strip()
    low = key.lower()
    if low in PHASE3_MODEL_REGISTRY:
        info = PHASE3_MODEL_REGISTRY[low]
        for root_dir in _candidate_model_roots():
            for folder in info["local_dirs"]:
                cand = root_dir / folder
                if cand.exists() and (any(cand.glob("*.safetensors")) or any(cand.glob("*.bin"))):
                    return str(cand)
        return str(info["hf_repo"])
    return resolve_model_path(model_key_or_path)


def sequential_load_phase3_generator(
    model_key: str,
    dry_run: bool = False,
    max_new_tokens: int = 48,
    quantization: str = "4bit",
) -> Any:
    """Cleanly load any Qwen / Llama-3.1 / Mistral Phase III backbone."""
    resolved = resolve_phase3_model_path(model_key)
    return sequential_load_generator(
        resolved,
        dry_run=dry_run,
        max_new_tokens=max_new_tokens,
        quantization=quantization,
    )


def is_phase3_backbone_available(model_key: str, allow_hf_online: bool = False) -> bool:
    """Verify whether model weights exist locally (or remote loading is permitted)."""
    if allow_hf_online:
        return True
    low = str(model_key).strip().lower()
    if low in PHASE3_MODEL_REGISTRY:
        for root_dir in _candidate_model_roots():
            for folder in PHASE3_MODEL_REGISTRY[low]["local_dirs"]:
                cand = root_dir / folder
                if cand.exists() and (any(cand.glob("*.safetensors")) or any(cand.glob("*.bin"))):
                    return True
        return False
    p = Path(model_key)
    return p.exists() or any((r / model_key).exists() for r in _candidate_model_roots())


def collect_hardware_manifest() -> Dict[str, Any]:
    """Record full GPU/driver/CUDA/PyTorch/Transformers manifest for WP10/WP11 reproducibility."""
    info: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_total_vram_mb": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1)
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    for pkg in ("transformers", "bitsandbytes", "sentence_transformers", "scipy", "datasets"):
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "installed")
        except Exception:
            info[f"{pkg}_version"] = "not_installed"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        info["nvidia_smi"] = out
    except Exception:
        info["nvidia_smi"] = "unavailable"
    return info


def write_jsonl_records(path: Path, records: Sequence[Dict[str, Any]]) -> Path:
    """Atomically write per-sample prediction / diagnostic trace records in JSONL format."""
    assert_safe_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return path


def classify_failure_case(
    raw_f1: float,
    is_abstained: bool,
    pool_gold_recall: float,
    selected_gold_recall: float,
) -> str:
    """
    Deterministic error taxonomy per WP6/WP11:
      distinguishes Retrieval Miss, Selection Drop, Generation Error,
      Valid Abstention Guard, and Harmful Abstention.
    """
    if is_abstained:
        return "harmful_abstention" if raw_f1 >= 0.5 else "valid_abstention_guard"
    if raw_f1 >= 0.5:
        return "correct"
    if pool_gold_recall < 0.5:
        return "retrieval_miss"
    if selected_gold_recall + 1e-6 < pool_gold_recall:
        return "selection_loss"
    return "generation_or_reasoning_error"


def summarize_failure_taxonomy(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize exact counts and percentages across deterministic failure categories."""
    n = max(1, len(records))
    cats = (
        "correct",
        "valid_abstention_guard",
        "harmful_abstention",
        "retrieval_miss",
        "selection_loss",
        "generation_or_reasoning_error",
    )
    counts = {c: sum(1 for r in records if r.get("failure_category") == c) for c in cats}
    pcts = {f"{c}_pct": round(100.0 * counts[c] / n, 2) for c in cats}
    return {"counts": counts, "rates_pct": pcts}


def build_phase3_scaled_cache(
    dataset_name: str,
    target_total: int = 1200,
    pool_k: int = 16,
    download: bool = False,
    allow_synthetic_fallback: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Prepare scaled Phase III dataset pool caches under `data/phase3/caches/`.
    Uses local Phase II raw files if large enough, or fetches full official splits
    from HuggingFace / hf-mirror when `--download` is set on AutoDL.
    Always assigns non-overlapping deterministic `dev` (first dev_n) and `test` splits
    with zero leakage into V5's [0:100] evaluation slice on hotpotqa/popqa.
    """
    PHASE3_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PHASE3_CACHE_DIR / f"{dataset_name}_p3_pool_k{pool_k}_n{target_total}.json"
    if cache_path.exists():
        return load_json_file(cache_path)

    raw_rows: List[Dict[str, Any]] = []
    if allow_synthetic_fallback:
        raw_rows = _generate_synthetic_multihop(dataset_name, min(target_total, 80))
    else:
        if dataset_name == "hotpotqa":
            p_raw = PHASE3_DATA_DIR / f"hotpotqa_val_{target_total}.json"
            p2_raw = PHASE2_DATA_DIR / "hotpotqa_dev_500.json"
            if p_raw.exists():
                raw_rows = load_json_file(p_raw)
            elif download:
                fetched = _fetch_hf_parquet_rows([("hotpotqa/hotpot_qa", "distractor"), ("hotpot_qa", "distractor")], target_total + 100)
                base_rows = load_json_file(p2_raw) if p2_raw.exists() else []
                seen = {str(r.get("id")) for r in base_rows}
                raw_rows = list(base_rows)
                for r in fetched:
                    rid = str(r.get("id"))
                    if rid not in seen:
                        raw_rows.append({
                            "id": rid,
                            "question": r["question"],
                            "ground_truth": r["answer"],
                            "answer": r["answer"],
                            "type": r.get("type", "bridge"),
                            "level": r.get("level", "medium"),
                            "supporting_facts": r.get("supporting_facts", {}),
                            "context": r.get("context", {}),
                        })
                        seen.add(rid)
                atomic_write_json(p_raw, raw_rows)
            elif p2_raw.exists():
                raw_rows = load_json_file(p2_raw)
        elif dataset_name == "popqa":
            p_raw = PHASE3_DATA_DIR / f"popqa_full_{target_total}.json"
            p1_raw = DATA_DIR / "popqa_test_300.json"
            if p_raw.exists():
                raw_rows = load_json_file(p_raw)
            elif download:
                import io
                import pandas as pd
                import requests

                urls = [
                    "https://hf-mirror.com/datasets/akariasai/PopQA/resolve/main/test.tsv",
                    "https://huggingface.co/datasets/akariasai/PopQA/resolve/main/test.tsv",
                ]
                tsv_text = None
                for u in urls:
                    try:
                        resp = requests.get(u, timeout=30)
                        if resp.status_code == 200:
                            tsv_text = resp.text
                            break
                    except Exception:
                        continue
                if tsv_text:
                    df = pd.read_csv(io.StringIO(tsv_text), sep="\t")
                    base_rows = load_json_file(p1_raw) if p1_raw.exists() else []
                    seen = {int(r["id"]) for r in base_rows}
                    # Deterministic stratified sampling of remaining rows matching Step 1 proportions
                    remain = df[~df["id"].isin(seen)].sample(n=min(target_total, len(df)), random_state=seed)
                    raw_rows = list(base_rows)
                    for _, row in remain.iterrows():
                        pa_raw = row["possible_answers"]
                        try:
                            pa = json.loads(pa_raw) if isinstance(pa_raw, str) else list(pa_raw)
                        except Exception:
                            pa = [str(pa_raw)]
                        raw_rows.append({
                            "id": int(row["id"]),
                            "question": str(row["question"]),
                            "subj": str(row.get("subj", "")),
                            "prop": str(row.get("prop", "")),
                            "obj": str(row.get("obj", "")),
                            "s_wiki_title": str(row.get("s_wiki_title", row.get("subj", ""))),
                            "o_wiki_title": str(row.get("o_wiki_title", row.get("obj", ""))),
                            "ground_truth": str(row.get("obj", "")),
                            "possible_answers": pa,
                            "s_pop": int(row["s_pop"]) if pd.notnull(row.get("s_pop")) else 0,
                            "o_pop": int(row["o_pop"]) if pd.notnull(row.get("o_pop")) else 0,
                        })
                    atomic_write_json(p_raw, raw_rows)
            elif p1_raw.exists():
                raw_rows = load_json_file(p1_raw)
        elif dataset_name == "2wikimultihop":
            p_raw = PHASE3_DATA_DIR / f"2wikimultihop_{target_total}.json"
            p2_raw = PHASE2_DATA_DIR / "2wikimultihop_300.json"
            if p_raw.exists():
                raw_rows = load_json_file(p_raw)
            elif download:
                fetched = _fetch_hf_parquet_rows([("framolfese/2WikiMultihopQA", ""), ("SagiPolaczek/2wikimultihopqa", "")], target_total)
                raw_rows = [_normalize_2wiki_row(r, i) for i, r in enumerate(fetched)]
                atomic_write_json(p_raw, raw_rows)
            elif p2_raw.exists():
                raw_rows = load_json_file(p2_raw)
        elif dataset_name == "musique":
            p_raw = PHASE3_DATA_DIR / f"musique_{target_total}.json"
            p2_raw = PHASE2_DATA_DIR / "musique_300.json"
            if p_raw.exists():
                raw_rows = load_json_file(p_raw)
            elif download:
                fetched = _fetch_hf_parquet_rows([("dgslibisey/MuSiQue", ""), ("bdsaglam/musique", "")], target_total)
                raw_rows = [_normalize_musique_row(r, i) for i, r in enumerate(fetched)]
                atomic_write_json(p_raw, raw_rows)
            elif p2_raw.exists():
                raw_rows = load_json_file(p2_raw)

    if not raw_rows:
        return load_phase2_cache(dataset_name, pool_k=pool_k, allow_synthetic_fallback=allow_synthetic_fallback)

    dense_model = _load_cpu_dense_model(dry_run=allow_synthetic_fallback)
    pop_corpus, pop_gold = (None, None)
    global_ret = None
    if dataset_name == "popqa" and not allow_synthetic_fallback:
        hp_rows = load_json_file(PHASE2_DATA_DIR / "hotpotqa_dev_500.json") if (PHASE2_DATA_DIR / "hotpotqa_dev_500.json").exists() else []
        pop_corpus, pop_gold = build_popqa_neutral_corpus(raw_rows[:target_total], hp_rows, mode="natural")
        global_ret = HybridRetriever(pop_corpus, shared_dense_model=dense_model)

    cached_items: List[Dict[str, Any]] = []
    n_selected = min(target_total, len(raw_rows))
    for idx, sample in enumerate(raw_rows[:n_selected]):
        sid = str(sample.get("id", f"{dataset_name}_{idx}"))
        q = str(sample.get("question", ""))
        golds = extract_gold_answers(sample)

        if global_ret is not None:
            gold_titles = (pop_gold or {}).get(sid, set())
            ranked_raw = global_ret.retrieve(q, top_k=pool_k)
            bm25_ids = [str(pop_corpus[doc_idx]["id"]) for doc_idx, _ in global_ret.bm25_retriever.retrieve(q, top_k=5)]
            dense_ids = [str(pop_corpus[doc_idx]["id"]) for doc_idx, _ in global_ret.dense_retriever.retrieve(q, top_k=5)]
        else:
            gold_titles = extract_gold_titles(sample)
            local_docs = _extract_local_context_docs(sample)
            if local_docs:
                lret = HybridRetriever(local_docs, shared_dense_model=dense_model)
                ranked_raw = lret.retrieve(q, top_k=min(pool_k, len(local_docs)))
                bm25_ids = [str(local_docs[doc_idx]["id"]) for doc_idx, _ in lret.bm25_retriever.retrieve(q, top_k=min(5, len(local_docs)))]
                dense_ids = [str(local_docs[doc_idx]["id"]) for doc_idx, _ in lret.dense_retriever.retrieve(q, top_k=min(5, len(local_docs)))]
            else:
                ranked_raw, bm25_ids, dense_ids = [], [], []

        pool = _normalize_ranked_docs(ranked_raw, gold_titles=gold_titles, top_k=pool_k)
        anchor = next((d for d in pool if d["is_gold"]), pool[0] if pool else None)
        conf_doc = inject_numeric_or_entity_conflict(anchor, seed_key=sid, avoid_answers=golds) if anchor else None

        # Assign split safely preserving V5 [0:100] reservation on hotpotqa/popqa
        dev_span = min(120, max(20, n_selected // 4))
        if dataset_name in ("hotpotqa", "popqa"):
            split = "v5_eval" if idx < min(100, n_selected // 3) else ("dev" if idx < min(220, n_selected // 2) else "test")
        else:
            split = "dev" if idx < dev_span else "test"

        cached_items.append({
            "index": idx,
            "id": sid,
            "dataset": dataset_name,
            "split": sample.get("split", split),
            "question": q,
            "ground_truth": golds[0] if golds else "",
            "possible_answers": golds,
            "type": sample.get("type", ""),
            "s_pop": sample.get("s_pop"),
            "prop": sample.get("prop"),
            "gold_titles": sorted(gold_titles),
            "pool": pool,
            "bm25_top5_ids": bm25_ids,
            "dense_top5_ids": dense_ids,
            "synthetic_conflict_doc": conf_doc,
        })

    payload = {
        "metadata": {"dataset": dataset_name, "num_samples": len(cached_items), "pool_k": pool_k, "seed": seed},
        "items": cached_items,
    }
    atomic_write_json(cache_path, payload)
    return payload


def load_phase3_dataset_cache(
    dataset_name: str,
    scale: str = "medium",
    pool_k: int = 16,
    allow_synthetic_fallback: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load `(dev_items, test_items)` for a dataset at `--scale smoke|medium|full`.
    Zero test overlap with `v5_eval` or `dev`.
    """
    dev_lim, test_lim = SCALE_LIMITS[scale][dataset_name]
    if scale in ("smoke", "medium"):
        cache = load_phase2_cache(dataset_name, pool_k=pool_k, allow_synthetic_fallback=allow_synthetic_fallback)
    else:
        cache = build_phase3_scaled_cache(
            dataset_name,
            target_total=max(1200, dev_lim + test_lim + 100),
            pool_k=pool_k,
            download=False,
            allow_synthetic_fallback=allow_synthetic_fallback,
        )

    all_items = list(cache.get("items", []))
    dev_items = [it for it in all_items if it.get("split") == "dev"][:dev_lim]
    test_items = [it for it in all_items if it.get("split") == "test"][:test_lim]
    if not dev_items:
        dev_items = all_items[:dev_lim]
    if not test_items:
        test_items = all_items[dev_lim : dev_lim + test_lim]
    return dev_items, test_items


def select_m2_by_graph_mode(
    question: str,
    pool: Sequence[Dict[str, Any]],
    per_doc_dists: Optional[Dict[str, Dict[str, float]]],
    params: Dict[str, Any],
    graph_mode: str = "entity_js",
    pool_k: int = 8,
    max_k: int = 5,
    algorithm: str = "exact_enum",
    lambda_conflict: Optional[float] = None,
    overlap_shingle_gate: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Unified Phase III M2 selector supporting `entity_js` (V5 baseline),
    `entity_cooccur` (no JS), and `overlap_gated_js`.
    """
    trimmed = list(pool)[:pool_k]
    if not trimmed:
        return [], {"evaluated_subsets": 0, "m2_elapsed_ms": 0.0}

    g1 = float(params.get("m2_gamma_1", 0.8))
    g2 = float(params.get("m2_gamma_2", 0.2))
    lam_div = float(params.get("m2_lambda_div", 0.4))
    rel_off = float(params.get("m2_reliability_offset", 0.22))
    min_k = int(params.get("m2_min_docs", 2))

    raw_map = {str(d.get("doc_id", d.get("id", d.get("title", "")))): d for d in trimmed}
    cands = dict_docs_to_evidence_candidates(trimmed, rel_off, g1, g2, calibrate=True)

    use_js = (graph_mode != "entity_cooccur") and bool(per_doc_dists)
    use_red = graph_mode != "entity_cooccur"
    if lambda_conflict is not None:
        lam_c = float(lambda_conflict)
    elif graph_mode == "entity_cooccur":
        lam_c = 0.0
    elif graph_mode == "overlap_gated_js":
        lam_c = 5.0
    elif graph_mode == "entity_js":
        lam_c = 1.0
    else:
        lam_c = 1.0
    eff_graph = "entity_js" if graph_mode == "overlap_gated_js" else graph_mode
    sh_gate = 0.20 if graph_mode == "overlap_gated_js" else overlap_shingle_gate

    imat, active = build_interaction_matrix_for_docs(
        cands,
        per_doc_dists or {},
        use_minhash_redundancy=use_red,
        use_js_conflict=use_js,
        graph_mode=eff_graph,
        overlap_conflict_shingle_gate=sh_gate,
    )
    opt = TwoStageM2Optimizer(
        lambda_coverage=g1,
        lambda_complementarity=g2,
        lambda_redundancy=lam_div,
        lambda_conflict=lam_c,
        selection_algorithm=algorithm,
        recency_aware=False,
    )
    res = opt.optimize(
        active,
        interaction_matrix=imat,
        max_k=min(max_k, len(active)),
        min_k=min(min_k, len(active)),
        query_entities=extract_entities(question),
        calibrate=False,
    )
    ordered = _candidates_to_doc_dicts(res.ordered_docs, raw_map)
    return ordered, {
        "evaluated_subsets": int(res.m2a_result.evaluated_subsets_count),
        "m2_elapsed_ms": round(float(res.total_elapsed_ms), 3),
        "graph_mode": str(graph_mode),
        "lambda_conflict": float(lam_c),
    }


def has_overlap_conflict_pair(docs: Sequence[Dict[str, Any]], threshold: float = 0.20) -> bool:
    """
    Zero-leakage lexical overlap screen for overlapping near-duplicate / conflicting claims.
    Inspects purely passage text (never oracle labels like `is_synthetic_conflict`) so M1
    early-exit can yield to overlap-gated M2 conflict resolution when candidates share >= threshold
    3-word shingles.
    """
    if len(docs) < 2:
        return False
    shingles = [text_shingles(str(d.get("text", "")), k=3) for d in docs]
    for i in range(len(shingles)):
        if not shingles[i]:
            continue
        for j in range(i + 1, len(shingles)):
            if shingles[j] and jaccard_sim(shingles[i], shingles[j]) >= float(threshold):
                return True
    return False


def should_bypass_m1_early_exit(
    docs: Sequence[Dict[str, Any]],
    m2_graph_mode: str,
    threshold: float = 0.20,
) -> bool:
    """Return True when overlap-gated conflict resolution must override M1 fast-exit."""
    return bool(m2_graph_mode == "overlap_gated_js" and has_overlap_conflict_pair(docs, threshold=threshold))


def resolve_doc_sp_labels(docs: Sequence[Dict[str, Any]], gold_titles: Sequence[str]) -> List[str]:
    """
    Resolve one canonical identifier per document matching `gold_titles` schema:
    uses `doc_id` when `gold_titles` holds PopQA doc IDs (`kg_rel_*`), otherwise `title`.
    """
    gold_set = {str(g).strip() for g in gold_titles if str(g).strip()}
    out: List[str] = []
    for d in docs:
        did = str(d.get("doc_id", d.get("id", ""))).strip()
        title = str(d.get("title", "")).strip()
        if did and did in gold_set and title not in gold_set:
            out.append(did)
        else:
            out.append(title or did)
    return out


def calibrate_dev_m1_tau_eu(
    dev_items: Sequence[Dict[str, Any]],
    generator: Any,
    params: Dict[str, Any],
    ckpt: Optional[Phase2CheckpointManager] = None,
    target_quantile: float = 0.60,
    model_tag: str = "qwen3b",
    dataset_name: str = "hotpotqa",
) -> float:
    """
    Calibrate a single model-level M1 threshold `tau_eu` on the anchor Dev split (default:
    HotpotQA Dev at V5's reference operating quantile ~0.60) using `compute_unified_m1_signal`.
    Model-level anchor calibration eliminates cross-family entropy scale mismatch (e.g., Llama-3.1
    vs Qwen2.5) without per-dataset quantile forcing or test-set leakage, preserving difficulty-aware
    early-exit variation across datasets (PopQA high EE vs MuSiQue high expansion).
    """
    cache_key = f"m1_tau_eu::{model_tag}::{dataset_name}::q{int(round(target_quantile * 100))}"
    if ckpt is not None and ckpt.has(cache_key):
        return float(ckpt.get(cache_key)["tau_eu"])

    decoupler = UncertaintyDecoupler(float(params.get("m1_tau_eu", 0.18)))
    u_vals: List[float] = []

    for it in dev_items:
        q = str(it.get("question", ""))
        pool = list(it.get("pool", []) or [])
        p0 = pool[: min(3, len(pool))]
        if not p0:
            continue
        iid = str(it.get("id", ""))
        d0_sig = "|".join(str(d.get("doc_id", "")) for d in p0)
        eu_ckey = f"wp6_eu_dev::{dataset_name}::{iid}::{d0_sig}"
        if ckpt is not None and ckpt.has(eu_ckey):
            u_val = float(ckpt.get(eu_ckey)["u_eu"])
        else:
            sig = compute_unified_m1_signal(generator, q, p0, eu_mode="v5_proxy", decoupler=decoupler)
            u_val = float(sig["u_eu"])
            if ckpt is not None:
                ckpt.record(eu_ckey, {"u_eu": u_val})
        u_vals.append(u_val)

    if not u_vals:
        tau = float(params.get("m1_tau_eu", 0.18))
    else:
        tau = float(np.quantile(u_vals, target_quantile))
        tau = float(np.clip(tau, 0.05, 0.90))

    if ckpt is not None:
        ckpt.record(
            cache_key,
            {
                "tau_eu": round(tau, 6),
                "sample_size": len(u_vals),
                "target_quantile": float(target_quantile),
                "anchor_dataset": dataset_name,
                "median_u_eu": round(float(np.median(u_vals)), 6) if u_vals else tau,
            },
        )
        ckpt.flush(force=True)

    return round(tau, 6)


def run_open_decode_cached(
    generator: Any,
    question: str,
    evidence_docs: Sequence[Dict[str, Any]],
    eff_beta: float,
    params: Dict[str, Any],
    ckpt: Optional[Phase2CheckpointManager] = None,
    cache_prefix: str = "gen",
    memory_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    max_new_tokens: int = 48,
    calib_from_b0: bool = False,
) -> Dict[str, Any]:
    """
    Deterministic (do_sample=False) prompt-hash deduplicated open-gate decoder.
    Exact zero-approximation reuse across methods/seeds/slices that pass the identical
    ordered passage texts and contrastive weight `eff_beta`, while recomputing
    method-specific `max_ev_rel`/`mean_ev_rel` and `raw_confidence` faithfully.
    """
    if not evidence_docs:
        ab_dec = M3CalibratedDecoder(CalibratedDecoderConfig(abstain_threshold=0.0, confidence_threshold=0.0, max_new_tokens=max_new_tokens))
        ab_res = ab_dec.generate_calibrated(generator, question, [])
        return {
            "raw_answer": ab_res.raw_answer,
            "raw_confidence": 0.0,
            "ctx_probs_by_temp": {},
            "max_ev_rel": 0.0,
            "mean_ev_rel": 0.0,
            "is_faithful": True,
            "has_unanchored": False,
        }

    norm_rels = [float(M3CalibratedDecoder.normalize_evidence_item(d)["reliability"]) for d in evidence_docs]
    max_ev_rel = float(max(norm_rels)) if norm_rels else 0.0
    mean_ev_rel = float(np.mean(norm_rels)) if norm_rels else 0.0
    ordered_norm = [M3CalibratedDecoder.normalize_evidence_item(d) for d in evidence_docs]
    prompt_text = build_prompt(question, ordered_norm, generator=generator)
    p_hash = hashlib.md5(prompt_text.encode("utf-8")).hexdigest()[:16]
    gkey = f"{cache_prefix}::{p_hash}::b{float(eff_beta):.3f}"

    base_rec: Optional[Dict[str, Any]] = None
    if memory_cache is not None and gkey in memory_cache:
        base_rec = memory_cache[gkey]
    elif ckpt is not None and ckpt.has(gkey):
        base_rec = ckpt.get(gkey)
        if memory_cache is not None:
            memory_cache[gkey] = base_rec

    if base_rec is None:
        open_dec = M3CalibratedDecoder(
            CalibratedDecoderConfig(
                abstain_threshold=0.0,
                confidence_threshold=0.0,
                beta=float(eff_beta),
                enable_contrastive=(float(eff_beta) > 0.0),
                preserve_evidence_order=True,
                max_new_tokens=max_new_tokens,
                confidence_temperature=1.0,
                confidence_tok_weight=1.0,
                confidence_platt_a=1.0,
                confidence_platt_b=0.0,
                contrastive_conf_guard=float(params.get("m3_contrastive_conf_guard", 0.90)),
                protect_eos=bool(params.get("m3_protect_eos", True)),
            )
        )
        dec_res = open_dec.generate_calibrated(generator, question, evidence_docs)
        tel = dec_res.metadata.get("telemetry", {})
        base_rec = {
            "raw_answer": dec_res.raw_answer,
            "ctx_probs_by_temp": tel.get("ctx_probs_by_temp", {}),
            "is_faithful": bool(tel.get("is_faithful", True)),
            "has_unanchored": bool(tel.get("has_unanchored", False)),
        }
        if memory_cache is not None:
            memory_cache[gkey] = base_rec
        if ckpt is not None:
            ckpt.record(gkey, base_rec)

    ctx_temps = base_rec.get("ctx_probs_by_temp", {})
    if calib_from_b0 and float(eff_beta) > 0.0:
        b0_res = run_open_decode_cached(
            generator=generator,
            question=question,
            evidence_docs=evidence_docs,
            eff_beta=0.0,
            params=params,
            ckpt=ckpt,
            cache_prefix=cache_prefix,
            memory_cache=memory_cache,
            max_new_tokens=max_new_tokens,
            calib_from_b0=False,
        )
        if b0_res.get("ctx_probs_by_temp"):
            ctx_temps = b0_res["ctx_probs_by_temp"]

    raw_conf = compute_calibrated_confidence(
        ctx_probs_by_temp=ctx_temps,
        max_ev_rel=max_ev_rel,
        mean_ev_rel=mean_ev_rel,
        temperature=1.0,
        w_tok=1.0,
        is_faithful=bool(base_rec.get("is_faithful", True)),
        has_unanchored=bool(base_rec.get("has_unanchored", False)),
        grounding_penalty_factor=0.70,
        fallback_step_probs=None if ctx_temps else [max_ev_rel],
        platt_a=1.0,
        platt_b=0.0,
    )
    return {
        "raw_answer": str(base_rec.get("raw_answer", "")),
        "raw_confidence": round(float(raw_conf), 4),
        "ctx_probs_by_temp": ctx_temps,
        "max_ev_rel": max_ev_rel,
        "mean_ev_rel": mean_ev_rel,
        "is_faithful": bool(base_rec.get("is_faithful", True)),
        "has_unanchored": bool(base_rec.get("has_unanchored", False)),
    }


def compute_multiseed_aggregate(per_seed_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate scalar numeric metrics across multiple random seeds (mean +/- std)."""
    if not per_seed_summaries:
        return {}
    keys = [k for k, v in per_seed_summaries[0].items() if isinstance(v, (int, float))]
    agg: Dict[str, Any] = {"num_seeds": len(per_seed_summaries), "mean": {}, "std": {}}
    for k in keys:
        vals = [float(s[k]) for s in per_seed_summaries if isinstance(s.get(k), (int, float))]
        if vals:
            agg["mean"][k] = round(float(np.mean(vals)), 4)
            agg["std"][k] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
    return agg


def resolve_wp8_recommended_graph_mode(
    default: str = "overlap_gated_js",
    wp8_path: Optional[Union[str, Path]] = None,
) -> Tuple[str, str]:
    """
    Safely inspect WP8 method validity decision to select the M2 graph mode.
    Returns (recommended_mode, decision_source).
    """
    path = Path(wp8_path) if wp8_path else (WP3_DIRS["WP8"] / "wp8_method_validity_results.json")
    if not path.is_file():
        return default, "fallback_default"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rec = (
            data.get("wp8_2_m2_validity", {})
            .get("validity_gate_decision", {})
            .get("recommended_m2_graph_mode")
        )
        if rec and isinstance(rec, str) and rec.strip():
            return rec.strip(), "wp8_decision"
    except Exception as _e_wp8:
        _ = _e_wp8
    return default, "fallback_default"


def cleanup_cached_model_weights(model_key: str) -> Dict[str, Any]:
    """
    Safely purge downloaded weights of a single large backbone to avoid Kaggle/AutoDL disk overflow,
    preserving embeddings (bge-small-en-v1.5), results/, and data/.
    """
    low = str(model_key).strip().lower()
    low = {"gemma2_9b": "gemma2"}.get(low, low)
    if low not in PHASE3_MODEL_REGISTRY:
        return {"model_key": model_key, "deleted_folders": [], "status": "unknown_model"}

    info = PHASE3_MODEL_REGISTRY[low]
    deleted: List[str] = []
    roots = _candidate_model_roots()

    for r in roots:
        for folder in info.get("local_dirs", []):
            target = r / folder
            # Guard against accidental deletion of root or vital subdirs
            if target.is_dir() and "bge-small-en-v1.5" not in target.name and "results" not in target.name and "data" not in target.name:
                try:
                    for sub in sorted(target.rglob("*"), reverse=True):
                        if sub.is_file() or sub.is_symlink():
                            sub.unlink()
                        elif sub.is_dir():
                            sub.rmdir()
                    target.rmdir()
                    deleted.append(str(target))
                except Exception as _e_rm:
                    _ = _e_rm

    return {
        "model_key": low,
        "deleted_folders": deleted,
        "status": "cleaned" if deleted else "no_folders_found",
    }


def merge_model_entries_dict(existing_map: Dict[str, Any], new_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge results across multi-batch Kaggle/AutoDL runs without overwriting prior model keys.
    """
    out = dict(existing_map)
    for k, v in new_map.items():
        if k not in out:
            out[k] = v
        elif isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_model_entries_dict(out[k], v)
        else:
            out[k] = v
    return out
