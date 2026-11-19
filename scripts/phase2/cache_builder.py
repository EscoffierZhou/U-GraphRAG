"""
cache_builder.py (WP0)
======================
Phase II Data Normalization, Multi-Dataset Downloader (HotpotQA / PopQA / 2WikiMultiHopQA /
MuSiQue), Zero-Leakage Split Assignment, CPU-Only Shared Top-K Pool Caching, and
4-Quadrant Controlled Probe Construction.

All downstream WP1-WP5 scripts consume identical frozen retrieval pools from
`data/phase2/caches/` so method differences are never confounded by retrieval variance.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.common import (
    DATA_DIR,
    MODELS_DIR,
    PHASE2_CACHE_DIR,
    PHASE2_DATA_DIR,
    FastDryRunDenseEncoder,
    atomic_write_json,
    compute_token_count,
    extract_entities,
    extract_gold_answers,
    extract_gold_titles,
    load_json_file,
    setup_phase2_logger,
    verify_v5_frozen_checksums,
)
from src.baseline_rag import HybridRetriever

logger = setup_phase2_logger("phase2.wp0_cache")

_CONFLICT_CITIES: Tuple[str, ...] = (
    "Geneva", "Lisbon", "Reykjavik", "Kyoto", "Antwerp", "Verona", "Nairobi", "Adelaide"
)


def _assign_split(dataset_name: str, idx: int) -> str:
    """Zero-leakage split mapping aligned with V5 frozen indices [0:100]."""
    if dataset_name in ("hotpotqa", "popqa"):
        if idx < 100:
            return "v5_eval"
        if idx < 180:
            return "dev"
        return "test"
    if idx < 80:
        return "dev"
    return "test"


def _extract_local_context_docs(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize local paragraph contexts from HotpotQA / 2Wiki / MuSiQue schemas."""
    if isinstance(sample.get("paragraphs"), list):
        docs: List[Dict[str, Any]] = []
        for idx, p in enumerate(sample["paragraphs"]):
            if not isinstance(p, dict):
                continue
            title = str(p.get("title", f"para_{idx}"))
            text = str(p.get("paragraph_text", p.get("text", "")))
            docs.append({"id": title, "title": title, "text": text, "idx": idx})
        return docs

    ctx = sample.get("context")
    docs = []
    if isinstance(ctx, dict):
        titles = ctx.get("title", []) or []
        sents_list = ctx.get("sentences", []) or []
        for idx, (t, s) in enumerate(zip(titles, sents_list)):
            title = str(t)
            text = " ".join(str(x) for x in s) if isinstance(s, list) else str(s)
            docs.append({"id": title or f"doc_{idx}", "title": title or f"doc_{idx}", "text": text})
    elif isinstance(ctx, list):
        for idx, item in enumerate(ctx):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                sents = item[1]
                text = " ".join(str(x) for x in sents) if isinstance(sents, list) else str(sents)
                docs.append({"id": title or f"doc_{idx}", "title": title or f"doc_{idx}", "text": text})
            elif isinstance(item, dict):
                title = str(item.get("title", item.get("id", f"doc_{idx}")))
                text = str(item.get("text", item.get("paragraph_text", item.get("content", ""))))
                docs.append({"id": title, "title": title, "text": text})
    return docs


def inject_numeric_or_entity_conflict(
    doc: Dict[str, Any],
    seed_key: str,
    avoid_answers: Sequence[str] = (),
) -> Dict[str, Any]:
    """
    Create a deterministic minimal-edit conflicting version of a document by altering
    either a 4-digit year, an integer/decimal number, or an object phrase.
    """
    text = str(doc.get("text", ""))
    title = str(doc.get("title", doc.get("doc_id", "doc")))
    h = int(hashlib.md5(f"{doc.get('doc_id', title)}|{seed_key}".encode("utf-8")).hexdigest()[:8], 16)
    new_text = text

    year_matches = list(re.finditer(r"\b(1[6-9][0-9]{2}|20[0-2][0-9])\b", text))
    if year_matches:
        m = year_matches[h % len(year_matches)]
        orig_y = int(m.group(1))
        shift = 17 if (orig_y + 17) <= 2025 else -19
        new_y = str(orig_y + shift)
        new_text = text[: m.start()] + new_y + text[m.end() :]
    else:
        num_matches = list(re.finditer(r"\b([1-9][0-9]{0,4})\b", text))
        if num_matches:
            m = num_matches[h % len(num_matches)]
            orig_n = int(m.group(1))
            new_n = str(orig_n + (13 if orig_n < 9000 else -11))
            new_text = text[: m.start()] + new_n + text[m.end() :]
        else:
            avoid_low = {str(a).lower() for a in avoid_answers}
            alt = next(
                (c for i, c in enumerate(_CONFLICT_CITIES) if c.lower() not in avoid_low and (i >= h % len(_CONFLICT_CITIES))),
                "Verona",
            )
            if " in " in text:
                parts = text.rsplit(" in ", 1)
                tail_words = parts[1].split(".", 1)
                rest = f".{tail_words[1]}" if len(tail_words) > 1 else "."
                new_text = f"{parts[0]} in {alt}{rest}"
            else:
                new_text = f"{text.rstrip()} However, independent archival records state it was {alt} instead."

    return {
        "doc_id": f"{doc.get('doc_id', doc.get('id', title))}__conflict",
        "title": f"{title} (Conflicting Source)",
        "text": new_text,
        "score": round(max(0.25, float(doc.get("score", 0.7)) * 0.94), 4),
        "fusion_score": float(doc.get("fusion_score", 0.02)),
        "entities": sorted(extract_entities(new_text, title=title)),
        "token_count": compute_token_count(new_text),
        "is_gold": False,
        "is_synthetic_conflict": True,
        "conflict_source_of": str(doc.get("doc_id", doc.get("id", title))),
    }


# -----------------------------------------------------------------------------
# Dataset Loaders & Remote Fetchers (with offline dry-run synthetic fixtures)
# -----------------------------------------------------------------------------

def _normalize_2wiki_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    ans = str(row.get("answer", row.get("ground_truth", "")))
    titles = []
    sf = row.get("supporting_facts", {})
    if isinstance(sf, dict):
        titles = [str(t) for t in (sf.get("title") or []) if str(t).strip()]
    elif isinstance(sf, list):
        titles = [str(item[0]) for item in sf if isinstance(item, (list, tuple)) and item]
    return {
        "id": str(row.get("_id", row.get("id", f"2wiki_{idx}"))),
        "dataset": "2wikimultihop",
        "split": _assign_split("2wikimultihop", idx),
        "question": str(row.get("question", "")),
        "ground_truth": ans,
        "possible_answers": [ans] if ans else [],
        "type": str(row.get("type", "bridge_comparison")),
        "supporting_facts": sf,
        "gold_titles": sorted(set(titles)),
        "context": row.get("context", {}),
    }


def _normalize_musique_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    ans = str(row.get("answer", row.get("ground_truth", "")))
    aliases = row.get("answer_aliases", []) or []
    if isinstance(aliases, str):
        aliases = [aliases]
    possible = []
    for cand in [ans] + list(aliases):
        s = str(cand).strip()
        if s and s not in possible:
            possible.append(s)

    paras = row.get("paragraphs", []) or []
    gold_titles: List[str] = []
    ctx_titles: List[str] = []
    ctx_sents: List[List[str]] = []
    for p_idx, p in enumerate(paras):
        if not isinstance(p, dict):
            continue
        t = str(p.get("title", f"para_{p_idx}"))
        txt = str(p.get("paragraph_text", p.get("text", "")))
        ctx_titles.append(t)
        ctx_sents.append([txt])
        if p.get("is_supporting"):
            gold_titles.append(t)

    return {
        "id": str(row.get("id", f"musique_{idx}")),
        "dataset": "musique",
        "split": _assign_split("musique", idx),
        "question": str(row.get("question", "")),
        "ground_truth": ans,
        "possible_answers": possible,
        "type": "multihop_musique",
        "supporting_facts": {"title": gold_titles, "sent_id": [0] * len(gold_titles)},
        "gold_titles": sorted(set(gold_titles)),
        "context": {"title": ctx_titles, "sentences": ctx_sents},
    }


def _fetch_hf_parquet_rows(repo_candidates: Sequence[Tuple[str, str]], limit: int) -> List[Dict[str, Any]]:
    """
    Download dataset rows via `datasets` library first, then HF parquet API / hf-mirror fallback.
    """
    # 1. Try HuggingFace `datasets` library (respecting mirror if already configured)
    for repo_id, config in repo_candidates:
        try:
            from datasets import load_dataset

            logger.info(f"Trying datasets.load_dataset('{repo_id}', '{config}')...")
            ds = load_dataset(repo_id, config, split="validation") if config else load_dataset(repo_id, split="validation")
            rows: List[Dict[str, Any]] = []
            for i in range(min(limit, len(ds))):
                rows.append(dict(ds[i]))
            if rows:
                logger.info(f"Fetched {len(rows)} rows via `datasets` from {repo_id}.")
                return rows
        except Exception as err:
            logger.warning(f"load_dataset failed for {repo_id}: {err}")

    # 2. Direct HF datasets-server / parquet fallback
    try:
        import pandas as pd
        import requests

        for repo_id, _ in repo_candidates:
            for base in ("https://huggingface.co", "https://hf-mirror.com"):
                try:
                    api_url = f"{base}/api/datasets/{repo_id}/parquet"
                    resp = requests.get(api_url, timeout=25)
                    if resp.status_code != 200:
                        continue
                    pdata = resp.json()
                    val_urls = []
                    for _, splits in pdata.items():
                        if isinstance(splits, dict):
                            val_urls.extend(splits.get("validation") or splits.get("dev") or [])
                    if not val_urls:
                        continue
                    rows = []
                    for u in val_urls:
                        pr = requests.get(u, timeout=45)
                        if pr.status_code == 200:
                            df = pd.read_parquet(io.BytesIO(pr.content))
                            for _, rec in df.iloc[: limit - len(rows)].iterrows():
                                rows.append(rec.to_dict())
                            if len(rows) >= limit:
                                return rows
                    if rows:
                        return rows
                except Exception as inner_err:
                    logger.debug(f"Parquet fetch error for {repo_id} @ {base}: {inner_err}")
    except ImportError:
        pass
    raise RuntimeError(f"Could not download dataset from candidates: {repo_candidates}")


def _generate_synthetic_multihop(dataset_name: str, count: int) -> List[Dict[str, Any]]:
    """Deterministic multi-hop synthetic corpus strictly for CPU `--dry-run` smoke testing."""
    rng = random.Random(2026 if dataset_name == "2wikimultihop" else 2027)
    records: List[Dict[str, Any]] = []
    cities = ["Prague", "Zurich", "Kyoto", "Bologna", "Valencia", "Ghent", "Oslo", "Lyon"]
    works = ["Chronicle", "Nocturne", "Astrolabe", "Meridian", "Harbor", "Fresco", "Sonata", "Treatise"]

    for i in range(count):
        p_name = f"Author_{dataset_name}_{i}"
        w_name = f"{works[i % len(works)]}_{i}"
        city = cities[(i * 3 + rng.randint(0, 2)) % len(cities)]
        yr = 1890 + (i % 95)
        gold_1_t = f"{w_name} Entry"
        gold_2_t = f"{p_name} Biography"
        gold_1_txt = f"{w_name} is a landmark work created by {p_name} in {yr}."
        gold_2_txt = f"{p_name} was educated and resided in {city} during the {yr // 10 * 10}s."

        ctx_titles = [gold_1_t, gold_2_t]
        ctx_sents = [[gold_1_txt], [gold_2_txt]]
        for d in range(6):
            d_t = f"Distractor_{dataset_name}_{i}_{d}"
            d_city = cities[(i + d + 1) % len(cities)]
            d_txt = f"{w_name} catalog index {d} discusses regional archives in {d_city} around {yr + d}."
            ctx_titles.append(d_t)
            ctx_sents.append([d_txt])

        records.append({
            "id": f"syn_{dataset_name}_{i}",
            "dataset": dataset_name,
            "split": _assign_split(dataset_name, i),
            "question": f"In which city did the creator of {w_name} reside?",
            "ground_truth": city,
            "possible_answers": [city],
            "type": "bridge",
            "supporting_facts": {"title": [gold_1_t, gold_2_t], "sent_id": [0, 0]},
            "gold_titles": [gold_1_t, gold_2_t],
            "context": {"title": ctx_titles, "sentences": ctx_sents},
            "is_synthetic_dryrun": True,
        })
    return records


def ensure_dataset_files(
    datasets_to_prepare: Sequence[str],
    limit_per_dataset: int = 300,
    hotpot_limit: int = 300,
    download: bool = False,
    allow_synthetic_fallback: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load local datasets or download / prepare normalized datasets under data/phase2/."""
    PHASE2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    loaded: Dict[str, List[Dict[str, Any]]] = {}

    for name in datasets_to_prepare:
        if name == "hotpotqa":
            p2_path = PHASE2_DATA_DIR / f"hotpotqa_dev_{hotpot_limit}.json"
            base_path = DATA_DIR / "hotpotqa_dev_300.json"
            if p2_path.exists():
                rows = load_json_file(p2_path)
            elif base_path.exists() and hotpot_limit <= 300:
                rows = load_json_file(base_path)[:hotpot_limit]
            elif download and hotpot_limit > 300:
                base_rows = load_json_file(base_path) if base_path.exists() else []
                raw = _fetch_hf_parquet_rows([("hotpotqa/hotpot_qa", "distractor"), ("hotpot_qa", "distractor")], hotpot_limit)
                rows = list(base_rows)
                seen = {str(r.get("id")) for r in rows}
                for r in raw:
                    rid = str(r.get("id"))
                    if rid not in seen and len(rows) < hotpot_limit:
                        rows.append({
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
                atomic_write_json(p2_path, rows)
            elif base_path.exists():
                rows = load_json_file(base_path)
            else:
                raise FileNotFoundError(f"HotpotQA not found at {base_path}")

            norm_rows = []
            for idx, r in enumerate(rows[:hotpot_limit]):
                r_copy = dict(r)
                r_copy["dataset"] = "hotpotqa"
                r_copy["split"] = _assign_split("hotpotqa", idx)
                r_copy["gold_titles"] = sorted(extract_gold_titles(r_copy))
                norm_rows.append(r_copy)
            loaded["hotpotqa"] = norm_rows

        elif name == "popqa":
            popqa_path = DATA_DIR / "popqa_test_300.json"
            if not popqa_path.exists():
                raise FileNotFoundError(f"PopQA not found at {popqa_path}")
            rows = load_json_file(popqa_path)[:limit_per_dataset]
            norm_rows = []
            for idx, r in enumerate(rows):
                r_copy = dict(r)
                r_copy["dataset"] = "popqa"
                r_copy["split"] = _assign_split("popqa", idx)
                norm_rows.append(r_copy)
            loaded["popqa"] = norm_rows

        elif name == "2wikimultihop":
            out_path = PHASE2_DATA_DIR / f"2wikimultihop_{limit_per_dataset}.json"
            if out_path.exists():
                rows = load_json_file(out_path)[:limit_per_dataset]
            elif download:
                raw = _fetch_hf_parquet_rows(
                    [("framolfese/2WikiMultihopQA", ""), ("SagiPolaczek/2wikimultihopqa", "")],
                    limit=limit_per_dataset,
                )
                rows = [_normalize_2wiki_row(r, i) for i, r in enumerate(raw[:limit_per_dataset])]
                atomic_write_json(out_path, rows)
            elif allow_synthetic_fallback:
                rows = _generate_synthetic_multihop("2wikimultihop", min(limit_per_dataset, 300))
            else:
                raise FileNotFoundError(
                    f"{out_path} missing. Run with --download on internet-connected host to fetch 2WikiMultiHopQA."
                )
            loaded["2wikimultihop"] = rows

        elif name == "musique":
            out_path = PHASE2_DATA_DIR / f"musique_{limit_per_dataset}.json"
            if out_path.exists():
                rows = load_json_file(out_path)[:limit_per_dataset]
            elif download:
                raw = _fetch_hf_parquet_rows(
                    [("dgslibisey/MuSiQue", ""), ("bdsaglam/musique", "")],
                    limit=limit_per_dataset,
                )
                rows = [_normalize_musique_row(r, i) for i, r in enumerate(raw[:limit_per_dataset])]
                atomic_write_json(out_path, rows)
            elif allow_synthetic_fallback:
                rows = _generate_synthetic_multihop("musique", min(limit_per_dataset, 300))
            else:
                raise FileNotFoundError(
                    f"{out_path} missing. Run with --download on internet-connected host to fetch MuSiQue."
                )
            loaded["musique"] = rows

        else:
            raise ValueError(f"Unsupported dataset name: {name}")

    return loaded


# -----------------------------------------------------------------------------
# PopQA Corpus Builders & Shared Top-K Pool Builder
# -----------------------------------------------------------------------------

_POPQA_PROP_NATURAL: Dict[str, str] = {
    "author": "{subj} is a written work authored by {obj}.",
    "capital": "The administrative capital of {subj} is {obj}.",
    "capital of": "{subj} serves as the capital city of {obj}.",
    "color": "The documented primary color of {subj} is {obj}.",
    "composer": "The musical score for {subj} was composed by {obj}.",
    "country": "{subj} is located within the country of {obj}.",
    "director": "The motion picture {subj} was directed by {obj}.",
    "father": "The biological father of {subj} is {obj}.",
    "genre": "{subj} belongs primarily to the {obj} genre.",
    "mother": "The mother of {subj} is {obj}.",
    "occupation": "The primary recorded occupation of {subj} is {obj}.",
    "place of birth": "{subj} was born in {obj}.",
    "producer": "{subj} was produced under the direction of {obj}.",
    "religion": "Historical sources associate {subj} with {obj}.",
    "screenwriter": "The screenplay for {subj} was written by {obj}.",
    "sport": "{subj} competes professionally in {obj}.",
}


def build_popqa_neutral_corpus(
    popqa_samples: Sequence[Dict[str, Any]],
    hotpot_samples: Sequence[Dict[str, Any]],
    mode: str = "natural",
) -> Tuple[List[Dict[str, Any]], Dict[str, Set[str]]]:
    """
    Build shared PopQA corpus without target leakage.
    mode='natural': uses diverse encyclopedic wording WITHOUT artificial copy-pasted anchor strings.
    mode='v5_compat': reproduces V5's shared bridge corpus for exact replication checks.
    """
    corpus: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    gold_id_map: Dict[str, Set[str]] = {}

    for item in popqa_samples:
        iid = str(item.get("id", len(corpus)))
        subj = str(item.get("subj") or item.get("s_wiki_title") or item["question"])
        s_wiki = str(item.get("s_wiki_title") or subj)
        prop = str(item.get("prop") or "attribute")
        obj = str(item.get("ground_truth") or "")

        gid = f"kg_rel_{iid}"
        gold_id_map[iid] = {gid}
        if gid not in seen_ids:
            rel_sent = _POPQA_PROP_NATURAL.get(
                prop, "Archival records list the {prop} of {subj} as {obj}."
            ).format(subj=subj, prop=prop, obj=obj)
            if mode == "v5_compat":
                anchor = f"{s_wiki} ({subj}) {s_wiki} factual dossier {subj} {s_wiki} canonical hub"
                text = f"{anchor}. {rel_sent}"
            else:
                text = f"Encyclopedia profile for {s_wiki} ({subj}): {rel_sent}"
            corpus.append({
                "id": gid,
                "title": f"{s_wiki} - Fact Sheet",
                "text": text,
                "entities": [s_wiki, subj, obj],
            })
            seen_ids.add(gid)

        # Overview background passage (naturally phrases the topic without artificial anchor hacks)
        bg_id = f"kg_bg_{iid}"
        if bg_id not in seen_ids:
            bg_text = (
                f"{s_wiki} ({subj}) {s_wiki} factual dossier {subj} {s_wiki} canonical hub. Overview examining {prop} of {subj}."
                if mode == "v5_compat"
                else f"{s_wiki} is an entity frequently referenced in cultural and biographical records regarding {prop} topics."
            )
            corpus.append({
                "id": bg_id,
                "title": f"{s_wiki} Overview",
                "text": bg_text,
                "entities": [s_wiki, subj],
            })
            seen_ids.add(bg_id)

        # Lexical distractor competing on {subj} + {prop} without stating the answer
        for tag, desc in (
            ("chronicle", f"Historical survey mentioning {subj} and general {prop} background in public archives."),
            ("directory", f"Reference index entry listing {s_wiki} cross-references related to {prop} classifications."),
        ):
            did = f"kg_{tag}_{iid}"
            if did not in seen_ids:
                corpus.append({
                    "id": did,
                    "title": f"{subj} {tag}",
                    "text": desc,
                    "entities": [s_wiki, subj],
                })
                seen_ids.add(did)

    # Mix real HotpotQA Wikipedia distractor paragraphs (~600 chunks)
    added_hp = 0
    for hs in hotpot_samples[:120]:
        for d in _extract_local_context_docs(hs):
            cid = f"hp_bg_{abs(hash(str(d['title']))) % 10000000}"
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            para = str(d.get("text", ""))[:450]
            if para.strip():
                corpus.append({
                    "id": cid,
                    "title": str(d.get("title", cid)),
                    "text": para,
                    "entities": [str(d.get("title", cid))],
                })
                added_hp += 1
                if added_hp >= 600:
                    return corpus, gold_id_map
    return corpus, gold_id_map


def _load_cpu_dense_model(dry_run: bool = False):
    """Load local BGE dense encoder strictly on CPU (or offline FastDryRunDenseEncoder in dry_run)."""
    if dry_run:
        return FastDryRunDenseEncoder()
    bge_local = MODELS_DIR / "bge-small-en-v1.5"
    model_path = str(bge_local) if bge_local.exists() else "BAAI/bge-small-en-v1.5"
    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading dense encoder on CPU: {model_path}")
        return SentenceTransformer(model_path, device="cpu")
    except Exception as err:
        logger.warning(f"SentenceTransformer unavailable ({err}); falling back to FastDryRunDenseEncoder.")
        return FastDryRunDenseEncoder()


def _normalize_ranked_docs(
    ranked_raw: Sequence[Dict[str, Any]],
    gold_titles: Set[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Convert retriever outputs into clean serialized candidate dicts."""
    if not ranked_raw:
        return []
    fscores = [float(d.get("fusion_score", d.get("score", 0.0))) for d in ranked_raw]
    s_min, s_max = min(fscores), max(fscores)
    span = (s_max - s_min) if (s_max - s_min) > 1e-9 else 1.0

    out: List[Dict[str, Any]] = []
    for idx, d in enumerate(ranked_raw[:top_k]):
        doc_id = str(d.get("id", d.get("doc_id", d.get("title", f"doc_{idx}"))))
        title = str(d.get("title", doc_id))
        text = str(d.get("text", ""))
        raw_fs = float(d.get("fusion_score", d.get("score", 0.0)))
        norm_s = (raw_fs - s_min) / span if len(ranked_raw) > 1 and s_max > s_min else max(0.20, 0.95 - 0.06 * idx)
        score = round(0.20 + 0.78 * norm_s, 4)
        ents = sorted(extract_entities(text, title=title) | {str(e).lower() for e in d.get("entities", []) if str(e).strip()})
        out.append({
            "doc_id": doc_id,
            "id": doc_id,
            "title": title,
            "text": text,
            "score": score,
            "fusion_score": round(raw_fs, 6),
            "entities": ents,
            "token_count": compute_token_count(text),
            "is_gold": (title in gold_titles) or (doc_id in gold_titles),
        })
    return out


def build_shared_pool_cache(
    dataset_name: str,
    samples: Sequence[Dict[str, Any]],
    pool_k: int = 16,
    shared_dense_model: Any = None,
    popqa_corpus: Optional[List[Dict[str, Any]]] = None,
    popqa_gold_map: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Any]:
    """Pre-compute identical top-K candidate pools and deterministic conflict variants for a dataset."""
    PHASE2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    global_ret: Optional[HybridRetriever] = None
    if popqa_corpus is not None:
        global_ret = HybridRetriever(popqa_corpus, shared_dense_model=shared_dense_model)

    cached_items: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        sid = str(sample.get("id", f"{dataset_name}_{idx}"))
        question = str(sample.get("question", ""))
        golds = extract_gold_answers(sample)

        if global_ret is not None:
            gold_titles = (popqa_gold_map or {}).get(sid, set())
            ranked_raw = global_ret.retrieve(question, top_k=pool_k)
        else:
            gold_titles = extract_gold_titles(sample)
            local_docs = _extract_local_context_docs(sample)
            if not local_docs:
                ranked_raw = []
            else:
                local_ret = HybridRetriever(local_docs, shared_dense_model=shared_dense_model)
                ranked_raw = local_ret.retrieve(question, top_k=min(pool_k, len(local_docs)))

        pool = _normalize_ranked_docs(ranked_raw, gold_titles=gold_titles, top_k=pool_k)
        anchor_doc = next((d for d in pool if d["is_gold"]), pool[0] if pool else None)
        conflict_variant = (
            inject_numeric_or_entity_conflict(anchor_doc, seed_key=sid, avoid_answers=golds)
            if anchor_doc is not None
            else None
        )

        cached_items.append({
            "index": idx,
            "id": sid,
            "dataset": dataset_name,
            "split": sample.get("split", _assign_split(dataset_name, idx)),
            "question": question,
            "ground_truth": golds[0] if golds else "",
            "possible_answers": golds,
            "type": sample.get("type", ""),
            "s_pop": sample.get("s_pop"),
            "prop": sample.get("prop"),
            "gold_titles": sorted(gold_titles),
            "pool": pool,
            "synthetic_conflict_doc": conflict_variant,
        })

    out_path = PHASE2_CACHE_DIR / f"{dataset_name}_pool_k{pool_k}.json"
    payload = {
        "metadata": {
            "dataset": dataset_name,
            "num_samples": len(cached_items),
            "pool_k": pool_k,
        },
        "items": cached_items,
    }
    atomic_write_json(out_path, payload)
    logger.info(f"Saved shared pool cache: {out_path} ({len(cached_items)} samples).")
    return payload


def build_four_quadrant_probe(
    hotpot_cache: Dict[str, Any],
    max_per_quadrant: int = 40,
) -> Dict[str, Any]:
    """
    Construct controlled 4-quadrant diagnostic probe from HotpotQA cache items:
      - Q1_known_consistent: well-formed query + all gold docs (no conflict)
      - Q2_unknown_sufficient: multi-hop bridge query + all gold docs (tests RAG resolution)
      - Q3_known_conflicting: gold doc + injected factual conflict doc
      - Q4_unknown_missing: gold docs stripped (distractors only -> tests insufficient evidence)
    """
    items = hotpot_cache.get("items", [])
    probe_items: List[Dict[str, Any]] = []
    # Prefer dev + unseen test items to avoid leaking any tuning decisions
    source_items = [it for it in items if it.get("split") in ("dev", "test")] or items

    for idx, it in enumerate(source_items[:max_per_quadrant]):
        pool = list(it.get("pool", []))
        gold_docs = [d for d in pool if d.get("is_gold")]
        distractor_docs = [d for d in pool if not d.get("is_gold")]
        if not gold_docs:
            continue
        base_docs = gold_docs[:2] + distractor_docs[:1]
        conflict_doc = it.get("synthetic_conflict_doc") or inject_numeric_or_entity_conflict(
            gold_docs[0], seed_key=f"q3_{it['id']}", avoid_answers=it.get("possible_answers", [])
        )

        quadrants = [
            ("Q1_known_consistent", gold_docs[:2] + distractor_docs[:1], False, True),
            ("Q2_unknown_sufficient", base_docs + distractor_docs[1:3], False, True),
            ("Q3_known_conflicting", [gold_docs[0], conflict_doc] + distractor_docs[:1], True, False),
            ("Q4_unknown_missing", distractor_docs[:3] if len(distractor_docs) >= 3 else distractor_docs, False, False),
        ]
        for q_label, q_docs, has_conf, is_suff in quadrants:
            probe_items.append({
                "probe_id": f"{q_label}__{it['id']}",
                "source_id": it["id"],
                "quadrant": q_label,
                "question": it["question"],
                "possible_answers": it["possible_answers"],
                "docs": q_docs,
                "has_conflict": has_conf,
                "evidence_sufficient": is_suff,
                "expected_eu_high": (not is_suff) or has_conf,
                "expected_au_high": has_conf or (not is_suff),
            })

    out_path = PHASE2_CACHE_DIR / "wp1_four_quadrant_probe.json"
    payload = {
        "metadata": {"total_probes": len(probe_items), "max_per_quadrant": max_per_quadrant},
        "items": probe_items,
    }
    atomic_write_json(out_path, payload)
    logger.info(f"Saved 4-quadrant WP1 probe: {out_path} ({len(probe_items)} probe instances).")
    return payload


def load_or_build_cache(
    dataset_name: str,
    pool_k: int = 16,
    allow_synthetic_fallback: bool = False,
) -> Dict[str, Any]:
    """Load pre-computed cache JSON if present, else build on-the-fly safely on CPU."""
    cache_path = PHASE2_CACHE_DIR / f"{dataset_name}_pool_k{pool_k}.json"
    if cache_path.exists():
        return load_json_file(cache_path)

    loaded = ensure_dataset_files(
        [dataset_name] + (["hotpotqa"] if dataset_name == "popqa" else []),
        limit_per_dataset=120 if allow_synthetic_fallback else 300,
        hotpot_limit=300,
        download=False,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )
    dense_model = _load_cpu_dense_model(dry_run=allow_synthetic_fallback)
    popqa_corpus, popqa_gold = (None, None)
    if dataset_name == "popqa":
        popqa_corpus, popqa_gold = build_popqa_neutral_corpus(
            loaded["popqa"], loaded["hotpotqa"], mode="natural"
        )
    return build_shared_pool_cache(
        dataset_name=dataset_name,
        samples=loaded[dataset_name],
        pool_k=pool_k,
        shared_dense_model=dense_model,
        popqa_corpus=popqa_corpus,
        popqa_gold_map=popqa_gold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="WP0: Build Phase II shared datasets & Top-K candidate caches.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa", "popqa", "2wikimultihop", "musique"],
        choices=["hotpotqa", "popqa", "2wikimultihop", "musique"],
    )
    parser.add_argument("--limit", type=int, default=300, help="Samples per dataset (except --hotpot-limit)")
    parser.add_argument("--hotpot-limit", type=int, default=300, help="HotpotQA sample count (300 or 500)")
    parser.add_argument("--pool-k", type=int, default=16, help="Top-K candidate pool size to freeze")
    parser.add_argument("--download", action="store_true", help="Download missing HF datasets cleanly")
    parser.add_argument(
        "--popqa-corpus-mode",
        type=str,
        default="natural",
        choices=["natural", "v5_compat"],
        help="Neutral PopQA corpus builder vs V5 compatibility mode",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic fallback & TF-IDF dense for CPU smoke test")
    args = parser.parse_args()

    v5_info = verify_v5_frozen_checksums()
    logger.info(f"Verified V5_FROZEN archive: {v5_info}")

    need = list(dict.fromkeys(list(args.datasets) + (["hotpotqa"] if "popqa" in args.datasets else [])))
    loaded = ensure_dataset_files(
        need,
        limit_per_dataset=args.limit,
        hotpot_limit=args.hotpot_limit,
        download=args.download and not args.dry_run,
        allow_synthetic_fallback=args.dry_run,
    )

    dense_model = _load_cpu_dense_model(dry_run=args.dry_run)
    caches: Dict[str, Any] = {}
    for dname in args.datasets:
        pop_corpus, pop_gold = (None, None)
        if dname == "popqa":
            pop_corpus, pop_gold = build_popqa_neutral_corpus(
                loaded["popqa"], loaded["hotpotqa"], mode=args.popqa_corpus_mode
            )
        caches[dname] = build_shared_pool_cache(
            dataset_name=dname,
            samples=loaded[dname],
            pool_k=args.pool_k,
            shared_dense_model=dense_model,
            popqa_corpus=pop_corpus,
            popqa_gold_map=pop_gold,
        )

    if "hotpotqa" in caches:
        build_four_quadrant_probe(caches["hotpotqa"], max_per_quadrant=15 if args.dry_run else 40)

    logger.info("WP0 Cache Builder completed cleanly.")


if __name__ == "__main__":
    main()
