# -*- coding: utf-8 -*-
"""
scripts/download_datasets.py
============================
Reliable dataset fetcher & manifest generator for Step 1 Benchmark:
- HotpotQA (distractor validation, top 300) -> data/hotpotqa_dev_300.json
- PopQA (test split, top 300)              -> data/popqa_test_300.json
- BAAI/bge-small-en-v1.5 retrieval model    -> models/bge-small-en-v1.5
- Manifest generation                      -> data/DATASET_MANIFEST.json

Supports direct huggingface.co connection and hf-mirror.com fallback.
"""

import os
import json
import io
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
MODELS_DIR = WORKSPACE_DIR / "models"


def fetch_hotpotqa_dev_300(out_path: Path, limit: int = 300) -> List[Dict[str, Any]]:
    print(f"[1/3] Downloading HotpotQA distractor validation split (top {limit})...")
    urls = [
        "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet",
        "https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
    ]
    
    buf = None
    for url in urls:
        try:
            print(f"  Attempting {url} ...")
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                buf = io.BytesIO(resp.content)
                print(f"  Successfully fetched {len(resp.content):,} bytes.")
                break
        except Exception as e:
            print(f"  Failed from {url}: {e}")
            continue

    if buf is None:
        raise RuntimeError("Failed to download HotpotQA parquet from all candidate mirrors.")

    df = pd.read_parquet(buf)
    print(f"  Full validation split size: {len(df)} rows.")

    df_subset = df.iloc[:limit]
    records = []
    for _, row in df_subset.iterrows():
        # Handle context object to ensure standard python list format
        ctx = row["context"]
        titles = list(ctx["title"]) if hasattr(ctx["title"], "tolist") else list(ctx["title"])
        sentences = []
        for s_list in ctx["sentences"]:
            sentences.append(list(s_list) if hasattr(s_list, "tolist") else list(s_list))

        sf = row.get("supporting_facts")
        sf_titles = []
        sf_sent_ids = []
        if isinstance(sf, dict):
            raw_t = sf.get("title", [])
            raw_s = sf.get("sent_id", [])
            sf_titles = [str(x) for x in (raw_t.tolist() if hasattr(raw_t, "tolist") else raw_t)]
            sf_sent_ids = [int(x) for x in (raw_s.tolist() if hasattr(raw_s, "tolist") else raw_s)]

        item = {
            "id": str(row["id"]),
            "question": str(row["question"]),
            "ground_truth": str(row["answer"]),
            "answer": str(row["answer"]),
            "type": str(row.get("type", "unknown")),
            "level": str(row.get("level", "unknown")),
            "supporting_facts": {
                "title": sf_titles,
                "sent_id": sf_sent_ids,
            },
            "context": {
                "title": titles,
                "sentences": sentences
            }
        }
        records.append(item)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(records)} records to {out_path} ({out_path.stat().st_size:,} bytes).")
    return records


def fetch_popqa_test_300(out_path: Path, limit: int = 300) -> List[Dict[str, Any]]:
    print(f"[2/3] Downloading PopQA test split (top {limit})...")
    urls = [
        "https://huggingface.co/datasets/akariasai/PopQA/resolve/main/test.tsv",
        "https://hf-mirror.com/datasets/akariasai/PopQA/resolve/main/test.tsv"
    ]

    text_content = None
    for url in urls:
        try:
            print(f"  Attempting {url} ...")
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                text_content = resp.text
                print(f"  Successfully fetched PopQA tsv ({len(resp.content):,} bytes).")
                break
        except Exception as e:
            print(f"  Failed from {url}: {e}")
            continue

    if text_content is None:
        raise RuntimeError("Failed to download PopQA tsv from all candidate mirrors.")

    df = pd.read_csv(io.StringIO(text_content), sep="\t")
    print(f"  Full test split size: {len(df)} rows.")

    # Stratified sampling across the 5 popularity bins so every prefix (e.g. top 100 or 300) is balanced across bins and relations
    df["s_pop_clean"] = df["s_pop"].fillna(0).astype(int)
    bin_ranges = [
        (0, 100),
        (100, 1000),
        (1000, 10000),
        (10000, 100000),
        (100000, 10**15),
    ]
    per_bin = max(1, limit // len(bin_ranges))
    bin_dfs = []
    for lo, hi in bin_ranges:
        sub = df[(df["s_pop_clean"] >= lo) & (df["s_pop_clean"] < hi)]
        if len(sub) > per_bin:
            sub = sub.sample(n=per_bin, random_state=42)
        bin_dfs.append(sub.reset_index(drop=True))

    # Interleave rows from the 5 bins so any [:N] slice has equal representation across all 5 popularity strata
    interleaved_rows = []
    max_len = max((len(bdf) for bdf in bin_dfs), default=0)
    for idx in range(max_len):
        for bdf in bin_dfs:
            if idx < len(bdf):
                interleaved_rows.append(bdf.iloc[idx])
                if len(interleaved_rows) >= limit:
                    break
        if len(interleaved_rows) >= limit:
            break

    records = []
    for row in interleaved_rows[:limit]:
        # Parse possible answers
        possible_answers_raw = row["possible_answers"]
        if isinstance(possible_answers_raw, str):
            try:
                possible_answers = json.loads(possible_answers_raw)
            except Exception:
                possible_answers = [possible_answers_raw]
        elif isinstance(possible_answers_raw, list):
            possible_answers = possible_answers_raw
        else:
            possible_answers = [str(possible_answers_raw)]

        subj_str = str(row["subj"]) if pd.notnull(row.get("subj")) else ""
        prop_str = str(row["prop"]) if pd.notnull(row.get("prop")) else ""
        obj_str = str(row["obj"]) if pd.notnull(row.get("obj")) else ""
        s_wiki = str(row["s_wiki_title"]) if pd.notnull(row.get("s_wiki_title")) else subj_str
        o_wiki = str(row["o_wiki_title"]) if pd.notnull(row.get("o_wiki_title")) else obj_str

        item = {
            "id": int(row["id"]),
            "question": str(row["question"]),
            "subj": subj_str,
            "prop": prop_str,
            "obj": obj_str,
            "s_wiki_title": s_wiki,
            "o_wiki_title": o_wiki,
            "ground_truth": obj_str,
            "possible_answers": possible_answers,
            "s_pop": int(row["s_pop"]) if pd.notnull(row["s_pop"]) else 0,
            "o_pop": int(row["o_pop"]) if pd.notnull(row["o_pop"]) else 0,
        }
        records.append(item)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(records)} records to {out_path} ({out_path.stat().st_size:,} bytes).")
    return records


def download_and_verify_bge_small(target_dir: Path) -> Path:
    print(f"[3/3] Checking and downloading BAAI/bge-small-en-v1.5 to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    required_file = target_dir / "model.safetensors"

    if not required_file.exists():
        from huggingface_hub import snapshot_download
        print("  Downloading model weights from HuggingFace...")
        # Direct snapshot download
        snapshot_download(
            repo_id="BAAI/bge-small-en-v1.5",
            local_dir=str(target_dir),
            ignore_patterns=["*.onnx*", "pytorch_model.bin"]
        )
    else:
        print(f"  Model weights already exist ({required_file.stat().st_size:,} bytes).")

    # Offline verification with sentence-transformers
    print("  Verifying offline loading with SentenceTransformer...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(target_dir), device="cpu")
    emb = model.encode(["Sanity verification query for BGE small"], convert_to_numpy=True)
    assert emb.shape == (1, 384), f"Unexpected embedding shape: {emb.shape}"
    print(f"  Offline load verification successful! Embedding dim: {emb.shape[1]}")
    return target_dir


def generate_manifest(
    hotpot_path: Path,
    hotpot_records: List[Dict[str, Any]],
    popqa_path: Path,
    popqa_records: List[Dict[str, Any]],
    bge_path: Path,
    manifest_path: Path
):
    print("Generating data/DATASET_MANIFEST.json ...")
    manifest = {
        "version": "1.0",
        "description": "Step 1 Benchmark Dataset & Embedding Assets Manifest",
        "datasets": {
            "hotpotqa_dev_300": {
                "file_path": str(hotpot_path.relative_to(WORKSPACE_DIR)),
                "num_samples": len(hotpot_records),
                "split": "validation",
                "setting": "distractor (10 paragraphs per question)",
                "sample_keys": list(hotpot_records[0].keys()),
                "sample_preview": {
                    "id": hotpot_records[0]["id"],
                    "question": hotpot_records[0]["question"],
                    "ground_truth": hotpot_records[0]["ground_truth"],
                    "type": hotpot_records[0]["type"],
                    "level": hotpot_records[0]["level"],
                    "num_context_docs": len(hotpot_records[0]["context"]["title"])
                }
            },
            "popqa_test_300": {
                "file_path": str(popqa_path.relative_to(WORKSPACE_DIR)),
                "num_samples": len(popqa_records),
                "split": "test",
                "sample_keys": list(popqa_records[0].keys()),
                "sample_preview": {
                    "id": popqa_records[0]["id"],
                    "question": popqa_records[0]["question"],
                    "ground_truth": popqa_records[0]["ground_truth"],
                    "possible_answers": popqa_records[0]["possible_answers"],
                    "s_pop": popqa_records[0]["s_pop"],
                    "o_pop": popqa_records[0]["o_pop"],
                    "prop": popqa_records[0]["prop"]
                }
            }
        },
        "models": {
            "retrieval_embedding": {
                "name": "BAAI/bge-small-en-v1.5",
                "local_dir": str(bge_path.relative_to(WORKSPACE_DIR)),
                "embedding_dim": 384,
                "framework": "sentence-transformers / HuggingFace",
                "status": "verified_offline"
            }
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest written to {manifest_path} ({manifest_path.stat().st_size:,} bytes).")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hotpot_path = DATA_DIR / "hotpotqa_dev_300.json"
    popqa_path = DATA_DIR / "popqa_test_300.json"
    manifest_path = DATA_DIR / "DATASET_MANIFEST.json"
    bge_dir = MODELS_DIR / "bge-small-en-v1.5"

    hotpot_records = fetch_hotpotqa_dev_300(hotpot_path, limit=300)
    popqa_records = fetch_popqa_test_300(popqa_path, limit=300)
    bge_path = download_and_verify_bge_small(bge_dir)

    generate_manifest(
        hotpot_path=hotpot_path,
        hotpot_records=hotpot_records,
        popqa_path=popqa_path,
        popqa_records=popqa_records,
        bge_path=bge_path,
        manifest_path=manifest_path
    )
    print("\nDataset downloading, offline model verification, and manifest setup finished successfully!")


if __name__ == "__main__":
    main()
