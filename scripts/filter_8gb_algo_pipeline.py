"""
8GB VRAM Feasible & Algorithm/Data-Pipeline Focused ACL Paper Selector
Pure Python pipeline with 0 LLM Token consumption.

Selection Philosophy:
1. Strict Hardware Feasibility: Fits within single-GPU 8GB VRAM (API / CPU / Lightweight SLM / Encoder).
   Heavy compute (pretraining, full fine-tuning of 70B/32B/13B, RLHF/PPO/GRPO/DPO, deepspeed/megatron/multi-gpu, fp8/kernels)
   is penalized and hard-excluded unless explicitly training-free/black-box.
2. Strict Non-Gating: Papers dominated by gating, routing, selective classification, early exit, abstention are excluded.
3. Dual Core Pillars: Balanced coverage across Data Pipeline & Curation vs. Training-Free Inference Algorithms,
   with robust evaluation on domain fit (conflicts, fact verification, hallucination, financial/numerical).
"""

import sys
import os
import json
import re
import shutil
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

# Base directory configuration
BASE_DIR = Path(r"F:\desktop\MyRouter\Nekoya\nekolab\[SDUFE]GeminiOutOfToken_Team")
MEMPALACE_DIR = BASE_DIR / "mempalace_acl"
CACHE_XML_DIR = MEMPALACE_DIR / "cache_xml"
ORIGINAL_OUTPUT_PATH = MEMPALACE_DIR / "selected_papers.json"
BACKUP_OUTPUT_PATH = MEMPALACE_DIR / "selected_papers_v1_backup.json"
OUTPUT_8GB_PATH = MEMPALACE_DIR / "selected_papers_8gb.json"
REPORT_MD_PATH = MEMPALACE_DIR / "selection_8gb_report.md"

# ACL Anthology XML Sources
XML_SOURCES = [
    ("2026.acl", "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2026.acl.xml"),
    ("2025.acl", "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2025.acl.xml"),
    ("2025.naacl", "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2025.naacl.xml"),
    ("2025.emnlp", "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2025.emnlp.xml"),
    ("2024.acl", "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2024.acl.xml"),
    ("2024.emnlp", "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2024.emnlp.xml"),
]

# Taxonomy and Scoring Configurations
TAXONOMY = {
    "data_pipeline_curation": {
        "weight": 3.5,
        "keywords": [
            "data pipeline", "data curation", "synthetic data", "data filtering",
            "claim decomposition", "evidence retrieval", "knowledge graph",
            "graph construction", "retrieval-augmented", "retrieval augmented",
            "data selection", "dataset construction", "denoising", "corpus",
            "structuring", "counterfactual data", "sample selection", "data synthesis",
            "curated dataset", "evidence extraction", "information extraction",
            "knowledge base", "document filtering", "data cleansing"
        ]
    },
    "training_free_inference_algo": {
        "weight": 3.6,
        "keywords": [
            "training-free", "training free", "zero-shot", "zero shot",
            "test-time", "test time", "inference-time", "inference time",
            "post-hoc", "post hoc", "conformal prediction", "bayesian",
            "probabilistic", "graph propagation", "optimization",
            "information-theoretic", "entropy", "mutual information",
            "iterative refinement", "debate", "critique", "self-consistency",
            "self consistency", "consensus", "aggregation", "verification",
            "causal", "counterfactual", "reranking", "re-ranking",
            "scoring function", "algorithmic reasoning", "prompt optimization"
        ]
    },
    "lightweight_8gb_friendly": {
        "weight": 2.6,
        "keywords": [
            "small language model", "slm", "bert", "roberta", "deberta", "finbert",
            "encoder", "embedding", "black-box", "black box", "api", "prompting",
            "workflow", "agentic", "quantization", "4-bit", "8-bit", "lightweight",
            "cpu", "low-resource compute", "low resource", "in-context learning",
            "icl", "few-shot", "prompt-based", "offline pipeline", "shallow model"
        ]
    },
    "domain_fit_robustness": {
        "weight": 2.8,
        "keywords": [
            "conflicting evidence", "fact verification", "misinformation",
            "sycophancy", "hallucination", "faithfulness", "factual consistency",
            "financial", "numerical reasoning", "tabular", "table reasoning",
            "confidence calibration", "uncertainty", "truthfulness", "unfaithful",
            "fact-checking", "factual hallucination"
        ]
    }
}

# Gating Specific Hard-Exclusion Keywords
GATING_KEYWORDS = [
    "gating", "selective classification", "selective prediction",
    "router", "routing mechanism", "dynamic routing", "mixture of routers",
    "abstention threshold", "abstain threshold", "early exit",
    "abstention mechanism", "deferral", "learning to abstain", "abstain"
]

# Heavy Compute Patterns that break 8GB VRAM feasibility
HEAVY_COMPUTE_PATTERNS = [
    r"\bpretraining\b", r"\bpre-training\b", r"\bcontinual pretraining\b", r"\bcontinual pre-training\b",
    r"\bfull fine-tuning\b", r"\bfull parameter fine-tuning\b", r"\brlhf\b", r"\bppo\b", r"\bgrpo\b", r"\bdpo\b",
    r"\bsupervised fine-tuning\b", r"\bsft\b", r"\bdistributed training\b", r"\bdeepspeed\b", r"\bfsdp\b",
    r"\bmegatron\b", r"\bmulti-gpu\b", r"\bmulti gpu\b", r"\ba100\b", r"\bh100\b", r"\bv100\b",
    r"\b8x\b", r"\bfp8\b", r"\bcuda kernel\b", r"\btriton kernel\b", r"\bcustom kernel\b",
    r"\b70b fine-tuning\b", r"\b32b fine-tuning\b", r"\b13b fine-tuning\b", r"\b14b fine-tuning\b",
    r"\bcluster of\b", r"\bgpu hours\b", r"\btpus\b"
]

# Lightweight Shield Patterns that protect against false-positive rejection
LIGHTWEIGHT_SHIELD_PATTERNS = [
    r"\btraining-free\b", r"\btraining free\b", r"\bzero-shot\b", r"\bzero shot\b",
    r"\bblack-box\b", r"\bblack box\b", r"\bapi\b", r"\bprompting\b", r"\bin-context\b",
    r"\bfrozen\b", r"\bno training\b", r"\bwithout any training\b", r"\bwithout fine-tuning\b",
    r"\binference-only\b", r"\bencoder-only\b", r"\bsmall language model\b", r"\bslm\b",
    r"\bbert\b", r"\broberta\b", r"\bdeberta\b", r"\bfinbert\b", r"\b4-bit\b", r"\bquantiz\b"
]


def setup_and_backup():
    """Ensure directories exist and backup original selected_papers.json if not already done."""
    MEMPALACE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_XML_DIR.mkdir(parents=True, exist_ok=True)

    if ORIGINAL_OUTPUT_PATH.exists():
        if not BACKUP_OUTPUT_PATH.exists():
            shutil.copy2(ORIGINAL_OUTPUT_PATH, BACKUP_OUTPUT_PATH)
            print(f"[✔] Created initial backup: {BACKUP_OUTPUT_PATH} ({BACKUP_OUTPUT_PATH.stat().st_size:,} bytes)")
        else:
            print(f"[*] Backup already exists at: {BACKUP_OUTPUT_PATH}")
    else:
        print("[*] No existing selected_papers.json found to backup.")


def get_xml_content(name: str, url: str) -> str | None:
    """Read XML from cache_xml directory or download from ACL Anthology."""
    cache_file = CACHE_XML_DIR / f"{name}.xml"

    # Check local cache first
    if cache_file.exists() and cache_file.stat().st_size > 50000:
        print(f"[+] Loaded {name} from local cache ({cache_file.stat().st_size:,} bytes)")
        with open(cache_file, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    # Otherwise fetch from network
    print(f"[*] Fetching {name} from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read().decode("utf-8", errors="replace")
                print(f"[+] Successfully downloaded {name} ({len(data):,} chars)")
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(data)
                return data
        except urllib.error.HTTPError as e:
            print(f"[-] HTTP {e.code} for {name} ({url})")
            break
        except Exception as e:
            print(f"[!] Network error on attempt {attempt}/3 for {name}: {e}")
            if attempt == 3 and cache_file.exists():
                print(f"[*] Falling back to existing cache file for {name}")
                with open(cache_file, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()

    return None


def parse_papers_from_xml(xml_content: str, source_tag: str):
    """Parse ACL Anthology XML content into standardized paper dictionaries."""
    try:
        root = ET.fromstring(xml_content)
    except Exception as e:
        print(f"[!] XML parsing failed for {source_tag}: {e}")
        return []

    collection_id = root.attrib.get("id", source_tag)
    papers = []

    for volume in root.findall(".//volume"):
        volume_id = volume.attrib.get("id", "")
        for paper in volume.findall("paper"):
            paper_num = paper.attrib.get("id", "")
            title_el = paper.find("title")
            abstract_el = paper.find("abstract")

            if title_el is None or abstract_el is None:
                continue

            title = "".join(title_el.itertext()).strip()
            abstract = "".join(abstract_el.itertext()).strip()

            if not title or not abstract or len(abstract) < 40:
                continue

            url_el = paper.find("url")
            if url_el is not None and url_el.text and url_el.text.strip():
                anthology_id = url_el.text.strip()
            else:
                if volume_id:
                    anthology_id = f"{collection_id}-{volume_id}.{paper_num}"
                else:
                    anthology_id = f"{collection_id}.{paper_num}"

            pdf_url = f"https://aclanthology.org/{anthology_id}.pdf"

            authors = []
            for a in paper.findall("author"):
                first = (a.findtext("first") or "").strip()
                last = (a.findtext("last") or "").strip()
                full_name = f"{first} {last}".strip()
                if not full_name and a.text:
                    full_name = a.text.strip()
                if full_name:
                    authors.append(full_name)

            year_val = 2024
            m = re.search(r"202\d", source_tag)
            if m:
                year_val = int(m.group(0))

            papers.append({
                "paper_id": anthology_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "volume_id": volume_id,
                "collection_id": collection_id,
                "year": year_val,
                "venue": f"ACL Anthology ({collection_id.upper()})",
                "pdf_url": pdf_url
            })

    print(f"[+] Extracted {len(papers)} candidate papers from {source_tag}")
    return papers


def evaluate_paper_feasibility_and_scores(paper: dict):
    """
    Evaluates:
    1. Gating exclusion (is_gating_focused)
    2. 8GB VRAM feasibility & rejection penalty (heavy compute filtering)
    3. Algorithm / Data-Pipeline category scores
    4. Hardware tier labeling
    """
    title_lower = paper["title"].lower()
    abstract_lower = paper["abstract"].lower()
    combined_text = f"{title_lower} {abstract_lower}"

    # 1. Gating Exclusion Check
    title_gating_hit = any(re.search(r"\b" + re.escape(kw) + r"\b", title_lower) for kw in GATING_KEYWORDS)
    abs_gating_count = sum(len(re.findall(r"\b" + re.escape(kw) + r"\b", abstract_lower)) for kw in GATING_KEYWORDS)
    is_gating_focused = title_gating_hit or (abs_gating_count >= 2)

    # 2. Heavy Compute & 8GB Feasibility Check
    heavy_hits = 0
    for pat in HEAVY_COMPUTE_PATTERNS:
        if re.search(pat, title_lower):
            heavy_hits += 3
        heavy_hits += len(re.findall(pat, abstract_lower))

    shield_hits = 0
    for pat in LIGHTWEIGHT_SHIELD_PATTERNS:
        if re.search(pat, title_lower):
            shield_hits += 3
        shield_hits += len(re.findall(pat, abstract_lower))

    # Strict title heavy compute check (e.g. paper title explicitly says "Pre-training...", "RLHF for...", "CUDA kernel...")
    title_heavy_explicit = any(re.search(pat, title_lower) for pat in [
        r"\bpretraining\b", r"\bpre-training\b", r"\brlhf\b", r"\bgrpo\b", r"\bppo\b", r"\bdpo\b",
        r"\bdistributed training\b", r"\bdeepspeed\b", r"\bfsdp\b", r"\bcuda kernel\b", r"\btriton kernel\b"
    ])

    if title_heavy_explicit and not any(re.search(pat, title_lower) for pat in [r"\btraining-free\b", r"\bzero-shot\b"]):
        hardware_feasibility_score = -50.0
        rejection_penalty = 100.0
    elif heavy_hits >= 3 and shield_hits == 0:
        hardware_feasibility_score = -20.0
        rejection_penalty = 50.0
    else:
        # Feasible: compute positive feasibility score
        hardware_feasibility_score = round(10.0 + (shield_hits * 1.5) - (heavy_hits * 1.0), 2)
        if shield_hits > 0 and hardware_feasibility_score < 5.0:
            hardware_feasibility_score = 5.0
        rejection_penalty = 0.0

    # 3. Category Scoring
    cat_scores = {}
    matched_tags = []
    matched_keywords = []

    for cat_name, cat_meta in TAXONOMY.items():
        score = 0.0
        hits = []
        for kw in cat_meta["keywords"]:
            t_count = len(re.findall(r"\b" + re.escape(kw) + r"\b", title_lower))
            a_count = len(re.findall(r"\b" + re.escape(kw) + r"\b", abstract_lower))
            if t_count > 0 or a_count > 0:
                length_boost = 1.0 + (len(kw.split()) * 0.3)
                kw_score = (t_count * 3.5 + a_count * 1.0) * cat_meta["weight"] * length_boost
                score += kw_score
                hits.append(kw)

        if hits:
            cat_scores[cat_name] = round(score, 2)
            matched_tags.append(cat_name)
            matched_keywords.extend(hits)

    final_score = round(sum(cat_scores.values()), 2)

    # 4. Hardware Tier Determination
    encoder_slm_hits = any(re.search(r"\b" + re.escape(kw) + r"\b", combined_text) for kw in [
        "bert", "roberta", "deberta", "finbert", "encoder", "embedding",
        "small language model", "slm", "quantization", "4-bit", "8-bit"
    ])
    if encoder_slm_hits:
        hardware_tier = "Lightweight-Encoder/8GB-OK"
    else:
        hardware_tier = "API/CPU/Training-Free"

    return {
        **paper,
        "is_gating_focused": is_gating_focused,
        "gating_excluded": not is_gating_focused,
        "hardware_feasibility_score": hardware_feasibility_score,
        "rejection_penalty": rejection_penalty,
        "hardware_tier": hardware_tier,
        "algo_pipeline_tags": matched_tags,
        "category_scores": cat_scores,
        "matched_keywords": list(dict.fromkeys(matched_keywords))[:12],
        "final_score": final_score
    }


def select_balanced_8gb_papers(all_evaluated_papers, target_total=90):
    """
    Selects top 90 papers with balanced coverage across:
    - Data Pipeline & Curation (>= 30 papers)
    - Training-Free Inference Algorithms (>= 30 papers)
    Filtering out gating and heavy compute papers.
    """
    gating_excluded_count = 0
    heavy_compute_excluded_count = 0
    eligible_papers = []

    for p in all_evaluated_papers:
        if p["is_gating_focused"]:
            gating_excluded_count += 1
            continue
        if p["hardware_feasibility_score"] < 0:
            heavy_compute_excluded_count += 1
            continue
        if p["final_score"] < 6.0:
            continue
        eligible_papers.append(p)

    print(f"[*] Total papers evaluated: {len(all_evaluated_papers):,}")
    print(f"    - Gating excluded: {gating_excluded_count:,}")
    print(f"    - Heavy compute / cluster excluded (8GB unfeasible): {heavy_compute_excluded_count:,}")
    print(f"    - Eligible 8GB-feasible papers: {len(eligible_papers):,}")

    # Stratified balance selection
    # Pillar A: data_pipeline_curation
    pipeline_pool = [p for p in eligible_papers if "data_pipeline_curation" in p["algo_pipeline_tags"]]
    pipeline_pool.sort(key=lambda x: (x["category_scores"].get("data_pipeline_curation", 0), x["final_score"]), reverse=True)

    # Pillar B: training_free_inference_algo
    training_free_pool = [p for p in eligible_papers if "training_free_inference_algo" in p["algo_pipeline_tags"]]
    training_free_pool.sort(key=lambda x: (x["category_scores"].get("training_free_inference_algo", 0), x["final_score"]), reverse=True)

    selected = []
    selected_ids = set()

    # Step 1: Select up to 30 top papers for Pillar A
    quota_pipeline = min(30, len(pipeline_pool))
    for p in pipeline_pool[:quota_pipeline]:
        selected.append(p)
        selected_ids.add(p["paper_id"])

    # Step 2: Select up to 30 top papers for Pillar B (excluding already selected)
    quota_tf = 30
    added_tf = 0
    for p in training_free_pool:
        if added_tf >= quota_tf:
            break
        if p["paper_id"] not in selected_ids:
            selected.append(p)
            selected_ids.add(p["paper_id"])
            added_tf += 1

    # Step 3: Fill remaining slots up to target_total with highest final_score eligible papers
    remaining_eligible = [p for p in eligible_papers if p["paper_id"] not in selected_ids]
    remaining_eligible.sort(key=lambda x: x["final_score"], reverse=True)

    needed = target_total - len(selected)
    for p in remaining_eligible[:needed]:
        selected.append(p)
        selected_ids.add(p["paper_id"])

    # Sort final selected papers by final_score descending
    selected.sort(key=lambda x: x["final_score"], reverse=True)

    stats = {
        "total_evaluated": len(all_evaluated_papers),
        "gating_excluded_count": gating_excluded_count,
        "heavy_compute_excluded_count": heavy_compute_excluded_count,
        "eligible_count": len(eligible_papers),
        "selected_count": len(selected),
    }

    return selected, stats


def generate_markdown_report(selected_papers, stats, report_path: Path):
    """Generate comprehensive markdown summary report for review."""
    tier_counts = Counter(p["hardware_tier"] for p in selected_papers)
    tag_counts = Counter()
    venue_counts = Counter(p["collection_id"] for p in selected_papers)

    for p in selected_papers:
        for tag in p["algo_pipeline_tags"]:
            tag_counts[tag] += 1

    top20 = selected_papers[:20]

    report_lines = [
        "# ACL Anthology 8GB 友好型精选论文报告 (8GB-Feasible & Algo/Pipeline Focused)",
        "",
        "## 1. 筛选背景与核心准则",
        "- **显存硬约束**：严格适配单卡 8GB 显存 / 本地 CPU / 商用 API 推理，重罚并剔除集群级预训练、全量 70B/32B 微调、RLHF/PPO/GRPO/DPO 及底层 CUDA/Triton 核函数论文。",
        "- **非门控原则**：全面剔除以 gating、selective classification、dynamic routing、early exit、abstention 为核心的门控导向论文。",
        "- **双支柱均衡**：严格保障 **数据管道 (Data Pipeline & Curation)** 与 **无训练算法 (Training-Free Inference Algo)** 均衡覆盖，配合事实核查、反事实、幻觉抑制与财务/数值推理场景。",
        "",
        "## 2. 宏观统计数据",
        f"- **总解析论文量**：{stats['total_evaluated']:,} 篇",
        f"- **门控主导剔除量**：{stats['gating_excluded_count']:,} 篇 (`is_gating_focused = True`)",
        f"- **重算力超标剔除量**：{stats['heavy_compute_excluded_count']:,} 篇 (`hardware_feasibility_score < 0`)",
        f"- **合规候选池**：{stats['eligible_count']:,} 篇",
        f"- **最终入选 Top 论文**：{len(selected_papers)} 篇 (100% Non-gating, 100% 8GB-Feasible)",
        "",
        "### 硬件层级分布 (Hardware Tier Distribution)",
    ]

    for tier, count in tier_counts.most_common():
        pct = (count / len(selected_papers)) * 100
        report_lines.append(f"- **{tier}**: {count} 篇 ({pct:.1f}%)")

    report_lines.append("")
    report_lines.append("### 算法与管道标签分布 (Algo/Pipeline Tags Distribution)")
    for tag, count in tag_counts.most_common():
        pct = (count / len(selected_papers)) * 100
        report_lines.append(f"- **{tag}**: {count} 篇 ({pct:.1f}%)")

    report_lines.append("")
    report_lines.append("### 顶会来源分布 (Venue Distribution)")
    for venue, count in venue_counts.most_common():
        report_lines.append(f"- **{venue}**: {count} 篇")

    report_lines.append("")
    report_lines.append("## 3. Top 20 核心论文清单")
    report_lines.append("")
    report_lines.append("| 排名 | Paper ID | 论文标题 | 硬件层级 | 核心标签 | 得分 | PDF 链接 |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for idx, p in enumerate(top20, 1):
        tags_str = ", ".join(p["algo_pipeline_tags"])
        title_esc = p["title"].replace("|", "\\|")
        report_lines.append(f"| {idx} | `{p['paper_id']}` | {title_esc} | {p['hardware_tier']} | {tags_str} | {p['final_score']} | [PDF]({p['pdf_url']}) |")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Report automatically generated by `scripts/filter_8gb_algo_pipeline.py` (0 LLM Token).*")

    content = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[✔] Markdown report saved to {report_path} ({len(content):,} chars)")


def main():
    print("=" * 70)
    print("ACL Anthology 8GB-Feasible & Algo/Pipeline Focused Selector")
    print("=" * 70)

    # Step 1: Setup dirs and backup
    setup_and_backup()

    # Step 2: Download or load cached XML dumps
    all_raw_papers = []
    for tag, url in XML_SOURCES:
        xml_content = get_xml_content(tag, url)
        if xml_content:
            papers = parse_papers_from_xml(xml_content, tag)
            all_raw_papers.extend(papers)
        else:
            print(f"[!] Warning: Unable to obtain XML for {tag}")

    print(f"\n[*] Total candidate papers parsed: {len(all_raw_papers):,}")
    if not all_raw_papers:
        print("[!] Fatal: No candidate papers extracted.")
        sys.exit(1)

    # Step 3: Evaluate feasibility and relevance scores
    print("[*] Evaluating 8GB feasibility, gating exclusion, and category taxonomy ...")
    evaluated_papers = [evaluate_paper_feasibility_and_scores(p) for p in all_raw_papers]

    # Step 4: Balanced selection
    selected_papers, stats = select_balanced_8gb_papers(evaluated_papers, target_total=90)

    # Prepare export JSON payload
    export_payload = {
        "metadata": {
            "total_selected": len(selected_papers),
            "target_total": 90,
            "hardware_constraint": "Single-GPU 8GB VRAM / Local CPU / Commercial API",
            "gating_policy": "Strict Non-Gating (all gating_focused papers excluded)",
            "selection_strategy": "Pure Python 8GB Feasibility & Algorithm/Data-Pipeline Curation Scoring (0 LLM Token)",
            "hardware_tier_distribution": dict(Counter(p["hardware_tier"] for p in selected_papers)),
            "algo_pipeline_tag_distribution": dict(Counter(tag for p in selected_papers for tag in p["algo_pipeline_tags"])),
            "venue_distribution": dict(Counter(p["collection_id"] for p in selected_papers)),
            "filtering_stats": stats
        },
        "papers": selected_papers
    }

    # Step 5: Save JSONs (both selected_papers_8gb.json and selected_papers.json)
    with open(OUTPUT_8GB_PATH, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)
    print(f"[✔] Successfully saved 8GB-selected papers to {OUTPUT_8GB_PATH} ({OUTPUT_8GB_PATH.stat().st_size:,} bytes)")

    with open(ORIGINAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)
    print(f"[✔] Successfully overwritten active {ORIGINAL_OUTPUT_PATH} ({ORIGINAL_OUTPUT_PATH.stat().st_size:,} bytes)")

    # Step 6: Generate Markdown summary report
    generate_markdown_report(selected_papers, stats, REPORT_MD_PATH)

    print("\n[✔] Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
