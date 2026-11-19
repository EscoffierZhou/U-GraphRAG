"""
ACL Anthology Metadata Harvester and Relevance Ranker
Harvests XML dumps directly from ACL Anthology, parses papers with abstracts,
and ranks them via pure-Python keyword and TF-IDF weighted scoring across 5 key research areas:
1. Multi-agent critique, collaboration, and debate
2. Confidence calibration and uncertainty (with is_gating_focused tag)
3. Conflicting evidence, fact verification, sycophancy, and hallucination
4. Reasoning efficiency and context compression
5. Financial and numerical reasoning
"""

import sys
import json
import math
import re
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

# Ensure exact absolute working directory to avoid bracket globbing issues
BASE_DIR = Path(r"F:\desktop\MyRouter\Nekoya\nekolab\[SDUFE]GeminiOutOfToken_Team")
OUTPUT_PATH = BASE_DIR / "mempalace_acl" / "selected_papers.json"

# Candidate XML URLs on official ACL Anthology Github repository
XML_SOURCES = [
    ("2026.acl", "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2026.acl.xml"),
    ("2025.acl", "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2025.acl.xml"),
    ("2025.naacl", "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2025.naacl.xml"),
    ("2025.emnlp", "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2025.emnlp.xml"),
    ("2024.acl", "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2024.acl.xml"),
    ("2024.emnlp", "https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/2024.emnlp.xml"),
]

# Targeted domain keywords and category weights
CATEGORY_TAXONOMY = {
    "multi_agent_critique_debate": {
        "weight": 3.2,
        "keywords": [
            "multi-agent", "multi agent", "multiagent", "debate", "critique",
            "collaborative reasoning", "agent collaboration", "consensus",
            "peer review", "agent society", "cooperative agents", "role-playing agents",
            "self-refinement", "multi-turn critique", "evaluator agent"
        ]
    },
    "confidence_calibration_uncertainty": {
        "weight": 2.2,
        "keywords": [
            "confidence calibration", "selective classification", "gating", "abstention",
            "uncertainty estimation", "calibrated confidence", "refusal", "when to trust",
            "overconfidence", "selective generation", "selective prediction", "abstain",
            "uncertainty quantification", "expected calibration error"
        ]
    },
    "conflicting_evidence_fact_verification": {
        "weight": 2.8,
        "keywords": [
            "conflicting evidence", "fact verification", "sycophancy", "hallucination",
            "factual consistency", "counterfactual", "knowledge conflict", "faithfulness",
            "truthfulness", "misinformation", "claim verification", "unfaithful",
            "hallucinated", "sycophantic"
        ]
    },
    "reasoning_efficiency_context_compression": {
        "weight": 2.4,
        "keywords": [
            "reasoning efficiency", "context compression", "kv cache", "prompt compression",
            "efficient reasoning", "chain-of-thought compression", "token efficiency",
            "compact context", "summarization compression", "selective pruning",
            "inference latency", "token-efficient"
        ]
    },
    "financial_numerical_reasoning": {
        "weight": 2.8,
        "keywords": [
            "financial reasoning", "numerical reasoning", "mathematical reasoning", "finqa",
            "tabular reasoning", "arithmetic reasoning", "quantitative", "stock forecasting",
            "earnings call", "math word problem", "financial table", "table qa"
        ]
    }
}

GATING_SPECIFIC_KEYWORDS = [
    "gating", "selective classification", "selective prediction",
    "abstention", "router", "routing", "dynamic routing",
    "abstain", "early exit", "confidence threshold"
]


def download_xml(name: str, url: str) -> str | None:
    print(f"[*] Attempting to fetch {name} from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            data = response.read().decode("utf-8", errors="replace")
            print(f"[+] Successfully fetched {name} ({len(data):,} chars)")
            return data
    except urllib.error.HTTPError as e:
        print(f"[-] HTTP {e.code} for {name} ({url}), skipping.")
    except Exception as e:
        print(f"[-] Network error fetching {name}: {e}, skipping.")
    return None


def parse_papers_from_xml(xml_content: str, source_tag: str):
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

            # Determine anthology ID and PDF URL
            url_el = paper.find("url")
            if url_el is not None and url_el.text and url_el.text.strip():
                anthology_id = url_el.text.strip()
            else:
                if volume_id:
                    anthology_id = f"{collection_id}-{volume_id}.{paper_num}"
                else:
                    anthology_id = f"{collection_id}.{paper_num}"

            # Standard ACL Anthology PDF URL
            pdf_url = f"https://aclanthology.org/{anthology_id}.pdf"

            # Parse authors
            authors = []
            for a in paper.findall("author"):
                first = (a.findtext("first") or "").strip()
                last = (a.findtext("last") or "").strip()
                full_name = f"{first} {last}".strip()
                if not full_name and a.text:
                    full_name = a.text.strip()
                if full_name:
                    authors.append(full_name)

            # Year extraction
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

    print(f"[+] Extracted {len(papers)} candidate papers with valid abstract from {source_tag}")
    return papers


def compute_relevance_scores(papers):
    scored_papers = []

    for p in papers:
        title_lower = p["title"].lower()
        abstract_lower = p["abstract"].lower()
        combined_text = f"{title_lower} {abstract_lower}"

        total_score = 0.0
        matched_categories = []
        matched_keywords = []

        # Check each category
        for cat_name, cat_meta in CATEGORY_TAXONOMY.items():
            cat_score = 0.0
            cat_hits = []
            for kw in cat_meta["keywords"]:
                # Match title (3.5x weight)
                t_count = len(re.findall(r"\b" + re.escape(kw) + r"\b", title_lower))
                # Match abstract (1.0x weight)
                a_count = len(re.findall(r"\b" + re.escape(kw) + r"\b", abstract_lower))

                if t_count > 0 or a_count > 0:
                    weight_factor = 1.0 + (len(kw.split()) * 0.4)
                    hit_score = (t_count * 3.5 + a_count * 1.0) * cat_meta["weight"] * weight_factor
                    cat_score += hit_score
                    cat_hits.append(kw)

            if cat_hits:
                matched_categories.append(cat_name)
                matched_keywords.extend(cat_hits)
                total_score += cat_score

        # Check is_gating_focused
        gating_hits = 0
        for gkw in GATING_SPECIFIC_KEYWORDS:
            if re.search(r"\b" + re.escape(gkw) + r"\b", title_lower):
                gating_hits += 3
            if re.search(r"\b" + re.escape(gkw) + r"\b", abstract_lower):
                gating_hits += 1

        is_gating = (gating_hits >= 3) or ("gating" in title_lower) or ("selective classification" in title_lower)

        if total_score > 0:
            scored_papers.append({
                **p,
                "relevance_score": round(total_score, 2),
                "is_gating_focused": is_gating,
                "matched_categories": list(set(matched_categories)),
                "matched_keywords": list(set(matched_keywords))[:10]
            })

    return scored_papers


def select_top_papers(scored_papers, target_total=90, max_gating=12):
    # Sort descending by score
    scored_papers.sort(key=lambda x: x["relevance_score"], reverse=True)

    selected = []
    gating_count = 0

    # First pass: high score items honoring max_gating
    for p in scored_papers:
        if len(selected) >= target_total:
            break
        if p["is_gating_focused"]:
            if gating_count < max_gating:
                selected.append(p)
                gating_count += 1
        else:
            selected.append(p)

    # If not full, fill with best remaining papers
    if len(selected) < target_total:
        seen_ids = {p["paper_id"] for p in selected}
        for p in scored_papers:
            if len(selected) >= target_total:
                break
            if p["paper_id"] not in seen_ids:
                selected.append(p)
                if p["is_gating_focused"]:
                    gating_count += 1

    return selected


def main():
    print("=" * 60)
    print("ACL Anthology Metadata Harvester Starting ...")
    print("=" * 60)

    all_papers = []
    for tag, url in XML_SOURCES:
        xml_data = download_xml(tag, url)
        if xml_data:
            papers = parse_papers_from_xml(xml_data, tag)
            all_papers.extend(papers)

    print(f"[*] Total parsed papers across all sources: {len(all_papers):,}")

    if not all_papers:
        print("[!] No papers parsed. Please check network connectivity.")
        sys.exit(1)

    print("[*] Computing relevance scores and taxonomies ...")
    scored = compute_relevance_scores(all_papers)
    print(f"[+] Total papers with positive relevance score: {len(scored):,}")

    top_selected = select_top_papers(scored, target_total=90, max_gating=12)
    print(f"[+] Selected Top {len(top_selected)} papers.")

    # Compute stats
    category_counts = Counter()
    gating_count = 0
    venue_counts = Counter()

    for p in top_selected:
        venue_counts[p["collection_id"]] += 1
        if p["is_gating_focused"]:
            gating_count += 1
        for cat in p["matched_categories"]:
            category_counts[cat] += 1

    output_payload = {
        "total_selected": len(top_selected),
        "target_total": 90,
        "selection_strategy": "Pure Python TF-IDF/Keyword Taxonomy Scoring (0 LLM Token)",
        "gating_focused_count": gating_count,
        "venue_distribution": dict(venue_counts),
        "category_distribution": dict(category_counts),
        "papers": top_selected
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"[✔] Successfully saved {len(top_selected)} papers to {OUTPUT_PATH}")
    print(f"    - File size: {OUTPUT_PATH.stat().st_size:,} bytes")
    print(f"    - Gating focused: {gating_count} / {len(top_selected)}")
    print("    - Category distribution:")
    for cat, count in category_counts.most_common():
        print(f"      * {cat}: {count}")


if __name__ == "__main__":
    main()
