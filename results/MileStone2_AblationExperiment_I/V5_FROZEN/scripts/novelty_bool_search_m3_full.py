#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
novelty_bool_search_m3_full.py
==============================
执行布尔表达式跨学术平台查重（arXiv / OpenAlex / Semantic Scholar / Crossref / 本地ACL全库 2024-2026）
针对 U-GraphRAG 的 Method 3（证据可靠性加权对比解码 CLeHe + 同步句级忠实度监测与回溯/拒答 SYNCHECK/FOD）
以及 M1+M2+M3 全链路组合，构造 6 组从严格到宽泛的布尔检索式，执行严格布尔匹配与真实学术查重，
并将完整检索日志与命中分析落盘到 Debate/search_logs_m3_full.json。
"""

import os
import re
import sys
import json
import time
import glob
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

# 基础目录配置
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEMPALACE_DIR = os.path.join(WORKSPACE_DIR, "mempalace_acl")
CACHE_XML_DIR = os.path.join(MEMPALACE_DIR, "cache_xml")
OUTPUT_JSON_PATH = os.path.join(WORKSPACE_DIR, "Debate", "search_logs_m3_full.json")

# HTTP 请求头配置 (遵守 OpenAlex / Crossref / arXiv 规范)
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) mailto:researcher@sdufe.edu.cn AcademicNoveltyChecker/1.0"
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, application/xml, text/xml, */*"
}
TIMEOUT = 12


# --------------------------------------------------------------------------
# 6 组布尔表达式定义与元数据
# --------------------------------------------------------------------------
QUERIES = [
    {
        "id": "Q7",
        "name": "Full Pipeline Strict",
        "level": "Strict (Full Pipeline M1+M2+M3)",
        "expression": '("Retrieval-Augmented Generation" OR "RAG") AND ("aleatoric" OR "epistemic") AND ("maximum spanning tree") AND ("contrastive decoding" OR "abstention")',
        "description": "全链路复合创新端到端查重：不确定性图谱最大生成树 + 证据对比解码/拒答机制",
        "api_keywords": {
            "arxiv": 'all:"retrieval-augmented generation" AND (all:"aleatoric" OR all:"epistemic" OR all:"maximum spanning tree")',
            "openalex": '"retrieval-augmented generation" "maximum spanning tree" "contrastive decoding"',
            "crossref": 'retrieval-augmented generation maximum spanning tree contrastive decoding',
            "semanticscholar": 'retrieval-augmented generation maximum spanning tree contrastive decoding'
        }
    },
    {
        "id": "Q8",
        "name": "M3 Strict",
        "level": "Strict (Method 3 Core)",
        "expression": '("Retrieval-Augmented Generation" OR "RAG") AND ("contrastive decoding" OR "entropy") AND ("selective abstention" OR "synchronous faithfulness" OR "backtracking")',
        "description": "M3 核心方案查重：RAG 场景下的对比解码/熵引导 + 同步句级忠实度监测/选择性拒答/回溯",
        "api_keywords": {
            "arxiv": 'all:"contrastive decoding" AND (all:"selective abstention" OR all:"synchronous faithfulness" OR all:"backtracking")',
            "openalex": '"contrastive decoding" "selective abstention" "faithfulness"',
            "crossref": 'contrastive decoding selective abstention synchronous faithfulness backtracking',
            "semanticscholar": 'contrastive decoding selective abstention synchronous faithfulness'
        }
    },
    {
        "id": "Q9",
        "name": "M2+M3 Bridge",
        "level": "Medium-Strict (M2-M3 Interface)",
        "expression": '("Retrieval-Augmented Generation" OR "RAG") AND ("evidence reliability" OR "uncertainty") AND ("contrastive decoding")',
        "description": "证据不确定性/可靠性度量与对比解码结合方案查重",
        "api_keywords": {
            "arxiv": 'all:"contrastive decoding" AND (all:"evidence reliability" OR all:"uncertainty") AND (all:"retrieval" OR all:"RAG")',
            "openalex": '"contrastive decoding" "evidence reliability" uncertainty RAG',
            "crossref": 'contrastive decoding evidence reliability uncertainty retrieval augmented generation',
            "semanticscholar": 'contrastive decoding evidence reliability uncertainty RAG'
        }
    },
    {
        "id": "Q10",
        "name": "Contrastive Decoding + Hallucination in RAG",
        "level": "Medium (Direct Competitors)",
        "expression": '("Retrieval-Augmented Generation" OR "RAG") AND ("contrastive decoding") AND ("hallucination" OR "faithfulness")',
        "description": "RAG 领域采用对比解码抑制幻觉或提升忠实度的既有基线工作（如 DoLa, CD, CAD 衍生等）",
        "api_keywords": {
            "arxiv": 'all:"contrastive decoding" AND (all:"hallucination" OR all:"faithfulness") AND (all:"retrieval" OR all:"RAG")',
            "openalex": '"contrastive decoding" hallucination faithfulness RAG',
            "crossref": 'contrastive decoding hallucination faithfulness retrieval-augmented generation',
            "semanticscholar": 'contrastive decoding hallucination faithfulness RAG'
        }
    },
    {
        "id": "Q11",
        "name": "Uncertainty + Selective Abstention in RAG",
        "level": "Broad-Medium (Competitors in Abstention)",
        "expression": '("Retrieval-Augmented Generation" OR "RAG") AND ("uncertainty") AND ("abstention" OR "unanswerable")',
        "description": "RAG 领域基于不确定性度量的选择性拒答或不可回答检测机制",
        "api_keywords": {
            "arxiv": '(all:"retrieval-augmented generation" OR all:"RAG") AND all:"uncertainty" AND (all:"abstention" OR all:"unanswerable")',
            "openalex": '"retrieval-augmented generation" uncertainty abstention unanswerable',
            "crossref": 'retrieval-augmented generation uncertainty abstention unanswerable',
            "semanticscholar": 'retrieval-augmented generation uncertainty abstention unanswerable'
        }
    },
    {
        "id": "Q12",
        "name": "Function Vectors + RAG",
        "level": "Novelty Verification (Intervention & Vectors)",
        "expression": '("function vector" OR "self-function vector" OR "attention head intervention") AND ("Retrieval-Augmented Generation" OR "RAG") AND ("uncertainty")',
        "description": "探测表征工程/功能向量/注意力头干预在 RAG 不确定性调控中的应用",
        "api_keywords": {
            "arxiv": '(all:"function vector" OR all:"attention head intervention") AND (all:"retrieval" OR all:"RAG")',
            "openalex": '"function vector" "retrieval augmented generation" uncertainty',
            "crossref": 'function vector attention head intervention retrieval augmented generation uncertainty',
            "semanticscholar": 'function vector retrieval augmented generation uncertainty'
        }
    }
]


# --------------------------------------------------------------------------
# 严格布尔匹配器
# --------------------------------------------------------------------------
def check_phrase_or_regex(text: str, token: str) -> bool:
    """检查文本中是否出现词组或正则词"""
    t = text.lower()
    tok = token.lower()
    if tok == "rag":
        return bool(re.search(r"\brag\b", t))
    return tok in t


def evaluate_query_match(query_id: str, title: str, abstract: str) -> tuple[bool, dict]:
    """
    对 title + abstract 进行严密布尔逻辑匹配
    返回: (is_matched, matched_clauses_dict)
    """
    full_text = f"{title or ''} {abstract or ''}".lower()

    # 通用子项判断
    has_rag = check_phrase_or_regex(full_text, "retrieval-augmented generation") or check_phrase_or_regex(full_text, "rag")
    has_aleatoric_or_epistemic = "aleatoric" in full_text or "epistemic" in full_text
    has_mst = "maximum spanning tree" in full_text
    has_contrastive_decoding = "contrastive decoding" in full_text
    has_entropy = "entropy" in full_text
    has_abstention = "abstention" in full_text
    has_unanswerable = "unanswerable" in full_text
    has_selective_abstention = "selective abstention" in full_text
    has_sync_faithfulness = "synchronous faithfulness" in full_text
    has_backtracking = "backtracking" in full_text
    has_evidence_reliability = "evidence reliability" in full_text
    has_uncertainty = "uncertainty" in full_text
    has_hallucination = "hallucination" in full_text
    has_faithfulness = "faithfulness" in full_text
    has_func_vector = ("function vector" in full_text or
                        "self-function vector" in full_text or
                        "attention head intervention" in full_text)

    if query_id == "Q7":
        c1 = has_rag
        c2 = has_aleatoric_or_epistemic
        c3 = has_mst
        c4 = has_contrastive_decoding or has_abstention
        matched = c1 and c2 and c3 and c4
        details = {
            "RAG": c1,
            "aleatoric_or_epistemic": c2,
            "maximum_spanning_tree": c3,
            "contrastive_decoding_or_abstention": c4
        }
        return matched, details

    elif query_id == "Q8":
        c1 = has_rag
        c2 = has_contrastive_decoding or has_entropy
        c3 = has_selective_abstention or has_sync_faithfulness or has_backtracking
        matched = c1 and c2 and c3
        details = {
            "RAG": c1,
            "contrastive_decoding_or_entropy": c2,
            "selective_abstention_or_sync_or_backtracking": c3
        }
        return matched, details

    elif query_id == "Q9":
        c1 = has_rag
        c2 = has_evidence_reliability or has_uncertainty
        c3 = has_contrastive_decoding
        matched = c1 and c2 and c3
        details = {
            "RAG": c1,
            "evidence_reliability_or_uncertainty": c2,
            "contrastive_decoding": c3
        }
        return matched, details

    elif query_id == "Q10":
        c1 = has_rag
        c2 = has_contrastive_decoding
        c3 = has_hallucination or has_faithfulness
        matched = c1 and c2 and c3
        details = {
            "RAG": c1,
            "contrastive_decoding": c2,
            "hallucination_or_faithfulness": c3
        }
        return matched, details

    elif query_id == "Q11":
        c1 = has_rag
        c2 = has_uncertainty
        c3 = has_abstention or has_unanswerable
        matched = c1 and c2 and c3
        details = {
            "RAG": c1,
            "uncertainty": c2,
            "abstention_or_unanswerable": c3
        }
        return matched, details

    elif query_id == "Q12":
        c1 = has_func_vector
        c2 = has_rag
        c3 = has_uncertainty
        matched = c1 and c2 and c3
        details = {
            "function_vector_or_intervention": c1,
            "RAG": c2,
            "uncertainty": c3
        }
        return matched, details

    return False, {}


# --------------------------------------------------------------------------
# 本地 ACL Anthology XML 全库加载
# --------------------------------------------------------------------------
def load_local_acl_database():
    """从 mempalace_acl/cache_xml/ 和 mempalace_acl/acl_anthology_xml/ 加载所有论文 (使用 os.listdir 规避路径方括号字符集问题)"""
    candidate_dirs = [CACHE_XML_DIR, os.path.join(MEMPALACE_DIR, "acl_anthology_xml")]
    xml_files = []
    for d in candidate_dirs:
        if os.path.exists(d):
            for fn in os.listdir(d):
                if fn.lower().endswith(".xml"):
                    xml_files.append(os.path.join(d, fn))

    papers = []
    seen_ids = set()

    print(f"[*] Loading Local ACL papers from {len(xml_files)} XML archives...")
    for fpath in sorted(xml_files):
        fname = os.path.basename(fpath)
        try:
            tree = ET.parse(fpath)
            root = tree.getroot()
            collection_id = root.attrib.get("id", fname.replace(".xml", ""))

            file_paper_count = 0
            for volume in root.findall(".//volume"):
                vol_id = volume.attrib.get("id", "")
                for paper in volume.findall("paper"):
                    pid = paper.attrib.get("id", "")
                    title_el = paper.find("title")
                    abstract_el = paper.find("abstract")
                    if title_el is None or abstract_el is None:
                        continue

                    title = "".join(title_el.itertext()).strip()
                    abstract = "".join(abstract_el.itertext()).strip()
                    if not title or not abstract or len(abstract) < 30:
                        continue

                    # Anthology ID
                    url_el = paper.find("url")
                    if url_el is not None and url_el.text and url_el.text.strip():
                        anthology_id = url_el.text.strip()
                    else:
                        anthology_id = f"{collection_id}-{vol_id}.{pid}" if vol_id else f"{collection_id}.{pid}"

                    if anthology_id in seen_ids:
                        continue
                    seen_ids.add(anthology_id)

                    # 作者
                    authors = []
                    for a in paper.findall("author"):
                        first = (a.findtext("first") or "").strip()
                        last = (a.findtext("last") or "").strip()
                        name = f"{first} {last}".strip() or (a.text or "").strip()
                        if name:
                            authors.append(name)

                    # 年份推断
                    m_year = re.search(r"202\d", fname)
                    year = int(m_year.group(0)) if m_year else 2024

                    papers.append({
                        "id": anthology_id,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "year": year,
                        "venue": f"ACL Anthology ({collection_id.upper()})",
                        "url": f"https://aclanthology.org/{anthology_id}"
                    })
                    file_paper_count += 1

            print(f"  - Loaded {file_paper_count:>5} papers from {fname}")
        except Exception as e:
            print(f"  [!] Failed to parse {fname}: {e}")

    print(f"[+] Total unique ACL papers loaded: {len(papers)}")
    return papers


# --------------------------------------------------------------------------
# 外部学术平台检索实现
# --------------------------------------------------------------------------
def http_get_json(url: str, headers: dict = None, timeout: int = TIMEOUT):
    """带超时与异常处理的 GET JSON 请求"""
    req_headers = dict(REQUEST_HEADERS)
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
            return json.loads(data), resp.status
    except Exception as e:
        return None, str(e)


def http_get_xml(url: str, headers: dict = None, timeout: int = TIMEOUT):
    """带超时与异常处理的 GET XML 请求"""
    req_headers = dict(REQUEST_HEADERS)
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
            return data, resp.status
    except Exception as e:
        return None, str(e)


def search_arxiv(query_spec: dict, max_results: int = 25) -> list[dict]:
    """通过 arXiv API 检索文献"""
    raw_query = query_spec["api_keywords"]["arxiv"]
    encoded_query = urllib.parse.quote(raw_query)
    url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"

    results = []
    xml_data, status = http_get_xml(url, timeout=TIMEOUT)
    if not xml_data:
        print(f"    [arXiv] Query failed or timeout: {status}")
        return results

    try:
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)
            published_el = entry.find("atom:published", ns)

            if title_el is None or summary_el is None:
                continue

            title = " ".join(title_el.text.split()) if title_el.text else ""
            abstract = " ".join(summary_el.text.split()) if summary_el.text else ""
            arxiv_url = id_el.text.strip() if id_el is not None and id_el.text else ""
            pub_date = published_el.text if published_el is not None and published_el.text else ""
            year = int(pub_date[:4]) if len(pub_date) >= 4 and pub_date[:4].isdigit() else 2024

            authors = []
            for author_el in entry.findall("atom:author", ns):
                name_el = author_el.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            results.append({
                "id": arxiv_url,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "venue": "arXiv preprint",
                "url": arxiv_url,
                "source": "arXiv"
            })
    except Exception as e:
        print(f"    [arXiv] XML parse exception: {e}")

    return results


def search_openalex(query_spec: dict, max_results: int = 25) -> list[dict]:
    """通过 OpenAlex API 检索文献"""
    raw_query = query_spec["api_keywords"]["openalex"]
    encoded_query = urllib.parse.quote(raw_query)
    url = f"https://api.openalex.org/works?search={encoded_query}&per_page={max_results}&sort=relevance_score:desc"

    results = []
    data, status = http_get_json(url, timeout=TIMEOUT)
    if not data or "results" not in data:
        print(f"    [OpenAlex] Query failed: {status}")
        return results

    for item in data.get("results", []):
        title = item.get("title") or ""
        year = item.get("publication_year") or 2024
        openalex_id = item.get("id") or ""
        doi = item.get("doi") or openalex_id

        # 还原 OpenAlex 倒排索引摘要
        abstract = ""
        inv_index = item.get("abstract_inverted_index")
        if inv_index and isinstance(inv_index, dict):
            word_positions = []
            for word, positions in inv_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort(key=lambda x: x[0])
            abstract = " ".join([w for _, w in word_positions])

        authors = []
        for a in item.get("authorships", []):
            name = a.get("author", {}).get("display_name")
            if name:
                authors.append(name)

        venue_name = "OpenAlex Publication"
        host_venue = item.get("primary_location", {}).get("source", {})
        if host_venue and host_venue.get("display_name"):
            venue_name = host_venue.get("display_name")

        results.append({
            "id": openalex_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "venue": venue_name,
            "url": doi,
            "source": "OpenAlex"
        })

    return results


def search_crossref(query_spec: dict, max_results: int = 20) -> list[dict]:
    """通过 Crossref API 检索文献"""
    raw_query = query_spec["api_keywords"]["crossref"]
    encoded_query = urllib.parse.quote(raw_query)
    url = f"https://api.crossref.org/works?query={encoded_query}&rows={max_results}&sort=score&order=desc"

    results = []
    data, status = http_get_json(url, timeout=TIMEOUT)
    if not data or "message" not in data:
        print(f"    [Crossref] Query failed: {status}")
        return results

    items = data.get("message", {}).get("items", [])
    for item in items:
        titles = item.get("title", [])
        title = titles[0] if titles else ""
        doi = item.get("DOI", "")
        url = item.get("URL", f"https://doi.org/{doi}" if doi else "")

        # 摘要处理（有时带 JATS XML 标签）
        raw_abstract = item.get("abstract", "")
        clean_abstract = re.sub(r"<[^>]+>", " ", raw_abstract).strip()

        # 出版年份
        year = 2024
        published = item.get("published") or item.get("issued")
        if published and "date-parts" in published and published["date-parts"]:
            dp = published["date-parts"][0]
            if dp and isinstance(dp[0], int):
                year = dp[0]

        authors = []
        for a in item.get("author", []):
            given = a.get("given", "").strip()
            family = a.get("family", "").strip()
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)

        venue_list = item.get("container-title", [])
        venue = venue_list[0] if venue_list else "Crossref Publication"

        results.append({
            "id": doi or url,
            "title": title,
            "abstract": clean_abstract,
            "authors": authors,
            "year": year,
            "venue": venue,
            "url": url,
            "source": "Crossref"
        })

    return results


def search_semantic_scholar(query_spec: dict, max_results: int = 15) -> list[dict]:
    """通过 Semantic Scholar API 检索文献（具备 429 降级防护与短延迟）"""
    raw_query = query_spec["api_keywords"]["semanticscholar"]
    encoded_query = urllib.parse.quote(raw_query)
    fields = "title,abstract,authors,year,venue,citationCount,externalIds,url"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit={max_results}&fields={fields}"

    # 预留间隔避免打满公共速率限制
    time.sleep(1.2)

    results = []
    data, status = http_get_json(url, timeout=TIMEOUT)
    if not data or "data" not in data:
        print(f"    [Semantic Scholar] Query failed or rate-limited: {status}")
        return results

    for item in data.get("data", []):
        title = item.get("title") or ""
        abstract = item.get("abstract") or ""
        year = item.get("year") or 2024
        paper_id = item.get("paperId") or ""
        ext_ids = item.get("externalIds") or {}
        paper_url = item.get("url") or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else "")

        authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
        venue = item.get("venue") or "Semantic Scholar Entry"

        results.append({
            "id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "venue": venue,
            "url": paper_url,
            "source": "Semantic Scholar"
        })

    return results


# --------------------------------------------------------------------------
# 主查重与聚合管线
# --------------------------------------------------------------------------
def run_novelty_bool_search():
    print("=" * 80)
    print(" [U-GraphRAG M3 & Full Pipeline] Academic Novelty Boolean Search Engine")
    print(f" Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. 加载本地 ACL 2024-2026 全量论文
    acl_papers = load_local_acl_database()

    output_report = {
        "metadata": {
            "task": "U-GraphRAG M3 & Full Pipeline Cross-Platform Academic Novelty Search",
            "execution_time": datetime.now().isoformat(),
            "local_acl_corpus_size": len(acl_papers),
            "platforms_audited": ["Local_ACL_Anthology", "arXiv", "OpenAlex", "Crossref", "Semantic_Scholar"],
            "query_count": len(QUERIES)
        },
        "summary": {},
        "query_details": {}
    }

    # 2. 逐一遍历 6 个布尔检索式
    for q in QUERIES:
        qid = q["id"]
        qname = q["name"]
        qexpr = q["expression"]
        print(f"\n--------------------------------------------------------------------------------")
        print(f"[*] Processing {qid} [{q['level']}]: {qname}")
        print(f"    Expression: {qexpr}")
        print(f"    Desc: {q['description']}")
        print(f"--------------------------------------------------------------------------------")

        query_record = {
            "id": qid,
            "name": qname,
            "level": q["level"],
            "expression": qexpr,
            "description": q["description"],
            "platforms": {}
        }

        # --- A. 本地 ACL 全库匹配 ---
        print(f"  [1/5] Checking Local ACL Anthology (total {len(acl_papers)} papers)...")
        acl_hits = []
        for p in acl_papers:
            matched, details = evaluate_query_match(qid, p["title"], p["abstract"])
            if matched:
                hit_item = {
                    "source": "Local_ACL_Anthology",
                    "id": p["id"],
                    "title": p["title"],
                    "authors": p["authors"][:5],
                    "year": p["year"],
                    "venue": p["venue"],
                    "url": p["url"],
                    "abstract_snippet": p["abstract"][:260] + ("..." if len(p["abstract"]) > 260 else ""),
                    "strict_match": True,
                    "matched_criteria": details
                }
                acl_hits.append(hit_item)

        print(f"  -> Local ACL Anthology strict hits: {len(acl_hits)}")
        query_record["platforms"]["Local_ACL_Anthology"] = {
            "searched_count": len(acl_papers),
            "matched_count": len(acl_hits),
            "hits": acl_hits
        }

        # --- B. arXiv API 检索与布尔校验 ---
        print(f"  [2/5] Querying arXiv API...")
        arxiv_candidates = search_arxiv(q, max_results=25)
        arxiv_hits = []
        for p in arxiv_candidates:
            matched, details = evaluate_query_match(qid, p["title"], p["abstract"])
            if matched:
                arxiv_hits.append({
                    "source": "arXiv",
                    "id": p["id"],
                    "title": p["title"],
                    "authors": p["authors"][:5],
                    "year": p["year"],
                    "venue": p["venue"],
                    "url": p["url"],
                    "abstract_snippet": p["abstract"][:260] + ("..." if len(p["abstract"]) > 260 else ""),
                    "strict_match": True,
                    "matched_criteria": details
                })
        print(f"  -> arXiv returned {len(arxiv_candidates)} candidates | Strict boolean hits: {len(arxiv_hits)}")
        query_record["platforms"]["arXiv"] = {
            "candidate_count": len(arxiv_candidates),
            "matched_count": len(arxiv_hits),
            "hits": arxiv_hits
        }

        # --- C. OpenAlex API 检索与布尔校验 ---
        print(f"  [3/5] Querying OpenAlex API...")
        openalex_candidates = search_openalex(q, max_results=25)
        openalex_hits = []
        for p in openalex_candidates:
            matched, details = evaluate_query_match(qid, p["title"], p["abstract"])
            if matched:
                openalex_hits.append({
                    "source": "OpenAlex",
                    "id": p["id"],
                    "title": p["title"],
                    "authors": p["authors"][:5],
                    "year": p["year"],
                    "venue": p["venue"],
                    "url": p["url"],
                    "abstract_snippet": p["abstract"][:260] + ("..." if len(p["abstract"]) > 260 else ""),
                    "strict_match": True,
                    "matched_criteria": details
                })
        print(f"  -> OpenAlex returned {len(openalex_candidates)} candidates | Strict boolean hits: {len(openalex_hits)}")
        query_record["platforms"]["OpenAlex"] = {
            "candidate_count": len(openalex_candidates),
            "matched_count": len(openalex_hits),
            "hits": openalex_hits
        }

        # --- D. Crossref API 检索与布尔校验 ---
        print(f"  [4/5] Querying Crossref API...")
        crossref_candidates = search_crossref(q, max_results=20)
        crossref_hits = []
        for p in crossref_candidates:
            matched, details = evaluate_query_match(qid, p["title"], p["abstract"])
            if matched:
                crossref_hits.append({
                    "source": "Crossref",
                    "id": p["id"],
                    "title": p["title"],
                    "authors": p["authors"][:5],
                    "year": p["year"],
                    "venue": p["venue"],
                    "url": p["url"],
                    "abstract_snippet": p["abstract"][:260] + ("..." if len(p["abstract"]) > 260 else ""),
                    "strict_match": True,
                    "matched_criteria": details
                })
        print(f"  -> Crossref returned {len(crossref_candidates)} candidates | Strict boolean hits: {len(crossref_hits)}")
        query_record["platforms"]["Crossref"] = {
            "candidate_count": len(crossref_candidates),
            "matched_count": len(crossref_hits),
            "hits": crossref_hits
        }

        # --- E. Semantic Scholar API 检索与布尔校验 ---
        print(f"  [5/5] Querying Semantic Scholar API...")
        ss_candidates = search_semantic_scholar(q, max_results=15)
        ss_hits = []
        for p in ss_candidates:
            matched, details = evaluate_query_match(qid, p["title"], p["abstract"])
            if matched:
                ss_hits.append({
                    "source": "Semantic_Scholar",
                    "id": p["id"],
                    "title": p["title"],
                    "authors": p["authors"][:5],
                    "year": p["year"],
                    "venue": p["venue"],
                    "url": p["url"],
                    "abstract_snippet": p["abstract"][:260] + ("..." if len(p["abstract"]) > 260 else ""),
                    "strict_match": True,
                    "matched_criteria": details
                })
        print(f"  -> Semantic Scholar returned {len(ss_candidates)} candidates | Strict boolean hits: {len(ss_hits)}")
        query_record["platforms"]["Semantic_Scholar"] = {
            "candidate_count": len(ss_candidates),
            "matched_count": len(ss_hits),
            "hits": ss_hits
        }

        # 聚合去重该检索式在所有平台的命中论文
        all_query_hits = []
        seen_titles = set()
        for plat_name, plat_data in query_record["platforms"].items():
            for h in plat_data.get("hits", []):
                norm_title = re.sub(r"\W+", "", h["title"].lower())
                if norm_title and norm_title not in seen_titles:
                    seen_titles.add(norm_title)
                    all_query_hits.append(h)

        query_record["deduplicated_strict_hits_count"] = len(all_query_hits)
        query_record["deduplicated_hits"] = all_query_hits

        output_report["query_details"][qid] = query_record

        # 汇总表项
        output_report["summary"][qid] = {
            "name": qname,
            "level": q["level"],
            "boolean_expression": qexpr,
            "total_deduplicated_strict_hits": len(all_query_hits),
            "platform_breakdown": {
                "Local_ACL_Anthology": len(acl_hits),
                "arXiv": len(arxiv_hits),
                "OpenAlex": len(openalex_hits),
                "Crossref": len(crossref_hits),
                "Semantic_Scholar": len(ss_hits)
            }
        }

        print(f"[Summary {qid}] Deduplicated Strict Hits: {len(all_query_hits)}")
        if all_query_hits:
            for idx, h in enumerate(all_query_hits[:3], 1):
                print(f"   ({idx}) [{h['source']} | {h['year']}] {h['title']}")

    # 3. 保存落盘为 JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Search logs successfully written to: {OUTPUT_JSON_PATH}")
    print(f"File Size: {os.path.getsize(OUTPUT_JSON_PATH):,} bytes")
    print("=" * 80)

    # 4. 打印整体汇总表
    print(f"\n{'ID':<5} | {'Level':<30} | {'ACL':<5} | {'arXiv':<6} | {'OpenAlex':<8} | {'Crossref':<8} | {'S2':<4} | {'Total'}")
    print("-" * 80)
    for qid, s in output_report["summary"].items():
        pb = s["platform_breakdown"]
        print(f"{qid:<5} | {s['level']:<30} | {pb['Local_ACL_Anthology']:<5} | {pb['arXiv']:<6} | {pb['OpenAlex']:<8} | {pb['Crossref']:<8} | {pb['Semantic_Scholar']:<4} | {s['total_deduplicated_strict_hits']}")
    print("-" * 80)


if __name__ == "__main__":
    run_novelty_bool_search()
