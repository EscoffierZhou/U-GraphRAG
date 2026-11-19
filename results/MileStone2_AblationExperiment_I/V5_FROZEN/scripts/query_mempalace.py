#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/query_mempalace.py
-----------------------------------------------------------------------------
MemPalace ACL: 记忆宫殿纯离线只读查询与透视引擎
功能特性：
1. cabinet 模式: 柜子长序号(cabinet_id)与短码(cabinet_code)精确/模糊寻址并展示抽屉原文。
2. fts 模式: 基于 SQLite FTS5 全文索引对抽屉内容进行高效检索与片段高亮(snippet)。
3. painpoints 模式: 聚焦 Limitations_PainPoints (D3) 抽屉，检索顶会痛点与论文创新切入点。
4. gains 模式: 透视 methodology_gains 实证涨点表，支持最小涨幅与赛道筛选。
5. room 模式: 列出指定房间(Room)与翼(Wing)内所有论文及拥有的 5 个抽屉柜号清单。
6. paper 模式: 深度透视单篇论文的元数据、5 个抽屉内容以及实证涨幅记录。
7. 支持 Windows UTF-8 中文终端输出与 --export-md Markdown 导出。

用法示例：
  python scripts/query_mempalace.py --mode cabinet --q W1-R03-D1#2025.acl-long.1549
  python scripts/query_mempalace.py --mode paper --q 2025.emnlp-main.1724
  python scripts/query_mempalace.py --mode painpoints --q "omission OR single table OR debate OR calibration" --limit 8
  python scripts/query_mempalace.py --mode gains --limit 8
  python scripts/query_mempalace.py --mode fts --q "pandas OR counterfactual OR uncertainty" --limit 6
  python scripts/query_mempalace.py --mode room --room R03
-----------------------------------------------------------------------------
"""

import argparse
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

# 确保 Windows 控制台与管道使用 UTF-8 编码
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# 映射字典 (与 scripts/build_cabinet_index.py 保持完全一致)
WING_MAP = {
    "FactVerification_Reasoning": "W1",
    "Financial_Domain_NLP": "W2",
    "RAG_Hallucination_Mitigation": "W3",
    "Efficient_Inference_Algorithms": "W4",
}
REV_WING_MAP = {v: k for k, v in WING_MAP.items()}

ROOM_MAP = {
    "Fact_Checking_Audit": "R01",
    "MultiAgent_Debate": "R02",
    "MultiHop_Logical_Reasoning": "R03",
    "Financial_Report_Analysis": "R04",
    "Evidence_Retrieval_Rerank": "R05",
    "Hallucination_Detection_Calibration": "R06",
    "Knowledge_Graph_Augmentation": "R07",
    "Lightweight_Model_Adaptation": "R08",
    "Prompt_Workflow_Engineering": "R09",
    "Training_Free_Heuristics": "R10",
}
REV_ROOM_MAP = {v: k for k, v in ROOM_MAP.items()}

DRAWER_MAP = {
    "Method_Atoms": "D1",
    "Data_Pipelines": "D2",
    "Limitations_PainPoints": "D3",
    "Datasets_Baselines": "D4",
    "Empirical_Gains": "D5",
}
REV_DRAWER_MAP = {v: k for k, v in DRAWER_MAP.items()}


def get_readonly_connection(db_path: str) -> sqlite3.Connection:
    """以只读 URI 模式打开 SQLite 数据库，避免意外修改"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    abs_path = os.path.abspath(db_path).replace("\\", "/")
    uri = f"file:{abs_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_wing(wing_input: Optional[str]) -> Optional[str]:
    """将 Wing 短码或长名统一标准化为完整长名"""
    if not wing_input:
        return None
    val = wing_input.strip()
    if val in REV_WING_MAP:
        return REV_WING_MAP[val]
    for full_name in WING_MAP:
        if full_name.lower() == val.lower():
            return full_name
    return val


def normalize_room(room_input: Optional[str]) -> Optional[str]:
    """将 Room 短码或长名统一标准化为完整长名"""
    if not room_input:
        return None
    val = room_input.strip()
    if val in REV_ROOM_MAP:
        return REV_ROOM_MAP[val]
    for full_name in ROOM_MAP:
        if full_name.lower() == val.lower():
            return full_name
    return val


def normalize_drawer(drawer_input: Optional[str]) -> Optional[str]:
    """将 Drawer 短码或长名统一标准化为完整长名"""
    if not drawer_input:
        return None
    val = drawer_input.strip()
    if val in REV_DRAWER_MAP:
        return REV_DRAWER_MAP[val]
    for full_name in DRAWER_MAP:
        if full_name.lower() == val.lower():
            return full_name
    return val


def format_card(title: str, fields: List[Tuple[str, Any]], body: Optional[str] = None) -> str:
    """输出美观的 Markdown 风格卡片"""
    lines = [f"### 📦 {title}", ""]
    for k, v in fields:
        lines.append(f"- **{k}**: {v}")
    if body:
        lines.append("")
        lines.append("```text")
        lines.append(body.strip())
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 查询功能实现
# -----------------------------------------------------------------------------

def query_cabinet(conn: sqlite3.Connection, query_term: str) -> Tuple[str, int]:
    """
    按 cabinet_code 或 cabinet_id 查询具体柜子/抽屉原文
    支持精准匹配、短码前缀模糊匹配或长序号模糊匹配
    """
    cur = conn.cursor()
    q = query_term.strip()

    # 1. 尝试完全匹配
    sql_exact = """
    SELECT id, paper_id, wing, room, drawer_type, title, content,
           is_gating_based, is_8gb_feasible, tags, cabinet_id, cabinet_code
    FROM palace_drawers
    WHERE cabinet_code = ? OR cabinet_id = ?
    """
    cur.execute(sql_exact, (q, q))
    rows = cur.fetchall()

    # 2. 若无结果，尝试前缀匹配与包含匹配
    if not rows:
        sql_like = """
        SELECT id, paper_id, wing, room, drawer_type, title, content,
               is_gating_based, is_8gb_feasible, tags, cabinet_id, cabinet_code
        FROM palace_drawers
        WHERE cabinet_code LIKE ? OR cabinet_id LIKE ? OR paper_id LIKE ?
        ORDER BY cabinet_code ASC
        LIMIT 20
        """
        cur.execute(sql_like, (f"{q}%", f"%{q}%", f"%{q}%"))
        rows = cur.fetchall()

    out = []
    out.append(f"# 🏛️ MemPalace Cabinet 查询结果 (检索词: `{q}`)")
    out.append(f"> 找到匹配记录: {len(rows)} 条\n")

    if not rows:
        out.append(f"⚠️ 未在 palace_drawers 中检索到符合 `{q}` 的柜子记录。")
        return "\n".join(out), 0

    for r in rows:
        fields = [
            ("柜子短码 (Code)", f"`{r['cabinet_code']}`"),
            ("柜子长序号 (ID)", f"`{r['cabinet_id']}`"),
            ("论文标识 (Paper ID)", r['paper_id']),
            ("所属宫殿", f"{r['wing']} / {r['room']}"),
            ("抽屉类别", r['drawer_type']),
            ("标 题", r['title']),
            ("8GB显存可行性", "✅ 支持" if r['is_8gb_feasible'] else "❌ 需大显存"),
            ("门控机制属性", "✅ 门控相关" if r['is_gating_based'] else "➖ 无门控"),
            ("标签 (Tags)", r['tags'] or "无"),
        ]
        out.append(format_card(r['title'], fields, body=r['content']))

    return "\n".join(out), len(rows)


def query_fts(conn: sqlite3.Connection, query_term: str, drawer_filter: Optional[str] = None, limit: int = 10) -> Tuple[str, int]:
    """
    基于 SQLite FTS5 全文虚拟表 palace_fts 进行全文检索与 snippet 高亮
    """
    cur = conn.cursor()
    q = query_term.strip()

    sql_parts = [
        "SELECT cabinet_code, cabinet_id, paper_id, wing, room, drawer_type, title,",
        "       snippet(palace_fts, 7, '【', '】', '...', 28) AS matched_snippet,",
        "       bm25(palace_fts) AS rank_score",
        "FROM palace_fts",
        "WHERE palace_fts MATCH ?"
    ]
    params: List[Any] = [q]

    if drawer_filter:
        norm_drawer = normalize_drawer(drawer_filter)
        sql_parts.append("AND drawer_type = ?")
        params.append(norm_drawer)

    sql_parts.append("ORDER BY rank_score ASC")
    sql_parts.append(f"LIMIT {limit}")

    sql = " ".join(sql_parts)

    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        # 若 FTS 语法解析报错，自动加上双引号转为精准词组查询
        safe_q = f'"{q.replace(chr(34), "")}"'
        params[0] = safe_q
        cur.execute(sql, params)
        rows = cur.fetchall()

    out = []
    out.append(f"# 🔍 MemPalace 全文检索 (FTS5) 结果")
    out.append(f"> 检索词: `{q}` | 抽屉过滤: `{drawer_filter or '全部'}` | 返回上限: {limit}")
    out.append(f"> 命中文档数: {len(rows)} 条\n")

    if not rows:
        out.append(f"⚠️ 全文检索未命中任何内容。请检查检索词拼写或尝试调整语法。")
        return "\n".join(out), 0

    for i, r in enumerate(rows, 1):
        fields = [
            ("序号 / 相关度得分", f"#{i} (BM25: {r['rank_score']:.3f})"),
            ("柜子短码", f"`{r['cabinet_code']}`"),
            ("论文标识", r['paper_id']),
            ("宫殿房间 / 抽屉", f"{r['wing']} / {r['room']} / {r['drawer_type']}"),
            ("论文标题", r['title']),
            ("匹配高亮片段", r['matched_snippet']),
        ]
        out.append(format_card(f"{r['paper_id']} - {r['drawer_type']}", fields))

    return "\n".join(out), len(rows)


def query_painpoints(conn: sqlite3.Connection, query_term: str, limit: int = 10) -> Tuple[str, int]:
    """
    专门在 Limitations_PainPoints 抽屉中检索顶会作者自述的痛点 (寻找创新研究切入点)
    """
    cur = conn.cursor()
    q = query_term.strip()

    # 优先采用 FTS5 进行 Limitations_PainPoints 检索
    fts_sql = """
    SELECT cabinet_code, cabinet_id, paper_id, wing, room, drawer_type, title,
           snippet(palace_fts, 7, '【', '】', '...', 35) AS matched_snippet,
           content, bm25(palace_fts) AS rank_score
    FROM palace_fts
    WHERE palace_fts MATCH ? AND drawer_type = 'Limitations_PainPoints'
    ORDER BY rank_score ASC
    LIMIT ?
    """

    rows = []
    try:
        cur.execute(fts_sql, (q, limit))
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        safe_q = f'"{q.replace(chr(34), "")}"'
        try:
            cur.execute(fts_sql, (safe_q, limit))
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            rows = []

    # 若 FTS 未查到，尝试从 palace_drawers 做 LIKE 回退
    if not rows:
        like_sql = """
        SELECT cabinet_code, cabinet_id, paper_id, wing, room, drawer_type, title,
               substr(content, 1, 300) AS matched_snippet, content, 0.0 AS rank_score
        FROM palace_drawers
        WHERE drawer_type = 'Limitations_PainPoints'
          AND (content LIKE ? OR title LIKE ? OR tags LIKE ?)
        LIMIT ?
        """
        cur.execute(like_sql, (f"%{q}%", f"%{q}%", f"%{q}%", limit))
        rows = cur.fetchall()

    out = []
    out.append(f"# ⚡ MemPalace 顶会痛点与局限性 (Limitations & PainPoints) 检索")
    out.append(f"> 检索目标: Limitations_PainPoints (D3 抽屉) | 关键词: `{q}` | 上限: {limit}")
    out.append(f"> 命中文献痛点: {len(rows)} 条\n")

    if not rows:
        out.append(f"⚠️ 在痛点抽屉中未发现匹配 `{q}` 的记录。")
        return "\n".join(out), 0

    for i, r in enumerate(rows, 1):
        fields = [
            ("序号", f"#{i}"),
            ("柜子短码", f"`{r['cabinet_code']}`"),
            ("论文标识", r['paper_id']),
            ("所属房间", f"{r['wing']} / {r['room']}"),
            ("论文标题", r['title']),
            ("关键痛点片段摘录", r['matched_snippet']),
        ]
        out.append(format_card(f"痛点切入点 #{i}: {r['paper_id']}", fields, body=r['content']))

    return "\n".join(out), len(rows)


def query_gains(conn: sqlite3.Connection, min_delta: float = 0.0, track_filter: Optional[str] = None, limit: int = 15) -> Tuple[str, int]:
    """
    从 methodology_gains 表按涨幅/赛道排序输出实证涨点与 PDF 页码证据句
    """
    cur = conn.cursor()
    sql_parts = [
        "SELECT id, paper_id, wing, room, category, methodology_name, benchmark,",
        "       metric_name, baseline_val, ours_val, delta_text, delta_numeric,",
        "       verbatim_evidence, page_hint",
        "FROM methodology_gains",
        "WHERE delta_numeric >= ?"
    ]
    params: List[Any] = [min_delta]

    if track_filter:
        sql_parts.append("AND (category LIKE ? OR wing LIKE ? OR room LIKE ?)")
        pat = f"%{track_filter.strip()}%"
        params.extend([pat, pat, pat])

    sql_parts.append("ORDER BY delta_numeric DESC, ours_val DESC")
    sql_parts.append(f"LIMIT {limit}")

    sql = " ".join(sql_parts)
    cur.execute(sql, params)
    rows = cur.fetchall()

    out = []
    out.append(f"# 📈 MemPalace 实证涨点天梯榜 (Methodology Gains)")
    out.append(f"> 最小涨幅: ≥{min_delta} | 赛道过滤: `{track_filter or '全部'}` | 显示上限: {limit}")
    out.append(f"> 命中涨点记录: {len(rows)} 条\n")

    if not rows:
        out.append(f"⚠️ 未找到符合条件的实证涨点记录。")
        return "\n".join(out), 0

    for i, r in enumerate(rows, 1):
        delta_str = f"+{r['delta_numeric']:.2f}" if r['delta_numeric'] > 0 else f"{r['delta_numeric']:.2f}"
        if r['delta_text'] and r['delta_text'].endswith("%"):
            delta_disp = f"{r['delta_text']} (数值: {delta_str})"
        else:
            delta_disp = f"{delta_str} ({r['delta_text'] or '实测涨点'})"

        baseline_disp = f"{r['baseline_val']:.2f}" if r['baseline_val'] is not None else "基准未明"
        ours_disp = f"{r['ours_val']:.2f}" if r['ours_val'] is not None else "未明"

        fields = [
            ("排名 / 涨幅", f"第 {i} 名 🚀 **{delta_disp}**"),
            ("论文标识", r['paper_id']),
            ("赛道 / 归属", f"【{r['category']}】 ({r['wing']} / {r['room']})"),
            ("方法名称", r['methodology_name']),
            ("评估基准 / 指标", f"{r['benchmark']} / {r['metric_name']}"),
            ("实验数据对比", f"Baseline: `{baseline_disp}` ➔ 本文方法: `{ours_disp}`"),
            ("PDF 证据出处", r['page_hint'] or "正文实验部分"),
            ("论文原文证据句", f'"{r["verbatim_evidence"]}"' if r["verbatim_evidence"] else "无直接原文引注"),
        ]
        out.append(format_card(f"TOP-{i} 涨点: {r['methodology_name']} ({r['paper_id']})", fields))

    return "\n".join(out), len(rows)


def query_room(conn: sqlite3.Connection, wing_param: Optional[str] = None, room_param: Optional[str] = None) -> Tuple[str, int]:
    """
    列出指定房间/翼内所有论文及各自拥有的柜子序号清单
    """
    cur = conn.cursor()
    norm_wing = normalize_wing(wing_param)
    norm_room = normalize_room(room_param)

    sql_parts = [
        "SELECT d.paper_id, d.wing, d.room, d.cabinet_code, d.cabinet_id, d.drawer_type, d.title,",
        "       p.venue, p.year, p.is_8gb_feasible",
        "FROM palace_drawers d",
        "LEFT JOIN papers p ON d.paper_id = p.paper_id",
        "WHERE 1=1"
    ]
    params: List[Any] = []

    if norm_wing:
        sql_parts.append("AND d.wing = ?")
        params.append(norm_wing)
    if norm_room:
        sql_parts.append("AND d.room = ?")
        params.append(norm_room)

    sql_parts.append("ORDER BY d.wing ASC, d.room ASC, d.paper_id ASC, d.drawer_type ASC")
    sql = " ".join(sql_parts)

    cur.execute(sql, params)
    rows = cur.fetchall()

    # 聚合按论文组织
    papers_dict: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        pid = r["paper_id"]
        if pid not in papers_dict:
            papers_dict[pid] = {
                "paper_id": pid,
                "title": r["title"],
                "wing": r["wing"],
                "room": r["room"],
                "venue": r["venue"],
                "year": r["year"],
                "is_8gb_feasible": r["is_8gb_feasible"],
                "drawers": []
            }
        papers_dict[pid]["drawers"].append({
            "cabinet_code": r["cabinet_code"],
            "cabinet_id": r["cabinet_id"],
            "drawer_type": r["drawer_type"]
        })

    out = []
    w_disp = norm_wing or "全部翼"
    r_disp = norm_room or "全部房间"
    out.append(f"# 🏛️ MemPalace 展厅房间文献与柜号清单 (Wing: `{w_disp}` | Room: `{r_disp}`)")
    out.append(f"> 聚合文献数: {len(papers_dict)} 篇 | 抽屉记录总数: {len(rows)} 条\n")

    if not papers_dict:
        out.append(f"⚠️ 指定翼/房间下未查询到文献与抽屉记录。")
        return "\n".join(out), 0

    for i, (pid, pinfo) in enumerate(papers_dict.items(), 1):
        drawer_bullets = []
        for d in pinfo["drawers"]:
            drawer_bullets.append(f"  - `{d['cabinet_code']}` ({d['drawer_type']})")
        drawer_text = "\n".join(drawer_bullets)

        fields = [
            ("序号", f"#{i}"),
            ("论文标识", f"`{pid}`"),
            ("发表信息", f"{pinfo['venue'] or 'ACL'} {pinfo['year'] or ''}"),
            ("论文题目", pinfo['title']),
            ("8GB显存适配", "✅ 适配 8GB" if pinfo['is_8gb_feasible'] else "❌ 需大算力"),
            ("归属展厅", f"{pinfo['wing']} / {pinfo['room']}"),
            ("拥有的 5 抽屉柜号", f"\n{drawer_text}"),
        ]
        out.append(format_card(f"{pid}: {pinfo['title']}", fields))

    return "\n".join(out), len(papers_dict)


def query_paper(conn: sqlite3.Connection, paper_id: str) -> Tuple[str, int]:
    """
    透视单篇论文的元数据、全部 5 个抽屉(含 cabinet_id, cabinet_code)与实证涨幅记录
    """
    cur = conn.cursor()
    pid = paper_id.strip()

    # 1. 论文基础元数据
    cur.execute("SELECT * FROM papers WHERE paper_id = ? OR paper_id LIKE ?", (pid, f"%{pid}%"))
    paper_rows = cur.fetchall()
    if not paper_rows:
        return f"# ❌ 未检索到论文 ID: `{pid}`\n请检查 paper_id 是否正确。", 0

    paper = paper_rows[0]
    real_pid = paper["paper_id"]

    # 2. 抽屉数据
    cur.execute("""
    SELECT cabinet_code, cabinet_id, drawer_type, title, content, is_gating_based, is_8gb_feasible, tags
    FROM palace_drawers
    WHERE paper_id = ?
    ORDER BY cabinet_code ASC
    """, (real_pid,))
    drawers = cur.fetchall()

    # 3. methodology_gains 实证涨点
    cur.execute("""
    SELECT methodology_name, benchmark, metric_name, baseline_val, ours_val, delta_text, delta_numeric, verbatim_evidence, page_hint
    FROM methodology_gains
    WHERE paper_id = ?
    ORDER BY delta_numeric DESC
    """, (real_pid,))
    gains = cur.fetchall()

    out = []
    out.append(f"# 🔬 MemPalace 论文全貌透视: `{real_pid}`")
    out.append("")

    # 论文基本信息
    fields = [
        ("论文标识", real_pid),
        ("论文题目", paper["title"] or "未知题目"),
        ("作者团队", paper["authors"] or "未知作者"),
        ("发表顶会 / 年份", f"{paper['venue'] or 'ACL/EMNLP'} {paper['year'] or ''}"),
        ("8GB GPU 可行性", "✅ 完全可行" if paper["is_8gb_feasible"] else "❌ 超出8GB显存限制"),
        ("门控机制聚焦", "✅ 门控相关" if paper["is_gating_focused"] else "➖ 无显式门控"),
        ("重型预训练依赖", "❌ 包含大模型全参训练" if paper["full_text_has_heavy_training"] else "✅ 免重型训练/轻量推理"),
        ("页数 / 字符数", f"{paper['page_count']} 页 / {paper['char_count']} 字符"),
    ]
    abstract_preview = paper["abstract"] if paper["abstract"] else "无摘要"
    out.append(format_card(f"论文元数据: {real_pid}", fields, body=f"【摘要 Preview】:\n{abstract_preview}"))

    # 实证涨点
    out.append(f"## 📊 实证涨幅与证据句 ({len(gains)} 条记录)")
    if gains:
        for g in gains:
            delta_str = f"+{g['delta_numeric']:.2f}" if g['delta_numeric'] > 0 else f"{g['delta_numeric']:.2f}"
            g_fields = [
                ("方法名称", g['methodology_name']),
                ("评测基准 / 指标", f"{g['benchmark']} / {g['metric_name']}"),
                ("基准 ➔ 本文", f"{g['baseline_val']} ➔ {g['ours_val']} (Δ: {g['delta_text'] or delta_str})"),
                ("证据出处", g['page_hint']),
                ("原文证据句", f'"{g["verbatim_evidence"]}"'),
            ]
            out.append(format_card(f"涨幅: {g['methodology_name']}", g_fields))
    else:
        out.append("> ➖ 本篇论文在 methodology_gains 表中暂无已提取的量化涨点数据。\n")

    # 5 个抽屉详情
    out.append(f"## 🗄️ 记忆宫殿 5 大抽屉原文 ({len(drawers)} 个抽屉)")
    if drawers:
        for d in drawers:
            d_fields = [
                ("柜子短码 (Code)", f"`{d['cabinet_code']}`"),
                ("柜子长序号 (ID)", f"`{d['cabinet_id']}`"),
                ("抽屉类型", d['drawer_type']),
                ("8GB可行性 / 门控标签", f"8GB可行: {bool(d['is_8gb_feasible'])} | 门控机制: {bool(d['is_gating_based'])}"),
                ("标签", d['tags'] or "无"),
            ]
            out.append(format_card(f"{d['drawer_type']} [{d['cabinet_code']}]", d_fields, body=d['content']))
    else:
        out.append("> ⚠️ 暂未查询到该论文在 palace_drawers 中的抽屉数据。\n")

    return "\n".join(out), 1


# -----------------------------------------------------------------------------
# 命令行入口
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MemPalace ACL: 记忆宫殿纯离线只读查询与透视引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的典型查询场景：
  1. 查询特定柜子原文:
     python scripts/query_mempalace.py --mode cabinet --q W1-R03-D1#2025.acl-long.1549
  2. 全文检索 (FTS5):
     python scripts/query_mempalace.py --mode fts --q "pandas OR counterfactual OR uncertainty" --limit 6
  3. 顶会痛点与切入点检索:
     python scripts/query_mempalace.py --mode painpoints --q "omission OR single table OR debate OR calibration" --limit 8
  4. 实证涨点天梯榜:
     python scripts/query_mempalace.py --mode gains --limit 8 --min-delta 0
  5. 展厅/房间文献与柜号清单:
     python scripts/query_mempalace.py --mode room --room R03
  6. 单篇论文全貌透视:
     python scripts/query_mempalace.py --mode paper --q 2025.emnlp-main.1724
        """
    )
    parser.add_argument(
        "--mode",
        choices=["cabinet", "fts", "painpoints", "gains", "room", "paper"],
        required=True,
        help="查询模式: cabinet(查柜子), fts(全文检索), painpoints(痛点库), gains(涨点榜), room(展厅房间), paper(论文透视)"
    )
    parser.add_argument("--q", type=str, default="", help="检索词、柜子编码、论文ID或查询表达式")
    parser.add_argument("--wing", type=str, default=None, help="指定 Wing 翼编码(如 W1)或长名称(如 FactVerification_Reasoning)")
    parser.add_argument("--room", type=str, default=None, help="指定 Room 房间编码(如 R03)或长名称(如 MultiHop_Logical_Reasoning)")
    parser.add_argument("--drawer", type=str, default=None, help="指定 Drawer 抽屉编码(如 D1)或类型(如 Method_Atoms)")
    parser.add_argument("--track", type=str, default=None, help="赛道名称过滤(用于 gains 模式)")
    parser.add_argument("--min-delta", type=float, default=0.0, help="最小涨幅过滤数值(用于 gains 模式，默认 0.0)")
    parser.add_argument("--limit", type=int, default=10, help="最大返回条数(默认 10)")
    parser.add_argument("--db", type=str, default="mempalace_acl/index.sqlite", help="SQLite 数据库文件路径(默认 mempalace_acl/index.sqlite)")
    parser.add_argument("--export-md", type=str, default=None, help="将查询结果保存至指定的 Markdown 文件路径")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    db_path = args.db
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.getcwd(), db_path)

    if not os.path.exists(db_path):
        print(f"❌ 错误: 数据库文件不存在: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = get_readonly_connection(db_path)

    try:
        if args.mode == "cabinet":
            if not args.q:
                print("❌ 错误: --mode cabinet 模式必须指定 --q 参数 (如 W1-R03-D1#2025.acl-long.1549)", file=sys.stderr)
                sys.exit(1)
            result_md, count = query_cabinet(conn, args.q)

        elif args.mode == "fts":
            if not args.q:
                print("❌ 错误: --mode fts 模式必须指定 --q 检索词", file=sys.stderr)
                sys.exit(1)
            result_md, count = query_fts(conn, args.q, drawer_filter=args.drawer, limit=args.limit)

        elif args.mode == "painpoints":
            if not args.q:
                print("❌ 错误: --mode painpoints 模式必须指定 --q 关键词", file=sys.stderr)
                sys.exit(1)
            result_md, count = query_painpoints(conn, args.q, limit=args.limit)

        elif args.mode == "gains":
            result_md, count = query_gains(conn, min_delta=args.min_delta, track_filter=args.track, limit=args.limit)

        elif args.mode == "room":
            result_md, count = query_room(conn, wing_param=args.wing, room_param=args.room)

        elif args.mode == "paper":
            if not args.q:
                print("❌ 错误: --mode paper 模式必须指定 --q <paper_id>", file=sys.stderr)
                sys.exit(1)
            result_md, count = query_paper(conn, args.q)

        else:
            print(f"❌ 未知查询模式: {args.mode}", file=sys.stderr)
            sys.exit(1)

        # 终端输出
        print(result_md)

        # 可选导出 Markdown 文件
        if args.export_md:
            export_path = os.path.abspath(args.export_md)
            export_dir = os.path.dirname(export_path)
            if export_dir and not os.path.exists(export_dir):
                os.makedirs(export_dir, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(result_md)
            print(f"\n💾 [导出成功] 查询结果已保存至: {export_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
