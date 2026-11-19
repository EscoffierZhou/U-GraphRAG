#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_cabinet_index.py
-----------------------------------------------------------------------------
MemPalace ACL: 落地 Wing_Room_Drawer 柜子序号系统 (cabinet_id & cabinet_code)
功能：
1. 幂等新增 cabinet_id 与 cabinet_code 两列于 palace_drawers；
2. 基于严格映射字典更新全量 palace_drawers 数据；
3. 为 cabinet_id / cabinet_code 建立普通索引；
4. 重建 palace_fts (FTS5) 虚拟表，纳入 cabinet_id 与 cabinet_code；
5. 从数据库直读并导出 mempalace_acl/CABINET_QUICK_REFERENCE.md；
6. 严格执行自校验 SQL。
-----------------------------------------------------------------------------
"""

import sys
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Set

# 工作区根目录与路径
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
MEMPALACE_DIR = WORKSPACE_DIR / "mempalace_acl"
DB_PATH = MEMPALACE_DIR / "index.sqlite"
QUICK_REF_PATH = MEMPALACE_DIR / "CABINET_QUICK_REFERENCE.md"

# 严格编码映射字典 (Strict Coding Dictionaries)
WING_CODE_MAP: Dict[str, str] = {
    "FactVerification_Reasoning": "W1",
    "Financial_Domain_NLP": "W2",
    "RAG_Hallucination_Mitigation": "W3",
    "Efficient_Inference_Algorithms": "W4",
}

ROOM_CODE_MAP: Dict[str, str] = {
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

DRAWER_CODE_MAP: Dict[str, str] = {
    "Method_Atoms": "D1",
    "Data_Pipelines": "D2",
    "Limitations_PainPoints": "D3",
    "Datasets_Baselines": "D4",
    "Empirical_Gains": "D5",
}

CORE_5_PAPERS = [
    "2025.acl-long.1549",
    "2025.emnlp-main.1724",
    "2024.acl-long.835",
    "2025.emnlp-main.764",
    "2024.emnlp-main.740",
]


def check_and_update_db(conn: sqlite3.Connection):
    cursor = conn.cursor()

    print("=== Step 1: Pre-inspection of SQLite Tables ===")
    cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Existing tables/views: {tables}")

    for t in tables:
        if t.startswith("palace_fts_") or "shadow" in t:
            continue
        try:
            cursor.execute(f"SELECT count(*) FROM {t}")
            cnt = cursor.fetchone()[0]
            print(f"  - Table '{t}': {cnt} rows")
        except Exception as e:
            print(f"  - Table '{t}': unable to count ({e})")

    # 检查 palace_drawers 现有列
    cursor.execute("PRAGMA table_info(palace_drawers)")
    drawer_cols = [row[1] for row in cursor.fetchall()]
    print(f"palace_drawers existing columns: {drawer_cols}")

    # 检查是否有未识别的 wing, room, drawer_type
    cursor.execute("SELECT DISTINCT wing FROM palace_drawers")
    existing_wings = {row[0] for row in cursor.fetchall()}
    cursor.execute("SELECT DISTINCT room FROM palace_drawers")
    existing_rooms = {row[0] for row in cursor.fetchall()}
    cursor.execute("SELECT DISTINCT drawer_type FROM palace_drawers")
    existing_drawers = {row[0] for row in cursor.fetchall()}

    unknown_wings = existing_wings - set(WING_CODE_MAP.keys())
    unknown_rooms = existing_rooms - set(ROOM_CODE_MAP.keys())
    unknown_drawers = existing_drawers - set(DRAWER_CODE_MAP.keys())

    if unknown_wings:
        raise ValueError(f"CRITICAL: Unknown wings in database: {unknown_wings}")
    if unknown_rooms:
        raise ValueError(f"CRITICAL: Unknown rooms in database: {unknown_rooms}")
    if unknown_drawers:
        raise ValueError(f"CRITICAL: Unknown drawer types in database: {unknown_drawers}")

    print("Pre-inspection validation PASSED: all wings, rooms, drawer_types match dictionaries.")

    print("\n=== Step 2: Idempotent Column Addition on palace_drawers ===")
    if "cabinet_id" not in drawer_cols:
        print("Adding column cabinet_id TEXT...")
        cursor.execute("ALTER TABLE palace_drawers ADD COLUMN cabinet_id TEXT")
    else:
        print("Column cabinet_id already exists.")

    if "cabinet_code" not in drawer_cols:
        print("Adding column cabinet_code TEXT...")
        cursor.execute("ALTER TABLE palace_drawers ADD COLUMN cabinet_code TEXT")
    else:
        print("Column cabinet_code already exists.")

    print("\n=== Step 3: Populate cabinet_id and cabinet_code ===")
    cursor.execute("SELECT id, paper_id, wing, room, drawer_type FROM palace_drawers")
    rows = cursor.fetchall()
    print(f"Updating {len(rows)} drawer rows...")

    updates = []
    for row_id, paper_id, wing, room, drawer_type in rows:
        w_code = WING_CODE_MAP.get(wing)
        r_code = ROOM_CODE_MAP.get(room)
        d_code = DRAWER_CODE_MAP.get(drawer_type)

        if not (w_code and r_code and d_code):
            raise ValueError(f"Cannot map row id={row_id}, paper_id={paper_id}, wing={wing}, room={room}, drawer={drawer_type}")

        cabinet_id = f"{wing}_{room}_{drawer_type}#{paper_id}"
        cabinet_code = f"{w_code}-{r_code}-{d_code}#{paper_id}"
        updates.append((cabinet_id, cabinet_code, row_id))

    cursor.executemany(
        "UPDATE palace_drawers SET cabinet_id = ?, cabinet_code = ? WHERE id = ?",
        updates
    )
    print(f"Updated {len(updates)} rows in palace_drawers successfully.")

    # 建立普通索引
    print("\n=== Step 4: Create Indexes for cabinet_id and cabinet_code ===")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_drawers_cabinet_id ON palace_drawers(cabinet_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_drawers_cabinet_code ON palace_drawers(cabinet_code)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_drawers_paper_id ON palace_drawers(paper_id)")
    print("Indexes created/verified.")

    # 重建 FTS5 虚拟表
    print("\n=== Step 5: Rebuild palace_fts (FTS5) Table ===")
    cursor.execute("DROP TABLE IF EXISTS palace_fts")
    cursor.execute("""
    CREATE VIRTUAL TABLE palace_fts USING fts5(
        cabinet_id,
        cabinet_code,
        paper_id,
        wing,
        room,
        drawer_type,
        title,
        content,
        tags,
        tokenize='porter unicode61'
    )
    """)

    cursor.execute("""
    INSERT INTO palace_fts (
        cabinet_id, cabinet_code, paper_id, wing, room, drawer_type, title, content, tags
    )
    SELECT
        cabinet_id, cabinet_code, paper_id, wing, room, drawer_type, title, content, tags
    FROM palace_drawers
    """)
    conn.commit()
    print("palace_fts virtual table rebuilt and populated successfully.")


def run_validations(conn: sqlite3.Connection):
    cursor = conn.cursor()
    print("\n=== Step 6: Database Integrity Validations ===")

    # 1. 空 cabinet_id / cabinet_code 检查
    cursor.execute("SELECT count(*) FROM palace_drawers WHERE cabinet_id IS NULL OR cabinet_id = ''")
    null_cid = cursor.fetchone()[0]
    cursor.execute("SELECT count(*) FROM palace_drawers WHERE cabinet_code IS NULL OR cabinet_code = ''")
    null_ccode = cursor.fetchone()[0]

    # 2. 行数对齐检查
    cursor.execute("SELECT count(*) FROM palace_drawers")
    drawers_cnt = cursor.fetchone()[0]
    cursor.execute("SELECT count(*) FROM palace_fts")
    fts_cnt = cursor.fetchone()[0]

    # 3. 检查 methodology_gains 与 papers 表行数
    cursor.execute("SELECT count(*) FROM methodology_gains")
    mg_cnt = cursor.fetchone()[0]
    cursor.execute("SELECT count(*) FROM papers")
    papers_cnt = cursor.fetchone()[0]

    print(f"Validation Result:")
    print(f"  - NULL/empty cabinet_id: {null_cid} (Expected: 0)")
    print(f"  - NULL/empty cabinet_code: {null_ccode} (Expected: 0)")
    print(f"  - palace_drawers count: {drawers_cnt}")
    print(f"  - palace_fts count: {fts_cnt}")
    print(f"  - methodology_gains count: {mg_cnt}")
    print(f"  - papers count: {papers_cnt}")

    if null_cid != 0 or null_ccode != 0:
        raise AssertionError(f"Validation FAILED: Found empty cabinet identifiers! (cid: {null_cid}, ccode: {null_ccode})")
    if drawers_cnt != fts_cnt:
        raise AssertionError(f"Validation FAILED: palace_drawers ({drawers_cnt}) != palace_fts ({fts_cnt})!")

    print("ALL VALIDATION CHECKS PASSED PERFECTLY!\n")
    return {
        "drawers_cnt": drawers_cnt,
        "fts_cnt": fts_cnt,
        "mg_cnt": mg_cnt,
        "papers_cnt": papers_cnt,
    }


def generate_cabinet_quick_reference(conn: sqlite3.Connection):
    cursor = conn.cursor()
    print("=== Step 7: Generating MemPalace Cabinet Quick Reference ===")

    # 1. 编码映射与全局统计
    cursor.execute("SELECT wing, count(*) FROM palace_drawers GROUP BY wing ORDER BY wing")
    wing_counts = cursor.fetchall()

    cursor.execute("SELECT wing, room, count(*) FROM palace_drawers GROUP BY wing, room ORDER BY wing, room")
    room_counts = cursor.fetchall()

    cursor.execute("SELECT drawer_type, count(*) FROM palace_drawers GROUP BY drawer_type ORDER BY drawer_type")
    drawer_counts = cursor.fetchall()

    cursor.execute("SELECT count(DISTINCT paper_id) FROM palace_drawers")
    unique_papers_cnt = cursor.fetchone()[0]

    cursor.execute("SELECT count(*) FROM palace_drawers")
    total_drawers_cnt = cursor.fetchone()[0]

    lines = []
    lines.append("# MemPalace ACL: 柜子序号系统全量速查表 (Cabinet Quick Reference)")
    lines.append("")
    lines.append("> **文档定位**：MemPalace 物理记忆宫殿柜子（Cabinet）及抽屉（Drawer）双轨寻址全景速查指南。  ")
    lines.append(f"> **生成方式**：由 `scripts/build_cabinet_index.py` 从 `mempalace_acl/index.sqlite` 物理直读导出。  ")
    lines.append(f"> **总收录资产**：**{unique_papers_cnt}** 篇 8GB 显存纯算法顶会论文，共计 **{total_drawers_cnt}** 个高价值知识抽屉。  ")
    lines.append("")

    # 章节 1：编码规范
    lines.append("## 1. 柜子编码规范与映射字典 (Cabinet Schema & Code Mapping)")
    lines.append("")
    lines.append("每个知识抽屉（Drawer）配备双重寻址坐标：")
    lines.append("1. **长序号 (Human-Readable `cabinet_id`)**：`{wing}_{room}_{drawer_type}#{paper_id}`")
    lines.append("2. **短码 (Dense-Coded `cabinet_code`)**：`{W_code}-{R_code}-{D_code}#{paper_id}`")
    lines.append("")
    lines.append("### 1.1 翼展 (Wing) 编码映射")
    lines.append("| 翼展英文名 (Wing) | 翼展短码 | 抽屉总数 | 核心定位与研究范畴 |")
    lines.append("| :--- | :---: | :---: | :--- |")
    wing_desc = {
        "FactVerification_Reasoning": "事实核验、因果推理、逻辑自洽与对抗辩论",
        "Financial_Domain_NLP": "金融研报数值抽取、合规审计、财报事实溯源与异构因果",
        "RAG_Hallucination_Mitigation": "知识库检索增强、幻觉探测、多跳重排与校准",
        "Efficient_Inference_Algorithms": "免训练启发式、门控稀疏激活、推理解码压缩与轻量适配",
    }
    for wing, cnt in wing_counts:
        lines.append(f"| `{wing}` | **`{WING_CODE_MAP.get(wing, 'UNK')}`** | {cnt} | {wing_desc.get(wing, '')} |")
    lines.append("")

    lines.append("### 1.2 展室 (Room) 编码映射")
    lines.append("| 展室英文名 (Room) | 所属翼展 | 展室短码 | 抽屉数量 | 研究主题 |")
    lines.append("| :--- | :--- | :---: | :---: | :--- |")
    room_desc = {
        "Fact_Checking_Audit": "事实真伪审计与细粒度核验",
        "MultiAgent_Debate": "多智能体分角色对抗与辩论质询",
        "MultiHop_Logical_Reasoning": "多跳逻辑链、因果树抽取与因果事实核验",
        "Financial_Report_Analysis": "金融财报、表格理解与数值推理",
        "Evidence_Retrieval_Rerank": "多跳证据检索、重排与上下文修剪",
        "Hallucination_Detection_Calibration": "幻觉量化探测、置信度校准与真实性对齐",
        "Knowledge_Graph_Augmentation": "知识图谱与拓扑先验增强",
        "Lightweight_Model_Adaptation": "轻量级小模型/SLM 适配与蒸馏",
        "Prompt_Workflow_Engineering": "提示词工程、结构化工作流与模式编排",
        "Training_Free_Heuristics": "免训练动态剪枝、门控感知与早退解码",
    }
    for wing, room, cnt in room_counts:
        lines.append(f"| `{room}` | `{wing}` | **`{ROOM_CODE_MAP.get(room, 'UNK')}`** | {cnt} | {room_desc.get(room, '')} |")
    lines.append("")

    lines.append("### 1.3 抽屉类型 (Drawer Type) 编码映射")
    lines.append("| 抽屉类型 (Drawer Type) | 抽屉短码 | 抽屉数量 | 内容定位 |")
    lines.append("| :--- | :---: | :---: | :--- |")
    drawer_desc = {
        "Method_Atoms": "核心算法架构、数学定义、门控公式与机制创新原子",
        "Data_Pipelines": "数据流处理、合成逻辑、提示词模板与前后处理流水线",
        "Limitations_PainPoints": "论文作者承认的真实短板、算力消耗与边界脆弱性（破局切入点）",
        "Datasets_Baselines": "标准基准数据集、对比 Baseline 体系与评价指标",
        "Empirical_Gains": "物理双栏 PDF 提取的实证量化涨点数据、消融降幅与原文见证证据",
    }
    for d_type, cnt in drawer_counts:
        lines.append(f"| `{d_type}` | **`{DRAWER_CODE_MAP.get(d_type, 'UNK')}`** | {cnt} | {drawer_desc.get(d_type, '')} |")
    lines.append("")

    # 章节 2：Core-5 王牌论文核心速查表
    lines.append("## 2. Core-5 王牌论文 25 个核心柜子速查表 (Core-5 Cabinet Matrix)")
    lines.append("")
    lines.append("五篇核心王牌论文深度承载了 **因果抽取 + 辩论对质 + 门控动态决策 + 稀疏早退剪枝** 的算法闭环。以下从数据库实时拉取其 25 个抽屉明细：")
    lines.append("")

    # 获取 Core-5 论文信息
    for pid in CORE_5_PAPERS:
        cursor.execute("SELECT title, wing, room FROM palace_drawers WHERE paper_id = ? LIMIT 1", (pid,))
        p_info = cursor.fetchone()
        if not p_info:
            continue
        p_title, p_wing, p_room = p_info

        lines.append(f"### 📄 `{pid}`: {p_title}")
        lines.append(f"- **MemPalace 坐标**：`{p_wing}` -> `{p_room}` (短码: `{WING_CODE_MAP.get(p_wing)}-{ROOM_CODE_MAP.get(p_room)}`)")
        lines.append("")
        lines.append("| 抽屉短码 | 抽屉类型 | 完整长序号 (cabinet_id) | 完整短码 (cabinet_code) | 抽屉概要 (Excerpt) |")
        lines.append("| :---: | :--- | :--- | :--- | :--- |")

        # 抽屉按 D1 到 D5 排序
        cursor.execute("""
            SELECT drawer_type, cabinet_id, cabinet_code, content
            FROM palace_drawers
            WHERE paper_id = ?
            ORDER BY drawer_type
        """, (pid,))
        drawers = cursor.fetchall()

        # 按 D1-D5 排序
        d_order = {"Method_Atoms": 1, "Data_Pipelines": 2, "Limitations_PainPoints": 3, "Datasets_Baselines": 4, "Empirical_Gains": 5}
        drawers.sort(key=lambda x: d_order.get(x[0], 99))

        for dtype, cid, ccode, content in drawers:
            # 清洗 content 用于表格展示 (前 80 字)
            summary_snippet = content.replace("\n", " ").strip()
            if len(summary_snippet) > 85:
                summary_snippet = summary_snippet[:82] + "..."
            summary_snippet = summary_snippet.replace("|", "\\|")
            lines.append(f"| `{DRAWER_CODE_MAP.get(dtype)}` | `{dtype}` | `{cid}` | `{ccode}` | {summary_snippet} |")
        lines.append("")

    # 章节 3：全部 8GB 可行论文总索引表
    lines.append("## 3. 全量 8GB 可行论文柜子总索引 (Complete 8GB Feasible Cabinet Catalog)")
    lines.append("")
    lines.append("按 `Wing -> Room -> Paper_ID` 排序，呈现所有受收录论文的基础前缀、短码前缀及其实际拥有的抽屉清单：")
    lines.append("")
    lines.append("| 序号 | 翼展短码 | 展室短码 | 论文 ID (paper_id) | 简短论文标题 | 基础长序号前缀 | 基础短码前缀 | 拥有抽屉列表 | 实证涨点抽屉 |")
    lines.append("| :---: | :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: |")

    cursor.execute("""
        SELECT DISTINCT paper_id, wing, room, title
        FROM palace_drawers
        ORDER BY wing, room, paper_id
    """)
    paper_rows = cursor.fetchall()

    for idx, (pid, wing, room, title) in enumerate(paper_rows, 1):
        w_code = WING_CODE_MAP.get(wing, "W?")
        r_code = ROOM_CODE_MAP.get(room, "R??")
        base_id_prefix = f"{wing}_{room}"
        base_code_prefix = f"{w_code}-{r_code}"

        # 查拥有的抽屉
        cursor.execute("SELECT drawer_type FROM palace_drawers WHERE paper_id = ?", (pid,))
        owned_dtypes = [r[0] for r in cursor.fetchall()]

        d_codes = [DRAWER_CODE_MAP.get(d, d) for d in sorted(owned_dtypes, key=lambda x: d_order.get(x, 99))]
        d_codes_str = ", ".join(d_codes)

        has_eg = "✅ D5 (Empirical_Gains)" if "Empirical_Gains" in owned_dtypes else "❌ 无"

        short_title = title.replace("\n", " ").strip()
        if len(short_title) > 45:
            short_title = short_title[:42] + "..."
        short_title = short_title.replace("|", "\\|")

        lines.append(f"| {idx} | `{w_code}` | `{r_code}` | `{pid}` | {short_title} | `{base_id_prefix}` | `{base_code_prefix}` | {d_codes_str} | {has_eg} |")

    lines.append("")
    lines.append("---")
    lines.append(f"*MemPalace ACL Cabinet Indexer | Generated on {Path(QUICK_REF_PATH).name}*  ")

    content_str = "\n".join(lines) + "\n"
    QUICK_REF_PATH.write_text(content_str, encoding="utf-8")
    print(f"Cabinet Quick Reference successfully written to: {QUICK_REF_PATH} ({len(content_str)} bytes)")


def main():
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        check_and_update_db(conn)
        val_res = run_validations(conn)
        generate_cabinet_quick_reference(conn)
        print("\n=== Script Execution Finished Successfully! ===")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
