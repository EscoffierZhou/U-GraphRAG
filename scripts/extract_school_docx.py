# -*- coding: utf-8 -*-
"""
extract_school_docx.py
解析 "学校文件/" 下的 6 个关键 .docx 文件，提取所有文本和表格，
生成结构化的 Markdown 汇总文档：学校文件/PARSED_SCHOOL_TASKS.md。
使用 Python 标准库 zipfile 与 xml.etree.ElementTree，零外部依赖。
"""

import os
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

def extract_docx_content(docx_path):
    """
    解析单个 docx 文件中的段落和表格内容，并返回 Markdown 格式文本。
    """
    if not os.path.exists(docx_path):
        return f"【错误：文件未找到: {docx_path}】"

    try:
        with zipfile.ZipFile(docx_path, 'r') as zf:
            if 'word/document.xml' not in zf.namelist():
                return f"【错误：未在 {docx_path} 中找到 word/document.xml】"
            xml_data = zf.read('word/document.xml')
    except Exception as e:
        return f"【读取 docx 解包失败: {e}】"

    try:
        root = ET.fromstring(xml_data)
    except Exception as e:
        return f"【XML解析失败: {e}】"

    body = root.find(f"{W_NS}body")
    if body is None:
        return "【正文 body 为空】"

    md_lines = []
    
    for elem in body:
        tag = elem.tag
        if tag == f"{W_NS}p":
            # 段落解析
            texts = []
            for t in elem.iter(f"{W_NS}t"):
                if t.text:
                    texts.append(t.text)
            p_text = "".join(texts).strip()
            if p_text:
                md_lines.append(p_text)
            else:
                # 保留适当空行
                if md_lines and md_lines[-1] != "":
                    md_lines.append("")
        elif tag == f"{W_NS}tbl":
            # 表格解析
            table_rows = []
            max_cols = 0
            for tr in elem.findall(f"{W_NS}tr"):
                row_cells = []
                for tc in tr.findall(f"{W_NS}tc"):
                    cell_texts = []
                    for t in tc.iter(f"{W_NS}t"):
                        if t.text:
                            cell_texts.append(t.text)
                    cell_content = " ".join("".join(cell_texts).split())
                    row_cells.append(cell_content)
                if row_cells:
                    table_rows.append(row_cells)
                    if len(row_cells) > max_cols:
                        max_cols = len(row_cells)
            
            if table_rows:
                md_lines.append("")
                # 规范化列宽，输出 Markdown 表格
                for r_idx, row in enumerate(table_rows):
                    # 补齐不足的列
                    padded_row = row + [""] * (max_cols - len(row))
                    # 转义管道符
                    escaped_row = [c.replace("|", "\\|") for c in padded_row]
                    md_lines.append("| " + " | ".join(escaped_row) + " |")
                    if r_idx == 0:
                        md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
                md_lines.append("")
        else:
            # 其他元素
            continue

    # 去除首尾连续空行
    while md_lines and md_lines[0] == "":
        md_lines.pop(0)
    while md_lines and md_lines[-1] == "":
        md_lines.pop()

    return "\n\n".join([line for line in md_lines if line is not None])


def main():
    import sys
    if "--dump-hits" in sys.argv:
        import json
        with open("Debate/search_logs_m1_m2.json", "r", encoding="utf-8") as f:
            m1_m2 = json.load(f)
        with open("Debate/search_logs_m3_full.json", "r", encoding="utf-8") as f:
            m3_full = json.load(f)

        print("=== M1_M2 JSON SUMMARY ===")
        for q in m1_m2["query_results"]:
            print(f"[{q['query_id']}] {q['query_name']} -> total hits: {q['total_verified_hits']}")
            for p in q["matched_papers"]:
                print(f"   * [{p['platform']}] {p['title']} ({p['year']}) - {p.get('url', '')}")

        print("\n=== M3_FULL JSON SUMMARY ===")
        for qid, q in m3_full["summary"].items():
            print(f"[{qid}] {q['name']} -> deduplicated hits: {q['total_deduplicated_strict_hits']}")
        
        print("\n=== M3_FULL DETAILS HITS ===")
        for qid in ["Q8", "Q9", "Q10", "Q12"]:
            qdetail = m3_full["query_details"].get(qid, {})
            hits = qdetail.get("deduplicated_hits", [])
            print(f"--- {qid}: {qdetail.get('name')} (Count: {len(hits)}) ---")
            for h in hits:
                print(f"   * [{h.get('source', '')}] {h.get('title', '')} ({h.get('year', '')})")
                print(f"     Authors: {', '.join(h.get('authors', []))}")
                print(f"     URL: {h.get('url', '')}")
                print(f"     Abstract: {h.get('abstract', '')[:200]}...")
        return

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    school_dir = os.path.join(base_dir, "学校文件")
    output_md_path = os.path.join(school_dir, "PARSED_SCHOOL_TASKS.md")

    target_files = [
        ("任务d1-1.docx", "任务阶段 d1-1：选题方向与导师需求拆解"),
        ("任务d1-2.docx", "任务阶段 d1-2：文献调研与方法构想"),
        ("任务d1-3.docx", "任务阶段 d1-3：创新点论证与查重检验"),
        ("任务d2-2.5.docx", "任务阶段 d2-2.5：核心方法实现与实验设计"),
        ("Oth_研究方向推荐.docx", "参考资料：导师研究方向推荐与重点赛道"),
        ("Oth_科研大脑Skill.docx", "参考技能：科研大脑 Skill 架构与操作规范")
    ]

    print(f"[*] 开始解析学校任务文档，目标目录: {school_dir}")
    print(f"[*] 解析文件数: {len(target_files)}")

    md_output_parts = [
        "# 山东财经大学智能体科研创新挑战赛 · 导师任务书与官方指南全集",
        "",
        "> 本文档由 `scripts/extract_school_docx.py` 自动从 `学校文件/` 下的原始 `.docx` 文件无损解析生成。",
        f"> 解析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "> 目的：使团队所有智能体与研究人员随时精准溯源导师布置的阶段性任务、指标考核、写作标准与方法论边界要求。",
        "",
        "---",
        "",
        "## 📑 目录索引",
        ""
    ]

    for idx, (filename, title) in enumerate(target_files, 1):
        anchor = title.lower().replace(" ", "-").replace("：", "-").replace("、", "-")
        md_output_parts.append(f"{idx}. [{title}](#{idx}-{filename.replace('.', '')})")

    md_output_parts.append("")
    md_output_parts.append("---")
    md_output_parts.append("")

    for idx, (filename, title) in enumerate(target_files, 1):
        file_path = os.path.join(school_dir, filename)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        print(f"[*] 正在解析 ({idx}/{len(target_files)}): {filename} ({file_size} bytes)...")

        content = extract_docx_content(file_path)
        
        section_id = f"{idx}-{filename.replace('.', '')}"
        md_output_parts.append(f"## <a id=\"{section_id}\"></a>{idx}. 【{filename}】{title}")
        md_output_parts.append(f"- **文件路径**: `学校文件/{filename}`")
        md_output_parts.append(f"- **原始大小**: {file_size} 字节")
        md_output_parts.append("")
        md_output_parts.append("### 文档正文与提取内容：")
        md_output_parts.append("")
        md_output_parts.append(content)
        md_output_parts.append("")
        md_output_parts.append("---")
        md_output_parts.append("")

    final_content = "\n".join(md_output_parts)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(final_content)

    print(f"[✓] 解析完成！输出文件已落盘: {output_md_path} ({len(final_content.encode('utf-8'))} bytes)")

if __name__ == "__main__":
    main()
