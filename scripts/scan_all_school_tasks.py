# -*- coding: utf-8 -*-
import os
import glob
import zipfile
import xml.etree.ElementTree as ET

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

def extract_docx_text(docx_path):
    try:
        with zipfile.ZipFile(docx_path, 'r') as zf:
            if 'word/document.xml' not in zf.namelist():
                return ""
            xml_data = zf.read('word/document.xml')
        root = ET.fromstring(xml_data)
        body = root.find(f"{W_NS}body")
        if body is None:
            return ""
        lines = []
        for elem in body:
            tag = elem.tag
            if tag == f"{W_NS}p":
                texts = [t.text for t in elem.iter(f"{W_NS}t") if t.text]
                p_text = "".join(texts).strip()
                if p_text:
                    lines.append(p_text)
            elif tag == f"{W_NS}tbl":
                for tr in elem.findall(f"{W_NS}tr"):
                    row = []
                    for tc in tr.findall(f"{W_NS}tc"):
                        c_text = "".join([t.text for t in tc.iter(f"{W_NS}t") if t.text]).strip()
                        row.append(c_text)
                    if any(row):
                        lines.append(" | ".join(row))
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def main():
    school_dir = os.path.abspath("学校文件")
    print(f"Checking directory: {school_dir}, exists: {os.path.exists(school_dir)}")
    if os.path.exists(school_dir):
        all_files = os.listdir(school_dir)
        print(f"Total files in dir: {len(all_files)}")
        docx_files = [os.path.join(school_dir, f) for f in all_files if f.endswith(".docx")]
    else:
        docx_files = []
    print(f"=== Total {len(docx_files)} docx files found ===")
    for p in sorted(docx_files):
        fname = os.path.basename(p)
        txt = extract_docx_text(p)
        lines = [line.strip() for line in txt.split("\n") if line.strip()]
        first_lines = lines[:6]
        print(f"\n[{fname}] (lines: {len(lines)})")
        for l in first_lines:
            print(f"   > {l[:120]}")

if __name__ == "__main__":
    main()
