"""
instruct_rag_icl_operator.py
============================
Training-free InstructRAG-ICL denoising baseline (ICLR 2025, Wei et al.) extracted from
`repos/InstructRAG/src/rag.json` and `src/data_utils.py`. Uses explicit self-synthesized
rationale generation to filter conflicting/noisy retrieved passages before extracting
the final short answer cleanly under identical postprocessing rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from results.MileStone4_Competitor.reuse.operators import build_prompt_with_generator
from src.baseline_rag import collect_eos_and_pad_ids
from src.m3_calibrated_decoder import CALIBRATION_TEMP_GRID, M3CalibratedDecoder

INSTRUCT_RAG_TASK_HEADER = (
    "Your task is to analyze the provided documents and answer the given question. "
    "Please generate a brief explanation of how the contents of these documents lead to your answer, "
    "filtering out any misleading, conflicting, or irrelevant passages. "
    "End your response with exactly one final line starting with 'Short Answer: '."
)


def build_instructrag_icl_prompt(
    question: str,
    evidence_docs: List[Dict[str, Any]],
    generator: Any = None,
) -> str:
    docs_text = "\n\n".join(
        f"Document {idx + 1} (Title: {d.get('title', f'Doc_{idx + 1}')}): {d.get('text', '')}"
        for idx, d in enumerate(evidence_docs)
    )
    user_msg = (
        f"{INSTRUCT_RAG_TASK_HEADER}\n\n"
        f"{docs_text}\n\n"
        f"Based on the provided information, answer the question:\n{question}"
    )
    tok = getattr(generator, "tokenizer", None)
    if tok is not None and getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{user_msg}\n\nExplanation and Short Answer:\n"


def extract_instructrag_short_answer(raw_text: str, generator: Any = None) -> str:
    """Extract concise final answer from InstructRAG rationale output."""
    text = (raw_text or "").strip()
    m = re.search(r"(?:Short Answer|Final Answer|Conclusion|Therefore, the answer is)\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        ans = lines[-1] if lines else ""
    if generator is not None and hasattr(generator, "_postprocess_answer"):
        return generator._postprocess_answer(ans)
    return ans.strip(" \t\r\n.\"'")


@dataclass
class InstructRAGResult:
    raw_answer: str
    full_rationale: str
    raw_confidence: float
    ctx_probs_by_temp: Dict[str, float]


def generate_with_instructrag_icl(
    generator: Any,
    question: str,
    evidence_docs: List[Dict[str, Any]],
    max_new_tokens: int = 96,
) -> InstructRAGResult:
    norm_docs = [M3CalibratedDecoder.normalize_evidence_item(d) for d in evidence_docs]
    if getattr(generator, "dry_run", True) or getattr(generator, "model", None) is None:
        ans, _ = generator.generate(
            build_prompt_with_generator(question, norm_docs, generator=generator),
            evidence_docs,
            max_new_tokens=48,
        )
        rel = max((d["reliability"] for d in norm_docs), default=0.5)
        conf = round(max(0.05, min(0.95, 0.55 + 0.25 * (rel - 0.5))), 4)
        return InstructRAGResult(
            raw_answer=ans,
            full_rationale=f"Short Answer: {ans}",
            raw_confidence=conf,
            ctx_probs_by_temp={str(float(t)): conf for t in CALIBRATION_TEMP_GRID},
        )

    model = generator.model
    tokenizer = generator.tokenizer
    prompt = build_instructrag_icl_prompt(question, evidence_docs, generator=generator)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    eos_ids, pad_id = collect_eos_and_pad_ids(tokenizer)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_ids,
            pad_token_id=pad_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    gen_tokens = out.sequences[0][inputs.input_ids.shape[1]:]
    full_txt = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    ans = extract_instructrag_short_answer(full_txt, generator=generator)

    # Mean un-adjusted token probability over answer span (tail tokens)
    tail_scores = out.scores[-min(len(out.scores), max(1, len(ans.split()) + 2)):] if out.scores else []
    temps = {str(float(t)): [] for t in CALIBRATION_TEMP_GRID}
    offset = max(0, len(out.scores) - len(tail_scores))
    for i, sc in enumerate(tail_scores):
        tok_idx = offset + i
        if tok_idx >= len(gen_tokens):
            break
        tid = int(gen_tokens[tok_idx].item())
        vec = sc[0].float()
        for t in CALIBRATION_TEMP_GRID:
            temps[str(float(t))].append(float(torch.softmax(vec / float(t), dim=-1)[tid].item()))
    avg_temps = {k: float(sum(v) / max(1, len(v))) if v else 0.5 for k, v in temps.items()}
    return InstructRAGResult(
        raw_answer=ans,
        full_rationale=full_txt,
        raw_confidence=round(avg_temps.get("1.0", 0.5), 4),
        ctx_probs_by_temp=avg_temps,
    )
