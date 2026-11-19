"""
cad_operator.py
===============
Faithful Context-Aware Decoding (CAD, NAACL 2024, Shi et al.) wrapper extracted from
`repos/DeCoRe/src/models/context_aware_decoding.py`:
  logits_cad = (1 + alpha) * logits_ctx - alpha * logits_prior  (default alpha = 0.5).
Records both un-adjusted context token probabilities and CAD-modulated probabilities
under identical stopping boundaries so CAD's calibration impact can be fairly audited.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from results.MileStone4_Competitor.reuse.operators import build_prompt_with_generator
from src.baseline_rag import META_TAIL_STOP_RE, collect_eos_and_pad_ids
from src.m3_calibrated_decoder import (
    CALIBRATION_TEMP_GRID,
    M3CalibratedDecoder,
    collect_newline_boundary_ids,
)


@dataclass
class CADResult:
    raw_answer: str
    raw_confidence_modulated: float
    raw_confidence_unadjusted: float
    mod_probs_by_temp: Dict[str, float]
    ctx_probs_by_temp: Dict[str, float]
    forward_calls: int


def generate_with_cad(
    generator: Any,
    question: str,
    evidence_docs: List[Dict[str, Any]],
    alpha: float = 0.5,
    max_new_tokens: int = 48,
) -> CADResult:
    """Run deterministic CAD decoding while preserving token probability telemetry."""
    norm_docs = [M3CalibratedDecoder.normalize_evidence_item(d) for d in evidence_docs]
    if getattr(generator, "dry_run", True) or getattr(generator, "model", None) is None:
        prompt = build_prompt_with_generator(question, norm_docs, generator=generator)
        ans, calls = generator.generate(prompt, evidence_docs, max_new_tokens=max_new_tokens)
        rel = max((d["reliability"] for d in norm_docs), default=0.5)
        conf_unadj = max(0.05, min(0.95, 0.52 + 0.30 * (rel - 0.5)))
        conf_mod = min(0.99, conf_unadj + 0.14)
        return CADResult(
            raw_answer=ans,
            raw_confidence_modulated=round(conf_mod, 4),
            raw_confidence_unadjusted=round(conf_unadj, 4),
            mod_probs_by_temp={str(float(t)): conf_mod for t in CALIBRATION_TEMP_GRID},
            ctx_probs_by_temp={str(float(t)): conf_unadj for t in CALIBRATION_TEMP_GRID},
            forward_calls=max(2, calls),
        )

    model = generator.model
    tokenizer = generator.tokenizer
    eos_ids, _ = collect_eos_and_pad_ids(tokenizer)
    eos_set = set(eos_ids)
    nl_ids = collect_newline_boundary_ids(tokenizer)

    ctx_prompt = build_prompt_with_generator(question, norm_docs, generator=generator)
    prior_prompt = build_prompt_with_generator(question, [], generator=generator)
    ctx_in = tokenizer(ctx_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    pri_in = tokenizer(prior_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    gen_ids: List[int] = []
    ctx_step_temps: Dict[str, List[float]] = {str(float(t)): [] for t in CALIBRATION_TEMP_GRID}
    mod_step_temps: Dict[str, List[float]] = {str(float(t)): [] for t in CALIBRATION_TEMP_GRID}

    with torch.no_grad():
        ctx_mask = ctx_in.attention_mask
        pri_mask = pri_in.attention_mask
        c_out = model(input_ids=ctx_in.input_ids, attention_mask=ctx_mask, use_cache=True)
        p_out = model(input_ids=pri_in.input_ids, attention_mask=pri_mask, use_cache=True)
        past_c, past_p = c_out.past_key_values, p_out.past_key_values
        cur_c, cur_p = c_out.logits[:, -1, :], p_out.logits[:, -1, :]
        one_c = torch.ones((1, 1), device=model.device, dtype=ctx_mask.dtype)
        one_p = torch.ones((1, 1), device=model.device, dtype=pri_mask.dtype)

        for step_i in range(max_new_tokens):
            raw_ctx_vec = cur_c[0].float()
            raw_pri_vec = cur_p[0].float()
            cad_vec = (1.0 + float(alpha)) * raw_ctx_vec - float(alpha) * raw_pri_vec
            next_tid = int(torch.argmax(cad_vec).item())

            if next_tid in eos_set:
                break
            if gen_ids and next_tid in nl_ids:
                break

            gen_ids.append(next_tid)
            for t in CALIBRATION_TEMP_GRID:
                tk = str(float(t))
                ctx_step_temps[tk].append(float(torch.softmax(raw_ctx_vec / float(t), dim=-1)[next_tid].item()))
                mod_step_temps[tk].append(float(torch.softmax(cad_vec / float(t), dim=-1)[next_tid].item()))

            if len(gen_ids) >= 3:
                partial = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                m_tail = META_TAIL_STOP_RE.search(partial)
                if m_tail and partial[: m_tail.start()].strip():
                    break

            if step_i + 1 >= max_new_tokens:
                break
            nxt = torch.tensor([[next_tid]], device=model.device, dtype=ctx_in.input_ids.dtype)
            ctx_mask = torch.cat([ctx_mask, one_c], dim=-1)
            pri_mask = torch.cat([pri_mask, one_p], dim=-1)
            c_out = model(input_ids=nxt, attention_mask=ctx_mask, past_key_values=past_c, use_cache=True)
            p_out = model(input_ids=nxt, attention_mask=pri_mask, past_key_values=past_p, use_cache=True)
            past_c, past_p = c_out.past_key_values, p_out.past_key_values
            cur_c, cur_p = c_out.logits[:, -1, :], p_out.logits[:, -1, :]

    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    cleaned = generator._postprocess_answer(decoded) if hasattr(generator, "_postprocess_answer") else decoded

    if gen_ids and cleaned and len(cleaned) + 2 < len(decoded):
        prefix_k = len(gen_ids)
        for ki in range(1, len(gen_ids) + 1):
            if cleaned in tokenizer.decode(gen_ids[:ki], skip_special_tokens=True):
                prefix_k = ki
                break
        ctx_step_temps = {k: v[:prefix_k] for k, v in ctx_step_temps.items()}
        mod_step_temps = {k: v[:prefix_k] for k, v in mod_step_temps.items()}

    ctx_avg = {k: float(sum(v) / max(1, len(v))) if v else 0.5 for k, v in ctx_step_temps.items()}
    mod_avg = {k: float(sum(v) / max(1, len(v))) if v else 0.5 for k, v in mod_step_temps.items()}
    return CADResult(
        raw_answer=cleaned,
        raw_confidence_modulated=round(mod_avg.get("1.0", 0.5), 4),
        raw_confidence_unadjusted=round(ctx_avg.get("1.0", 0.5), 4),
        mod_probs_by_temp=mod_avg,
        ctx_probs_by_temp=ctx_avg,
        forward_calls=2,
    )
