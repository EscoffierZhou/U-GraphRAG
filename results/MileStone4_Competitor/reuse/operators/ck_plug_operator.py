"""
ck_plug_operator.py
===================
Faithful pure-PyTorch extraction of CK-PLUG (ICLR 2026, Bi et al., arXiv:2503.15888)
from `repos/CK-PLUG/transformers-4.49/src/transformers/generation/utils.py#L2585-L2773`.
Runs natively on standard HuggingFace `AutoModelForCausalLM` (including 4-bit NF4)
without installing a modified `transformers` fork, and logs BOTH un-modulated (`ctx`)
and CK-PLUG modulated token probabilities so post-modulation calibration drift
(ECE/AUROC/AURC) can be audited rigorously.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from results.MileStone4_Competitor.reuse.operators import build_prompt_with_generator
from src.baseline_rag import META_TAIL_STOP_RE, collect_eos_and_pad_ids
from src.m3_calibrated_decoder import (
    CALIBRATION_TEMP_GRID,
    M3CalibratedDecoder,
    collect_newline_boundary_ids,
)


def relative_top_filter_pair(
    scores: torch.Tensor,
    scores_student: torch.Tensor,
    relative_top: float = 0.01,
    filter_value: float = -1e10,
    min_tokens_to_keep: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact port of CK-PLUG `relative_top_filter` keeping top probability support across both branches."""
    scores_norm = scores.log_softmax(dim=-1)
    student_norm = scores_student.log_softmax(dim=-1)
    k = max(1, min(int(min_tokens_to_keep), scores_norm.shape[-1]))

    sorted_ctx, _ = torch.sort(scores_norm, descending=True)
    ctx_thresh = torch.min(
        sorted_ctx[..., k - 1],
        torch.max(scores_norm, dim=-1).values + math.log(max(1e-8, float(relative_top))),
    ).unsqueeze(-1)

    sorted_stu, _ = torch.sort(student_norm, descending=True)
    stu_thresh = torch.min(
        sorted_stu[..., k - 1],
        torch.max(student_norm, dim=-1).values + math.log(max(1e-8, float(relative_top))),
    ).unsqueeze(-1)

    mask = (scores_norm < ctx_thresh) & (student_norm < stu_thresh)
    ctx_out = scores.masked_fill(mask, filter_value)
    stu_out = scores_student.masked_fill(mask, filter_value)
    return ctx_out, stu_out, mask


def compute_ck_plug_step_logits(
    logits_ctx: torch.Tensor,
    logits_student: torch.Tensor,
    alpha: float = 0.5,
    adaptive: bool = True,
    select_top: int = 10,
    relative_top: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply single-step CK-PLUG Confidence Gain detection and logit modulation:
      IG = H_student - H_ctx; trigger adjustment when IG - |H_ctx| < 0 (i.e., H_student < 2*H_ctx).
    """
    ctx_f, stu_f, mask = relative_top_filter_pair(
        logits_ctx.clone().float(),
        logits_student.clone().float(),
        relative_top=relative_top,
        filter_value=-1e10,
        min_tokens_to_keep=select_top,
    )
    probs_ctx = torch.softmax(ctx_f, dim=-1)
    probs_stu = torch.softmax(stu_f, dim=-1)
    h_ctx = float(-torch.sum(probs_ctx * torch.log(probs_ctx + 1e-9), dim=-1).item())
    h_stu = float(-torch.sum(probs_stu * torch.log(probs_stu + 1e-9), dim=-1).item())
    ig = h_stu - h_ctx
    is_adjust = (ig - abs(h_ctx)) < 0.0

    if is_adjust:
        stu_work = stu_f.clone()
        stu_work[mask] = -1e3
        logits_diff = ctx_f - stu_work
        stu_work[mask] = -1e10
        if adaptive:
            denom = h_ctx + h_stu * (1.0 + abs(h_ctx - h_stu) / max(1e-8, h_ctx + h_stu))
            denom = max(1e-8, denom)
            stu_part = 2.0 * stu_work * h_ctx / denom
            ctx_part = 2.0 * logits_diff * h_stu / denom
            out_logits = stu_part + ctx_part
        else:
            out_logits = float(alpha) * stu_work + (1.0 - float(alpha)) * logits_diff
    else:
        out_logits = ctx_f

    return out_logits, {
        "h_ctx": h_ctx,
        "h_student": h_stu,
        "ig": ig,
        "adjusted": bool(is_adjust),
    }


@dataclass
class CKPlugResult:
    raw_answer: str
    raw_confidence_modulated: float
    raw_confidence_unadjusted: float
    mod_probs_by_temp: Dict[str, float]
    ctx_probs_by_temp: Dict[str, float]
    adjust_fraction: float
    forward_calls: int


def generate_with_ck_plug(
    generator: Any,
    question: str,
    evidence_docs: List[Dict[str, Any]],
    alpha: float = 0.5,
    adaptive: bool = True,
    select_top: int = 10,
    max_new_tokens: int = 48,
) -> CKPlugResult:
    """Run deterministic greedy decoding under CK-PLUG while recording calibration telemetry."""
    norm_docs = [M3CalibratedDecoder.normalize_evidence_item(d) for d in evidence_docs]
    if getattr(generator, "dry_run", True) or getattr(generator, "model", None) is None:
        prompt = build_prompt_with_generator(question, norm_docs, generator=generator)
        ans, calls = generator.generate(prompt, evidence_docs, max_new_tokens=max_new_tokens)
        rel = max((d["reliability"] for d in norm_docs), default=0.5)
        conf_unadj = max(0.05, min(0.95, 0.55 + 0.30 * (rel - 0.5)))
        # Simulated slight overconfidence drift under logit modulation
        conf_mod = min(0.99, conf_unadj + 0.08)
        return CKPlugResult(
            raw_answer=ans,
            raw_confidence_modulated=round(conf_mod, 4),
            raw_confidence_unadjusted=round(conf_unadj, 4),
            mod_probs_by_temp={str(float(t)): conf_mod for t in CALIBRATION_TEMP_GRID},
            ctx_probs_by_temp={str(float(t)): conf_unadj for t in CALIBRATION_TEMP_GRID},
            adjust_fraction=0.5,
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
    stu_in = tokenizer(prior_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    gen_ids: List[int] = []
    ctx_step_temps: Dict[str, List[float]] = {str(float(t)): [] for t in CALIBRATION_TEMP_GRID}
    mod_step_temps: Dict[str, List[float]] = {str(float(t)): [] for t in CALIBRATION_TEMP_GRID}
    adj_flags: List[float] = []

    with torch.no_grad():
        ctx_mask = ctx_in.attention_mask
        stu_mask = stu_in.attention_mask
        c_out = model(input_ids=ctx_in.input_ids, attention_mask=ctx_mask, use_cache=True)
        s_out = model(input_ids=stu_in.input_ids, attention_mask=stu_mask, use_cache=True)
        past_c, past_s = c_out.past_key_values, s_out.past_key_values
        cur_c, cur_s = c_out.logits[:, -1, :], s_out.logits[:, -1, :]
        one_c = torch.ones((1, 1), device=model.device, dtype=ctx_mask.dtype)
        one_s = torch.ones((1, 1), device=model.device, dtype=stu_mask.dtype)

        for step_i in range(max_new_tokens):
            raw_ctx_vec = cur_c[0].float()
            adj_logits, info = compute_ck_plug_step_logits(
                cur_c, cur_s, alpha=alpha, adaptive=adaptive, select_top=select_top
            )
            adj_flags.append(1.0 if info["adjusted"] else 0.0)
            adj_vec = adj_logits[0].float()
            next_tid = int(torch.argmax(adj_vec).item())

            if next_tid in eos_set:
                break
            if gen_ids and next_tid in nl_ids:
                break

            gen_ids.append(next_tid)
            for t in CALIBRATION_TEMP_GRID:
                tk = str(float(t))
                ctx_step_temps[tk].append(float(torch.softmax(raw_ctx_vec / float(t), dim=-1)[next_tid].item()))
                mod_step_temps[tk].append(float(torch.softmax(adj_vec / float(t), dim=-1)[next_tid].item()))

            if len(gen_ids) >= 3:
                partial = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                m_tail = META_TAIL_STOP_RE.search(partial)
                if m_tail and partial[: m_tail.start()].strip():
                    break

            if step_i + 1 >= max_new_tokens:
                break
            nxt = torch.tensor([[next_tid]], device=model.device, dtype=ctx_in.input_ids.dtype)
            ctx_mask = torch.cat([ctx_mask, one_c], dim=-1)
            stu_mask = torch.cat([stu_mask, one_s], dim=-1)
            c_out = model(input_ids=nxt, attention_mask=ctx_mask, past_key_values=past_c, use_cache=True)
            s_out = model(input_ids=nxt, attention_mask=stu_mask, past_key_values=past_s, use_cache=True)
            past_c, past_s = c_out.past_key_values, s_out.past_key_values
            cur_c, cur_s = c_out.logits[:, -1, :], s_out.logits[:, -1, :]

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
    return CKPlugResult(
        raw_answer=cleaned,
        raw_confidence_modulated=round(mod_avg.get("1.0", 0.5), 4),
        raw_confidence_unadjusted=round(ctx_avg.get("1.0", 0.5), 4),
        mod_probs_by_temp=mod_avg,
        ctx_probs_by_temp=ctx_avg,
        adjust_fraction=round(sum(adj_flags) / max(1, len(adj_flags)), 4),
        forward_calls=2,
    )
