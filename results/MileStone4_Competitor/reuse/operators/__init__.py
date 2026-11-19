"""Curated zero-training competitor operators for MileStone_After benchmarking."""

from typing import Any, Dict, List


def build_prompt_with_generator(
    question: str,
    evidence_docs: List[Dict[str, Any]],
    generator: Any = None,
) -> str:
    """Delegate prompt formatting to generator.build_prompt when available, matching M3 baseline."""
    build_fn = getattr(generator, "build_prompt", None)
    if callable(build_fn):
        return build_fn(question, evidence_docs)
    if evidence_docs:
        refs = "\n".join(
            f"[{i + 1}] {d.get('title', f'Doc_{i + 1}')}: {d.get('text', '')}"
            for i, d in enumerate(evidence_docs)
        )
        return (
            "You are a precise extractive QA assistant. Output ONLY the exact short answer.\n\n"
            f"References:\n{refs}\n\nQuestion: {question}\nShort Answer:"
        )
    return f"Output ONLY the exact short answer with no explanation.\n\nQuestion: {question}\nShort Answer:"

