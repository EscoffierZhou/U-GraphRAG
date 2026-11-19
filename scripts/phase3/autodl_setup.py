"""
autodl_setup.py
===============
AutoDL Environment Bootstrap, Dependency/Hardware Verification, Dataset Scaling,
and Third-Family Backbone Downloader (ModelScope / HuggingFace / hf-mirror).

Follows AboutMileStone_3.md Section VI:
  - Captures exact GPU/CUDA/PyTorch/Transformers/bitsandbytes environment manifest
  - Downloads BAAI/bge-small-en-v1.5 and scaled QA datasets if absent
  - Downloads requested LLM backbones (`qwen3b`, `qwen7b`, `llama3_8b`, `mistral7b`)
    supporting ModelScope (ungated China AutoDL speed!) and HuggingFace (`HF_TOKEN` / `hf-mirror`)
  - Executes a 1-prompt generation & EOS/chat-template smoke verification before
    launching any large experiment.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Disable multi-line download progress bars that spam Kaggle / Colab web logs
if not os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"):
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase3.phase3_common import (
    MILE3_DIR,
    MODELS_DIR,
    PHASE3_MODEL_GROUPS,
    PHASE3_MODEL_REGISTRY,
    SCALE_LIMITS,
    atomic_write_json,
    build_phase3_scaled_cache,
    collect_hardware_manifest,
    is_phase3_backbone_available,
    load_phase3_dataset_cache,
    resolve_phase3_model_list,
    sequential_load_phase3_generator,
    setup_phase2_logger,
    teardown_generator,
    verify_v5_frozen_checksums,
)

logger = setup_phase2_logger("phase3.autodl_setup")


def _download_with_modelscope(ms_repo: str, local_dir: Path) -> bool:
    """Download model weights cleanly from ModelScope (works without HF gate in China AutoDL)."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    except ImportError:
        logger.warning("modelscope package not installed. Run `pip install modelscope` or pass `--source hf`.")
        return False

    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[ModelScope] Downloading {ms_repo} -> {local_dir} ...")
    try:
        downloaded_path = Path(ms_snapshot_download(ms_repo, local_dir=str(local_dir)))
        return downloaded_path.exists() and (
            any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))
        )
    except Exception as err:
        logger.warning(f"[ModelScope] Download failed for {ms_repo}: {err}")
        return False


def _is_overseas_notebook() -> bool:
    """Detect Google Colab or Kaggle overseas runtimes where huggingface.co is directly fast."""
    return bool(os.getenv("COLAB_RELEASE_TAG") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"))


def _get_models_target_root() -> Path:
    """Target directory for downloaded models (respects UGRAPHRAG_MODELS_DIR for Kaggle /tmp/models)."""
    custom_models_dir = os.getenv("UGRAPHRAG_MODELS_DIR")
    return Path(custom_models_dir) if custom_models_dir else MODELS_DIR


def _download_with_hf(hf_repo: str, local_dir: Path) -> bool:
    """Download model weights from HuggingFace (or hf-mirror inside China) respecting HF_TOKEN."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub package not available.")
        return False

    token = os.environ.get("HF_TOKEN") or None
    if not os.environ.get("HF_ENDPOINT") and not _is_overseas_notebook():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    local_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logger.info(f"[HF] Downloading {hf_repo} (endpoint={endpoint}) -> {local_dir} ... (progress bars silenced)")
    try:
        snapshot_download(
            repo_id=hf_repo,
            local_dir=str(local_dir),
            token=token,
            ignore_patterns=["*.bin", "*.pt", "*.pth", "*.msgpack", "*.h5", "*.onnx*", "original/*"],
            max_workers=8,
            tqdm_class=None,
        )
        elapsed = time.time() - t0
        sz_gb = sum(f.stat().st_size for f in local_dir.glob("*.safetensors")) / (1024 ** 3)
        logger.info(f"[HF] Downloaded {hf_repo} in {elapsed:.1f}s ({sz_gb:.2f} GB).")
        return any(local_dir.glob("*.safetensors"))
    except Exception as err:
        logger.warning(f"[HF] Download failed for {hf_repo}: {err}")
        return False


def ensure_bge_dense_model(download: bool) -> Dict[str, Any]:
    """Ensure local BAAI/bge-small-en-v1.5 weights exist for fast CPU dense retrieval."""
    for candidate_root in (_get_models_target_root(), MODELS_DIR, Path("/root/autodl-tmp/models"), Path("/tmp/models")):
        cand_dir = candidate_root / "bge-small-en-v1.5"
        if (cand_dir / "model.safetensors").exists() or (cand_dir / "pytorch_model.bin").exists():
            return {"status": "present", "path": str(cand_dir)}
    bge_dir = _get_models_target_root() / "bge-small-en-v1.5"
    if not download:
        return {"status": "missing_dry_fallback", "path": str(bge_dir)}
    if _is_overseas_notebook():
        ok = (
            _download_with_hf("BAAI/bge-small-en-v1.5", bge_dir)
            or _download_with_modelscope("BAAI/bge-small-en-v1.5", bge_dir)
            or _download_with_modelscope("AI-ModelScope/bge-small-en-v1.5", bge_dir)
        )
    else:
        ok = (
            _download_with_modelscope("BAAI/bge-small-en-v1.5", bge_dir)
            or _download_with_modelscope("AI-ModelScope/bge-small-en-v1.5", bge_dir)
            or _download_with_hf("BAAI/bge-small-en-v1.5", bge_dir)
        )
    if ok:
        return {"status": "downloaded", "path": str(bge_dir)}
    return {"status": "download_failed", "path": str(bge_dir)}


def ensure_backbone(model_key: str, download: bool, source: str = "auto") -> Dict[str, Any]:
    """Ensure one Phase III LLM backbone is cached cleanly on disk."""
    if is_phase3_backbone_available(model_key, allow_hf_online=False):
        return {"model": model_key, "status": "present"}
    if not download:
        return {"model": model_key, "status": "missing_not_requested"}
    if model_key not in PHASE3_MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'. Valid: {list(PHASE3_MODEL_REGISTRY.keys())}")

    info = PHASE3_MODEL_REGISTRY[model_key]
    target_dir = _get_models_target_root() / info["local_dirs"][0]
    ok = False
    hf_primary = info["official_hf_repo"] if (os.environ.get("HF_TOKEN") and info.get("official_hf_repo")) else info["hf_repo"]
    if _is_overseas_notebook():
        # Overseas Kaggle/Colab: HuggingFace CDN is 100-300 MB/s vs slow ModelScope; keep ModelScope as fallback
        if source in ("hf", "auto"):
            ok = _download_with_hf(hf_primary, target_dir)
            if not ok and hf_primary != info["hf_repo"]:
                ok = _download_with_hf(info["hf_repo"], target_dir)
        if not ok and source in ("modelscope", "auto"):
            ok = _download_with_modelscope(info["modelscope_repo"], target_dir)
    else:
        if source in ("modelscope", "auto"):
            if source == "modelscope" or (info["gated_hf"] and not os.environ.get("HF_TOKEN")):
                ok = _download_with_modelscope(info["modelscope_repo"], target_dir)
        if not ok and source in ("hf", "auto"):
            ok = _download_with_hf(hf_primary, target_dir)
            if not ok and hf_primary != info["hf_repo"]:
                ok = _download_with_hf(info["hf_repo"], target_dir)
        if not ok and source == "auto":
            ok = _download_with_modelscope(info["modelscope_repo"], target_dir)

    return {"model": model_key, "status": "downloaded" if ok else "failed", "target_dir": str(target_dir)}


def smoke_verify_backbones(
    models: Sequence[str],
    dry_run: bool,
    quantization: str = "4bit",
) -> Dict[str, Any]:
    """Verify chat template, EOS tokens, and 1-prompt generation on each available backbone."""
    res: Dict[str, Any] = {}
    sample_docs = [{"title": "Zurich", "text": "Zurich is the largest city in Switzerland, noted for global banking."}]
    question = "What country is Zurich located in?"
    for mk in models:
        if not dry_run and not is_phase3_backbone_available(mk, allow_hf_online=False):
            res[mk] = {"status": "skipped_missing"}
            continue
        gen = None
        try:
            gen = sequential_load_phase3_generator(mk, dry_run=dry_run, max_new_tokens=16, quantization=quantization)
            prompt = gen.build_prompt(question, sample_docs)
            ans, calls = gen.generate(prompt, sample_docs, max_new_tokens=16)
            res[mk] = {
                "status": "ok",
                "sample_answer": ans,
                "forward_calls": calls,
                "prompt_len_chars": len(prompt),
            }
        except Exception as err:
            res[mk] = {"status": "error", "error": str(err)}
        finally:
            teardown_generator(gen)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoDL Milestone 3 Setup, Dataset Scaling & Backbone Verification.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Backbone model keys (e.g. qwen3b, qwen7b, llama3_8b, mistral7b, gemma2)",
    )
    parser.add_argument(
        "--model-group",
        type=str,
        default=None,
        choices=["core3", "extra2", "all5"],
        help="3+2 Kaggle anti-disk-overflow group selector (core3 | extra2 | all5)",
    )
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "popqa", "2wikimultihop", "musique"])
    parser.add_argument("--scale", type=str, default="medium", choices=list(SCALE_LIMITS.keys()))
    parser.add_argument("--download-datasets", action="store_true")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "modelscope", "hf"])
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "bf16"])
    parser.add_argument("--out-file", type=str, default=str(MILE3_DIR / "setup_verification.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    active_models = resolve_phase3_model_list(models=args.models, model_group=args.model_group)

    out_path = Path(args.out_file)
    setup_phase2_logger("phase3.autodl_setup", out_path.parent / "autodl_setup.log")
    v5_info = verify_v5_frozen_checksums()
    hw = collect_hardware_manifest()
    disk = shutil.disk_usage(str(PROJECT_ROOT))
    hw["disk_free_gb"] = round(disk.free / (1024 ** 3), 2)

    bge_status = ensure_bge_dense_model(download=args.download_models and not args.dry_run)
    model_status = {mk: ensure_backbone(mk, download=args.download_models and not args.dry_run, source=args.source) for mk in active_models}

    ds_status: Dict[str, Any] = {}
    for ds in args.datasets:
        if args.scale == "full" and not args.dry_run:
            dev_lim, test_lim = SCALE_LIMITS["full"][ds]
            cache = build_phase3_scaled_cache(
                ds,
                target_total=max(1200, dev_lim + test_lim + 120),
                download=args.download_datasets,
                allow_synthetic_fallback=False,
            )
            dev_items = [it for it in cache["items"] if it["split"] == "dev"][:dev_lim]
            test_items = [it for it in cache["items"] if it["split"] == "test"][:test_lim]
        else:
            dev_items, test_items = load_phase3_dataset_cache(ds, scale=args.scale, allow_synthetic_fallback=args.dry_run)
        ds_status[ds] = {"n_dev": len(dev_items), "n_test": len(test_items)}

    smoke_status = smoke_verify_backbones(active_models, dry_run=args.dry_run, quantization=args.quantization)

    payload = {
        "v5_integrity": v5_info,
        "hardware_manifest": hw,
        "bge_dense_status": bge_status,
        "active_models": active_models,
        "model_download_status": model_status,
        "dataset_splits": ds_status,
        "smoke_checks": smoke_status,
    }
    atomic_write_json(out_path, payload)
    logger.info(f"Saved AutoDL setup manifest -> {out_path}")


if __name__ == "__main__":
    main()
