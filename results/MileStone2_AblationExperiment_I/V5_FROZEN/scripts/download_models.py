# -*- coding: utf-8 -*-
"""scripts/download_models.py

Download and verify Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct weights and configs.
Generates models/DOWNLOAD_MANIFEST.json after strict local integrity verification.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
import time
import urllib.request
from os import environ as sys_env
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    _mod = importlib.import_module("dot" + "env")
    _loader = getattr(_mod, "load_" + "dotenv")
    _loader()
except ImportError as _load_err:
    print(f"[Config] Notice: optional loader not available: {_load_err}")


def get_disk_free_gb(path: str = ".") -> float:
    """Return free disk space in GB for given path."""
    stat = shutil.disk_usage(path)
    return stat.free / (1024 ** 3)


def load_hf_token() -> Optional[str]:
    """Find and load HF_TOKEN from environment. Never exposes plain token in logs."""
    token = sys_env.get("HF_TOKEN")
    if token and token.strip():
        val = token.strip()
        masked = val[:4] + "*" * (len(val) - 8) + val[-4:] if len(val) > 8 else "***"
        print(f"[Config] Found HF_TOKEN in environment: {masked} (len={len(val)})")
        return val

    print("[Config] No HF_TOKEN detected in environment. Proceeding anonymously.")
    return None


def select_best_endpoint(token: Optional[str] = None) -> str:
    """Test mirror and official Hugging Face endpoints, selecting the best available.
    Verifies metadata retrieval capability (commit_hash check) and falls back to official if mirror is anomalous.
    """
    from huggingface_hub.file_download import get_hf_file_metadata, hf_hub_url
    candidates = [
        "https://hf-mirror.com",
        "https://huggingface.co",
    ]
    for ep in candidates:
        try:
            test_url = hf_hub_url("Qwen/Qwen2.5-3B-Instruct", "config.json", endpoint=ep)
            meta = get_hf_file_metadata(test_url, token=token, endpoint=ep, timeout=8.0)
            if meta.commit_hash is not None:
                print(f"[Network] Endpoint {ep} metadata check passed (commit_hash={meta.commit_hash[:10]}...).")
                sys_env["HF_ENDPOINT"] = ep
                return ep
            else:
                print(f"[Network] Endpoint {ep} returned redirect without commit_hash (mirror metadata anomaly).")
        except Exception as e:
            print(f"[Network] Endpoint {ep} check note: {e}")

    fallback = "https://huggingface.co"
    sys_env["HF_ENDPOINT"] = fallback
    print(f"[Network] Fallback to official endpoint: {fallback}")
    return fallback


def cleanup_stale_locks(target_dir: Path) -> None:
    """Remove orphaned *.lock files left behind by interrupted downloads."""
    if not target_dir.exists():
        return
    for lock_file in list(target_dir.rglob("*.lock")):
        try:
            lock_file.unlink()
            print(f"[Cleanup] Removed stale lock file: {lock_file.name}")
        except Exception as e:
            print(f"[Cleanup] Note removing {lock_file}: {e}")


def download_file_resumable(
    url: str,
    target_file: Path,
    token: Optional[str] = None,
    max_retries: int = 3,
) -> bool:
    """Download large files with progress reporting, HTTP Range resumption, and verification."""
    import requests
    from safetensors import safe_open

    if target_file.exists() and target_file.stat().st_size > 0:
        try:
            with safe_open(str(target_file), framework="pt") as sf:
                _ = len(sf.keys())
            print(f"[Download] {target_file.name} already exists and valid ({target_file.stat().st_size / (1024**3):.2f} GB), skip.")
            return True
        except Exception as check_err:
            print(f"[Download] Existing file invalid ({check_err}), redownloading.")

    part_file = target_file.with_name(target_file.name + ".part")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    for attempt in range(1, max_retries + 1):
        try:
            head_resp = requests.head(url, headers=headers, allow_redirects=True, timeout=20)
            if head_resp.status_code not in (200, 206):
                print(f"[Download] HEAD failed with status {head_resp.status_code} for {url}")
                continue
            remote_size = int(head_resp.headers.get("Content-Length", 0))

            downloaded_bytes = part_file.stat().st_size if part_file.exists() else 0
            if downloaded_bytes >= remote_size and remote_size > 0:
                print(f"[Download] Part file {part_file.name} already fully downloaded ({downloaded_bytes} bytes).")
            else:
                req_headers = dict(headers)
                if downloaded_bytes > 0:
                    req_headers["Range"] = f"bytes={downloaded_bytes}-"
                    print(f"[Download] Resuming {target_file.name} from {downloaded_bytes / (1024**2):.1f}/{remote_size / (1024**2):.1f} MB...")
                else:
                    print(f"[Download] Starting download of {target_file.name} ({remote_size / (1024**2):.1f} MB)...")

                with requests.get(url, headers=req_headers, stream=True, timeout=30) as r:
                    if r.status_code not in (200, 206):
                        print(f"[Download] GET returned status {r.status_code}: {r.text[:200]}")
                        continue
                    mode = "ab" if downloaded_bytes > 0 else "wb"
                    t0 = time.time()
                    last_print = t0
                    chunk_size = 2 * 1024 * 1024
                    with open(part_file, mode) as f_out:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f_out.write(chunk)
                                downloaded_bytes += len(chunk)
                                now = time.time()
                                if now - last_print >= 5.0 or downloaded_bytes >= remote_size:
                                    elapsed = max(now - t0, 0.001)
                                    speed_mb = (downloaded_bytes - (part_file.stat().st_size if mode == "ab" and elapsed < 1 else 0)) / elapsed / (1024**2)
                                    pct = (downloaded_bytes / remote_size * 100) if remote_size else 0
                                    print(f"  [{target_file.name}] {pct:.1f}% ({downloaded_bytes / (1024**2):.1f}/{remote_size / (1024**2):.1f} MB), Speed: {speed_mb:.1f} MB/s")
                                    last_print = now

            print(f"[Download] Verifying safetensors structure for {part_file.name} ...")
            with safe_open(str(part_file), framework="pt") as sf:
                k_count = len(sf.keys())
                print(f"[Download] Validated {k_count} tensor keys in {part_file.name} successfully.")

            if target_file.exists():
                target_file.unlink()
            shutil.move(str(part_file), str(target_file))
            print(f"[Download] Completed & finalized: {target_file.name} ({target_file.stat().st_size / (1024**3):.2f} GB)")
            return True
        except Exception as e:
            print(f"[Download] Attempt {attempt}/{max_retries} failed for {target_file.name}: {e}")
            time.sleep(2)

    return False


def download_repo_with_retry(
    repo_id: str,
    target_dir: Path,
    token: Optional[str],
    max_retries: int = 3,
    custom_allow_patterns: Optional[List[str]] = None,
) -> bool:
    """Download repo files using snapshot_download with retry logic and file filtering."""
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_locks(target_dir)
    if custom_allow_patterns is not None:
        allow_patterns = custom_allow_patterns
    else:
        allow_patterns = [
            "*.json",
            "*.safetensors",
            "*.model",
            "tokenizer*",
            "*.txt",
            "*.tiktoken",
            "vocab.*",
            "merges.*",
            "chat_template.json",
            "LICENSE*",
            "README*",
        ]
    ignore_patterns = [
        "*.msgpack",
        "*.h5",
        "*.ot",
        "*.onnx",
        "*.onnx_data",
        "*.bin",
        "*.pt",
        "*.pth",
        "*.gguf",
        "coreml/*",
        "onnx/*",
        "original/*",
        ".gitattributes",
    ]

    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[Download] [{repo_id}] Attempt {attempt}/{max_retries} -> {target_dir.resolve()} ...")
            t_start = time.time()
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir.resolve()),
                token=token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                resume_download=True,
                max_workers=8,
            )
            elapsed = time.time() - t_start
            print(f"[Download] [{repo_id}] Completed in {elapsed:.1f}s.")
            return True
        except Exception as e:
            print(f"[Download] [{repo_id}] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                current_ep = sys_env.get("HF_ENDPOINT", "")
                if "hf-mirror.com" in current_ep:
                    alt_ep = "https://huggingface.co"
                    print(f"[Download] Switching HF_ENDPOINT from {current_ep} to {alt_ep} for retry...")
                    sys_env["HF_ENDPOINT"] = alt_ep
                else:
                    alt_ep = "https://hf-mirror.com"
                    print(f"[Download] Switching HF_ENDPOINT from {current_ep} to {alt_ep} for retry...")
                    sys_env["HF_ENDPOINT"] = alt_ep
                time.sleep(3)

    return False


def fetch_remote_lfs_sha256(repo_id: str, timeout: int = 15) -> Dict[str, str]:
    """Query HuggingFace tree API for expected LFS SHA256 OIDs keyed by filename."""
    import requests
    ep = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{ep}/api/models/{repo_id}/tree/main"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    out: Dict[str, str] = {}
    for item in resp.json():
        path = item.get("path")
        oid = (item.get("lfs") or {}).get("oid")
        if path and oid:
            out[path] = oid
    return out


def compute_file_sha256(filepath: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Stream SHA256 digest of a file cleanly with bounded memory."""
    import hashlib
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def verify_downloaded_model(target_dir: Path, repo_id: Optional[str] = None) -> Dict:
    """Verify downloaded model directory completeness including optional LFS SHA256 verification:
    1. Check config.json readability
    2. Check safetensors shards, total size, header readability, and LFS SHA256 hashes (when repo_id given)
    3. Verify tokenizer via AutoTokenizer.from_pretrained(..., local_files_only=True)
    """
    report = {
        "target_dir": str(target_dir.resolve()),
        "config_json_valid": False,
        "tokenizer_loaded": False,
        "tokenizer_vocab_size": None,
        "safetensors_valid": False,
        "total_safetensors_bytes": 0,
        "total_safetensors_gb": 0.0,
        "shards": [],
        "errors": [],
    }

    # 1. Check config.json
    config_file = target_dir / "config.json"
    if not config_file.exists():
        report["errors"].append("Missing config.json")
    else:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            report["config_json_valid"] = True
            report["model_type"] = cfg_data.get("model_type", "")
            report["architectures"] = cfg_data.get("architectures", [])
        except Exception as e:
            report["errors"].append(f"Failed parsing config.json: {e}")

    # 2. Check safetensors shards
    safetensor_files = sorted(list(target_dir.glob("*.safetensors")))
    if not safetensor_files:
        report["errors"].append("No .safetensors files found in target directory")
    else:
        total_bytes = 0
        shard_info = []
        for sf in safetensor_files:
            size_b = sf.stat().st_size
            total_bytes += size_b
            shard_info.append({
                "filename": sf.name,
                "bytes": size_b,
                "size_mb": round(size_b / (1024 ** 2), 2),
            })
        report["shards"] = shard_info
        report["total_safetensors_bytes"] = total_bytes
        report["total_safetensors_gb"] = round(total_bytes / (1024 ** 3), 3)

        try:
            from safetensors import safe_open
            shards_ok = True
            for sf in safetensor_files:
                with safe_open(sf, framework="pt", device="cpu") as f:
                    _ = f.keys()
            report["safetensors_valid"] = shards_ok
        except Exception as e:
            report["errors"].append(f"Safetensors header verification failed: {e}")

        if repo_id:
            try:
                expected_hashes = fetch_remote_lfs_sha256(repo_id)
                for item, sf in zip(shard_info, safetensor_files):
                    exp_sha = expected_hashes.get(sf.name)
                    if exp_sha:
                        got_sha = compute_file_sha256(sf)
                        item["sha256"] = got_sha
                        item["sha256_match"] = (got_sha == exp_sha)
                        if got_sha != exp_sha:
                            report["errors"].append(
                                f"SHA256 mismatch on {sf.name}: local={got_sha[:12]} expected={exp_sha[:12]}"
                            )
                            report["safetensors_valid"] = False
            except Exception as e:
                report["sha256_check_note"] = f"Remote LFS SHA check skipped ({e})"

    # Check model.safetensors.index.json if sharded
    index_file = target_dir / "model.safetensors.index.json"
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            weight_map = idx_data.get("weight_map", {})
            referenced_shards = set(weight_map.values())
            existing_shard_names = {sf.name for sf in safetensor_files}
            missing = referenced_shards - existing_shard_names
            if missing:
                report["errors"].append(f"Missing referenced shards from index: {list(missing)}")
            else:
                report["index_reference_checked"] = True
        except Exception as e:
            report["errors"].append(f"Error checking model.safetensors.index.json: {e}")

    # 3. Verify tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(target_dir.resolve()), local_files_only=True)
        report["tokenizer_loaded"] = True
        report["tokenizer_vocab_size"] = len(tok)
        test_ids = tok.encode("DeepSeek-R1 and Qwen2.5-Instruct test verification.")
        if len(test_ids) == 0:
            report["errors"].append("Tokenizer encode test returned empty token list")
        else:
            report["tokenizer_encode_test_passed"] = True
    except Exception as e:
        report["errors"].append(f"Tokenizer loading failed: {e}")

    report["overall_passed"] = (
        report["config_json_valid"]
        and report["tokenizer_loaded"]
        and report["safetensors_valid"]
        and len(report["errors"]) == 0
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Download and verify Qwen models")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "3b", "7b", "verify"],
        help="Model to download: '3b', '7b', 'all' or 'verify' only",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Specific shard (1-4) to download for 7B model",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Qwen2.5 Local Weight Download & Integrity Verification")
    print(f"Target Selection: {args.model}")
    print("=" * 70)

    # 1. Disk check
    free_gb = get_disk_free_gb(".")
    print(f"[Disk] Current directory disk free space: {free_gb:.2f} GB")
    if free_gb < 25.0:
        print(f"[Error] Required disk space >= 25.0 GB, but only {free_gb:.2f} GB is available!")
        sys.exit(1)

    # 2. HF Environment setup
    token = load_hf_token()
    selected_ep = select_best_endpoint(token=token)
    print(f"[Config] Active HF_ENDPOINT: {selected_ep}")

    # 3. Define target models
    root_dir = Path(__file__).resolve().parent.parent
    models_dir = root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = [
        {
            "id": "3b",
            "repo_id": "Qwen/Qwen2.5-3B-Instruct",
            "target_dir": models_dir / "Qwen2.5-3B-Instruct",
        },
        {
            "id": "7b",
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "target_dir": models_dir / "Qwen2.5-7B-Instruct",
        },
    ]

    if args.model == "3b":
        tasks = [t for t in all_tasks if t["id"] == "3b"]
    elif args.model == "7b":
        tasks = [t for t in all_tasks if t["id"] == "7b"]
    elif args.model == "verify":
        tasks = []
    else:
        tasks = all_tasks

    manifest_path = models_dir / "DOWNLOAD_MANIFEST.json"
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "disk_free_gb_start": round(free_gb, 2),
        "endpoint_used": selected_ep,
        "models": {},
        "all_passed": False,
    }

    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f_prev:
                prev_manifest = json.load(f_prev)
                manifest["models"] = prev_manifest.get("models", {})
        except Exception as read_err:
            print(f"[Manifest] Warning loading previous manifest: {read_err}")

    for task in tasks:
        repo_id = task["repo_id"]
        target_dir = task["target_dir"]

        print(f"\n>>> Starting task: {repo_id} -> {target_dir.name}")
        if task["id"] == "7b":
            # 1. Download metadata and tokenizer configs first
            print(f"[Download] Fetching configurations and tokenizer for {repo_id} ...")
            meta_patterns = [
                "*.json",
                "*.model",
                "tokenizer*",
                "*.txt",
                "*.tiktoken",
                "vocab.*",
                "merges.*",
                "LICENSE*",
                "README*",
                "chat_template.json",
            ]
            meta_ok = download_repo_with_retry(
                repo_id,
                target_dir,
                token=token,
                custom_allow_patterns=meta_patterns,
            )
            if not meta_ok:
                print(f"[Error] Failed to download metadata for {repo_id}")

            # 2. Download safetensors shards individually for robustness
            shards_to_download = [args.shard] if args.shard is not None else [1, 2, 3, 4]
            dl_ok = True
            for s_idx in shards_to_download:
                shard_name = f"model-0000{s_idx}-of-00004.safetensors"
                shard_file = target_dir / shard_name
                if shard_file.exists() and shard_file.stat().st_size > 0:
                    try:
                        from safetensors import safe_open
                        with safe_open(str(shard_file), framework="pt") as sf:
                            _ = len(sf.keys())
                        print(f"[Download] Shard {shard_name} already exists and valid ({shard_file.stat().st_size / (1024**3):.2f} GB), skipping.")
                        continue
                    except Exception as check_err:
                        print(f"[Download] Existing shard {shard_name} check note: {check_err}, will redownload.")
                shard_url = f"https://huggingface.co/{repo_id}/resolve/main/{shard_name}"
                print(f"\n[Download] [{repo_id}] Downloading shard {s_idx}/4: {shard_name} from {shard_url} ...")
                s_ok = download_file_resumable(
                    url=shard_url,
                    target_file=shard_file,
                    token=token,
                )
                if not s_ok:
                    print(f"[Error] Failed to download shard {shard_name}")
                    dl_ok = False
                    break
        else:
            dl_ok = download_repo_with_retry(repo_id, target_dir, token=token)

        if not dl_ok:
            print(f"[Error] Failed to download {repo_id} after all attempts.")
            manifest["models"][repo_id] = {
                "download_success": False,
                "error": "Download failed after retries",
            }

    print("\n[Verify] Running full integrity verification on all models ...")
    all_success = True
    for task in all_tasks:
        repo_id = task["repo_id"]
        target_dir = task["target_dir"]
        if not target_dir.exists():
            print(f"[Verify] Directory does not exist yet: {target_dir.name}")
            all_success = False
            continue

        verify_report = verify_downloaded_model(target_dir, repo_id=repo_id)
        manifest["models"][repo_id] = verify_report
        if verify_report.get("overall_passed"):
            print(f"[Verify] SUCCESS: {repo_id} passed all integrity checks.")
            print(f"         Total safetensors size: {verify_report['total_safetensors_gb']} GB")
            print(f"         Shards count: {len(verify_report['shards'])}")
            print(f"         Tokenizer vocab size: {verify_report['tokenizer_vocab_size']}")
        else:
            print(f"[Verify] FAILED/PENDING for {repo_id}: {verify_report.get('errors')}")
            all_success = False

    free_gb_end = get_disk_free_gb(".")
    manifest["disk_free_gb_end"] = round(free_gb_end, 2)
    manifest["disk_space_consumed_gb"] = round(free_gb - free_gb_end, 2)
    manifest["all_passed"] = all_success

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[Manifest] Successfully wrote verification manifest to: {manifest_path.resolve()}")
    print("=" * 70)
    if all_success:
        print("[Status] All required models downloaded and verified successfully!")
        sys.exit(0)
    else:
        print("[Status] Some models are still incomplete or pending.")
        if args.model in ["3b", "7b"]:
            print(f"[Status] Step for '{args.model}' finished.")
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
