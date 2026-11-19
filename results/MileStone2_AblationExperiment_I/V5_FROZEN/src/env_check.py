"""
src/env_check.py
================
Hardware & Environment Audit for U-GraphRAG Step 1 Baseline.
Authentically probes GPU VRAM, CUDA, PyTorch, installed packages,
and any existing weights or benchmark datasets.
"""

import os
import sys
import platform
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List

def check_python_and_system() -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "architecture": list(platform.architecture()),
        "processor": platform.processor(),
    }

def check_cuda_and_torch() -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
        "errors": []
    }
    try:
        import torch
        res["torch_version"] = torch.__version__
        res["cuda_available"] = torch.cuda.is_available()
        res["cuda_version"] = torch.version.cuda
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            res["cudnn_version"] = torch.backends.cudnn.version()

        if torch.cuda.is_available():
            res["device_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                total_mem_gb = prop.total_memory / (1024 ** 3)
                free_mem_gb, total_mem_query_gb = torch.cuda.mem_get_info(i)
                free_mem_gb /= (1024 ** 3)
                total_mem_query_gb /= (1024 ** 3)
                res["devices"].append({
                    "device_index": i,
                    "name": prop.name,
                    "major_compute_capability": prop.major,
                    "minor_compute_capability": prop.minor,
                    "total_memory_gb": round(total_mem_gb, 3),
                    "free_memory_gb": round(free_mem_gb, 3),
                    "multi_processor_count": prop.multi_processor_count,
                    "is_8gb_vram_class": (7.0 <= total_mem_gb <= 8.5)
                })
    except Exception as e:
        res["errors"].append(f"Torch/CUDA check error: {type(e).__name__}: {e}")
    return res

def check_nvidia_smi_and_nvml() -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "nvidia_smi_available": False,
        "nvidia_smi_raw": None,
        "driver_version": None,
        "nvml_available": False,
        "nvml_devices": [],
        "errors": []
    }
    # Check nvidia-smi
    try:
        smi_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total,memory.free,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if smi_out.returncode == 0:
            res["nvidia_smi_available"] = True
            res["nvidia_smi_raw"] = smi_out.stdout.strip()
        else:
            res["errors"].append(f"nvidia-smi returned code {smi_out.returncode}: {smi_out.stderr.strip()}")
    except Exception as e:
        res["errors"].append(f"nvidia-smi check error: {type(e).__name__}: {e}")

    # Check pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        res["nvml_available"] = True
        res["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            res["nvml_devices"].append({
                "index": i,
                "name": name,
                "total_mb": round(mem_info.total / (1024 ** 2), 2),
                "free_mb": round(mem_info.free / (1024 ** 2), 2),
                "used_mb": round(mem_info.used / (1024 ** 2), 2)
            })
        pynvml.nvmlShutdown()
    except ImportError:
        res["errors"].append("pynvml is not installed in current environment.")
    except Exception as e:
        res["errors"].append(f"pynvml error: {type(e).__name__}: {e}")
        
    return res

def check_python_packages() -> Dict[str, Any]:
    packages_to_check = [
        "transformers",
        "bitsandbytes",
        "accelerate",
        "sentence_transformers",
        "rank_bm25",
        "datasets",
        "sklearn",
        "networkx",
        "torch",
        "numpy",
        "scipy",
        "tqdm",
        "pydantic"
    ]
    results: Dict[str, Any] = {}
    for pkg in packages_to_check:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed_unknown_version")
            results[pkg] = {
                "installed": True,
                "version": str(version),
                "path": str(getattr(mod, "__file__", "builtin_or_unknown"))
            }
        except ImportError as e:
            results[pkg] = {
                "installed": False,
                "version": None,
                "error": str(e)
            }
        except Exception as e:
            results[pkg] = {
                "installed": False,
                "error": f"{type(e).__name__}: {e}"
            }
    return results

def find_candidate_model_caches() -> Dict[str, Any]:
    """Search HF cache, ModelScope cache, and common local directories for model weights."""
    results: Dict[str, Any] = {
        "hf_cache_dir": None,
        "ms_cache_dir": None,
        "found_models": [],
        "scan_warnings": []
    }
    
    user_home = Path.home()
    hf_cache = os.getenv("HF_HOME")
    if not hf_cache:
        hf_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if not hf_cache:
        hf_cache = str(user_home / ".cache" / "huggingface" / "hub")
    results["hf_cache_dir"] = hf_cache

    ms_cache = os.getenv("MODELSCOPE_CACHE")
    if not ms_cache:
        ms_cache = str(user_home / ".cache" / "modelscope" / "hub")
    results["ms_cache_dir"] = ms_cache

    search_dirs = [
        Path(hf_cache),
        Path(ms_cache),
        Path("F:/model_weights"),
        Path("F:/models"),
        Path("F:/weights"),
        Path("F:/huggingface"),
        Path("F:/pretrained_models"),
        Path("D:/model_weights"),
        Path("D:/models"),
        Path("D:/weights"),
        Path("C:/models"),
        Path.cwd() / "models",
        Path.cwd() / "weights",
        Path.cwd().parent / "models",
        Path.cwd().parent.parent / "models",
    ]

    target_keywords = ["qwen", "bge", "llama", "chatglm", "mistral", "baichuan", "embedding"]

    checked_dirs = set()
    for base_dir in search_dirs:
        try:
            if not base_dir.exists():
                continue
            canonical_path = str(base_dir.resolve())
            if canonical_path in checked_dirs:
                continue
            checked_dirs.add(canonical_path)

            for item in base_dir.rglob("*"):
                try:
                    rel = item.relative_to(base_dir)
                    if len(rel.parts) > 3:
                        continue
                except ValueError as rel_err:
                    results["scan_warnings"].append(f"relative_to error on {item}: {rel_err}")
                    continue
                name_lower = item.name.lower()
                if any(kw in name_lower for kw in target_keywords):
                    if item.is_dir() and any((item / f).exists() for f in ["config.json", "model.safetensors", "pytorch_model.bin"]):
                        results["found_models"].append({
                            "name": item.name,
                            "path": str(item.resolve()),
                            "has_config": (item / "config.json").exists(),
                            "has_safetensors": (item / "model.safetensors").exists() or len(list(item.glob("*.safetensors"))) > 0
                        })
        except Exception as scan_err:
            results["scan_warnings"].append(f"Error scanning {base_dir}: {scan_err}")

    return results

def find_candidate_datasets() -> Dict[str, Any]:
    """Search for QA datasets like HotpotQA, PopQA, 2WikiMultiHopQA."""
    results: Dict[str, Any] = {
        "searched_keywords": ["hotpot", "popqa", "2wiki", "multihop", "trivia", "nq", "rag"],
        "found_datasets": [],
        "scan_warnings": []
    }
    
    search_dirs = [
        Path.cwd(),
        Path.cwd() / "data",
        Path.cwd() / "datasets",
        Path.cwd().parent,
        Path.cwd().parent / "data",
        Path("F:/datasets"),
        Path("F:/data"),
        Path("D:/datasets"),
        Path.home() / ".cache" / "huggingface" / "datasets"
    ]

    checked_dirs = set()
    for base_dir in search_dirs:
        try:
            if not base_dir.exists():
                continue
            canonical_path = str(base_dir.resolve())
            if canonical_path in checked_dirs:
                continue
            checked_dirs.add(canonical_path)

            for item in base_dir.rglob("*"):
                try:
                    rel = item.relative_to(base_dir)
                    if len(rel.parts) > 3:
                        continue
                except ValueError as rel_err:
                    results["scan_warnings"].append(f"relative_to error on {item}: {rel_err}")
                    continue
                name_lower = item.name.lower()
                if any(kw in name_lower for kw in results["searched_keywords"]):
                    if item.is_file() and item.suffix in [".json", ".jsonl", ".csv", ".parquet", ".arrow"]:
                        results["found_datasets"].append({
                            "name": item.name,
                            "path": str(item.resolve()),
                            "size_bytes": item.stat().st_size
                        })
                    elif item.is_dir() and (len(list(item.glob("*.json"))) > 0 or len(list(item.glob("*.jsonl"))) > 0 or len(list(item.glob("*.parquet"))) > 0):
                        results["found_datasets"].append({
                            "name": item.name,
                            "path": str(item.resolve()),
                            "is_dir": True
                        })
        except Exception as scan_err:
            results["scan_warnings"].append(f"Error scanning {base_dir}: {scan_err}")

    return results

def main():
    print("=" * 60)
    print("U-GraphRAG Step 1: Environment & Hardware Audit")
    print("=" * 60)

    report = {
        "system": check_python_and_system(),
        "cuda_torch": check_cuda_and_torch(),
        "nvidia_smi_nvml": check_nvidia_smi_and_nvml(),
        "packages": check_python_packages(),
        "models": find_candidate_model_caches(),
        "datasets": find_candidate_datasets()
    }

    # Pretty print summary to console
    print("\n[1] Python & OS:")
    print(f"  Python: {report['system']['python_version'].split()[0]} ({report['system']['python_executable']})")
    print(f"  Platform: {report['system']['platform']}")

    print("\n[2] CUDA & PyTorch:")
    print(f"  PyTorch Version: {report['cuda_torch']['torch_version']}")
    print(f"  CUDA Available: {report['cuda_torch']['cuda_available']}")
    print(f"  CUDA Version: {report['cuda_torch']['cuda_version']}")
    print(f"  cuDNN Version: {report['cuda_torch']['cudnn_version']}")
    if report['cuda_torch']['errors']:
        print(f"  Errors encountered: {report['cuda_torch']['errors']}")
    for dev in report['cuda_torch']['devices']:
        print(f"  GPU [{dev['device_index']}]: {dev['name']}")
        print(f"    Total Memory: {dev['total_memory_gb']} GB")
        print(f"    Free Memory:  {dev['free_memory_gb']} GB")
        print(f"    8GB Class Check: {'PASS (Target Hardware Confirmed)' if dev['is_8gb_vram_class'] else 'FAIL/NON-8GB'}")

    print("\n[3] NVIDIA-SMI & NVML:")
    print(f"  NVIDIA-SMI Available: {report['nvidia_smi_nvml']['nvidia_smi_available']}")
    if report['nvidia_smi_nvml']['nvidia_smi_raw']:
        print(f"  Query output: {report['nvidia_smi_nvml']['nvidia_smi_raw']}")
    print(f"  NVML Available: {report['nvidia_smi_nvml']['nvml_available']}")
    if report['nvidia_smi_nvml'].get('driver_version'):
        print(f"  Driver Version: {report['nvidia_smi_nvml']['driver_version']}")
    for dev in report['nvidia_smi_nvml']['nvml_devices']:
        print(f"  NVML GPU [{dev['index']}]: {dev['name']} Total: {dev['total_mb']} MB, Free: {dev['free_mb']} MB")
    if report['nvidia_smi_nvml']['errors']:
        print(f"  NVML Errors: {report['nvidia_smi_nvml']['errors']}")

    print("\n[4] Core Packages Audit:")
    for pkg, info in report['packages'].items():
        if info.get('installed'):
            status = f"INSTALLED (v{info.get('version')})"
        else:
            status = f"NOT INSTALLED ({info.get('error', 'unknown')})"
        print(f"  - {pkg:<22}: {status}")

    print("\n[5] Model Weights Cache Detection:")
    print(f"  HF Cache Dir: {report['models']['hf_cache_dir']}")
    print(f"  MS Cache Dir: {report['models']['ms_cache_dir']}")
    print(f"  Found Models Count: {len(report['models']['found_models'])}")
    for m in report['models']['found_models'][:10]:
        print(f"  - {m['name']} ({m['path']})")

    print("\n[6] QA Datasets Detection:")
    print(f"  Found Datasets Count: {len(report['datasets']['found_datasets'])}")
    for d in report['datasets']['found_datasets'][:10]:
        print(f"  - {d['name']} ({d['path']})")

    if report['models']['scan_warnings']:
        print(f"  Model scan warnings: {len(report['models']['scan_warnings'])}")
    if report['datasets']['scan_warnings']:
        print(f"  Dataset scan warnings: {len(report['datasets']['scan_warnings'])}")

    print(f"\nAudit completed successfully. All hardware & package details probed.")

if __name__ == "__main__":
    main()
