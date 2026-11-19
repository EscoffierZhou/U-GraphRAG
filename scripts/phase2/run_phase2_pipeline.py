"""
run_phase2_pipeline.py
======================
Sequential Stage Orchestrator for MileStone 2 (WP0 -> WP5) on a single 8GB VRAM GPU.
Each work package runs in its own clean Python subprocess so PyTorch CUDA context & memory
are completely reclaimed between stages.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase2.common import MILE2_DIR, setup_phase2_logger, verify_v5_frozen_checksums

logger = setup_phase2_logger("phase2.orchestrator")


def _run_stage(stage_name: str, cmd: List[str]) -> None:
    logger.info(f"=== Starting Stage {stage_name}: {' '.join(cmd)} ===")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Stage {stage_name} failed with exit code {proc.returncode} after {elapsed:.1f}s")
    logger.info(f"=== Completed Stage {stage_name} in {elapsed:.1f}s ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MileStone 2 Phase II stages sequentially (8GB safe).")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["wp0", "wp1", "wp2", "wp3", "wp4", "wp5"],
        choices=["wp0", "wp1", "wp2", "wp3", "wp4", "wp5"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download-datasets", action="store_true", help="Pass --download to WP0 cache_builder")
    parser.add_argument("--hotpot-limit", type=int, default=300)
    parser.add_argument("--wp4-models", nargs="+", default=["qwen3b", "qwen7b"])
    parser.add_argument("--wp4-datasets", nargs="+", default=["hotpotqa", "popqa", "2wikimultihop", "musique"])
    args = parser.parse_args()

    setup_phase2_logger("phase2.orchestrator", MILE2_DIR / "phase2_orchestrator.log")
    v5_info = verify_v5_frozen_checksums()
    logger.info(f"V5_FROZEN integrity verified: {v5_info}")

    py = sys.executable
    p2_dir = PROJECT_ROOT / "scripts" / "phase2"

    stage_cmds: Dict[str, List[str]] = {
        "wp0": [
            py, str(p2_dir / "cache_builder.py"),
            "--hotpot-limit", str(args.hotpot_limit),
            *(["--download"] if args.download_datasets else []),
            *(["--dry-run"] if args.dry_run else []),
        ],
        "wp1": [
            py, str(p2_dir / "wp1_eu_validation.py"),
            "--models", *args.wp4_models[:2],
            *(["--dry-run"] if args.dry_run else []),
        ],
        "wp2": [
            py, str(p2_dir / "wp2_m2_ablation.py"),
            "--dataset", "hotpotqa", "--split", "test", "--model", "qwen3b",
            *(["--dry-run"] if args.dry_run else []),
        ],
        "wp3": [
            py, str(p2_dir / "wp3_m3_calibration.py"),
            "--dataset", "hotpotqa", "--model", "qwen3b",
            *(["--dry-run"] if args.dry_run else []),
        ],
        "wp4": [
            py, str(p2_dir / "wp4_generalization.py"),
            "--models", *args.wp4_models,
            "--datasets", *args.wp4_datasets,
            "--split", "test",
            *(["--dry-run"] if args.dry_run else []),
        ],
        "wp5": [
            py, str(p2_dir / "wp5_stress_and_profile.py"),
            "--model", "qwen3b",
            *(["--dry-run"] if args.dry_run else []),
        ],
    }

    t_all = time.time()
    for st in args.stages:
        verify_v5_frozen_checksums()
        _run_stage(st, stage_cmds[st])
    verify_v5_frozen_checksums()
    logger.info(f"All selected Phase II stages ({args.stages}) completed in {time.time() - t_all:.1f}s.")


if __name__ == "__main__":
    main()
