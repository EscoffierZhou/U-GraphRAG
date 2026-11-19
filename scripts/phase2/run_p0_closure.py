"""
run_p0_closure.py
=================
Sequential GPU Orchestrator for MileStone 2 Closure Experiments (P0-1 .. P0-4).
Runs each P0 work package in its own clean Python subprocess so PyTorch CUDA allocations
are completely released between stages (safe on 8GB VRAM RTX 4070 Laptop GPU), and
verifies V5_FROZEN SHA256 integrity before and after every stage.
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

from scripts.phase2.common import WP_DIRS, setup_phase2_logger, verify_v5_frozen_checksums

logger = setup_phase2_logger("phase2.p0_orchestrator")


def _run_stage(name: str, cmd: List[str]) -> float:
    logger.info(f"=== Starting {name}: {' '.join(cmd)} ===")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Stage {name} failed with exit code {proc.returncode} after {elapsed:.1f}s")
    logger.info(f"=== Completed {name} in {elapsed:.1f}s ===")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MileStone 2 P0-1..P0-4 closure suite sequentially.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["p0_1", "p0_2", "p0_3", "p0_4"],
        choices=["p0_1", "p0_2", "p0_3", "p0_4"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=str(WP_DIRS["P0"]))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    setup_phase2_logger("phase2.p0_orchestrator", out_dir / "p0_orchestrator.log")
    v5_info = verify_v5_frozen_checksums()
    logger.info(f"Verified V5_FROZEN archive: {v5_info}")

    py = sys.executable
    p2_dir = PROJECT_ROOT / "scripts" / "phase2"
    dry_flag = ["--dry-run"] if args.dry_run else []

    cmds: Dict[str, List[str]] = {
        "p0_1": [py, str(p2_dir / "p0_1_m2_conflict_and_scale.py"), "--out-dir", str(out_dir), *dry_flag],
        "p0_2": [py, str(p2_dir / "p0_2_m1_signal_unification.py"), "--out-dir", str(out_dir), *dry_flag],
        "p0_3": [py, str(p2_dir / "p0_3_m3_adaptive_gating.py"), "--out-dir", str(out_dir), *dry_flag],
        "p0_4": [py, str(p2_dir / "p0_4_wp5_expanded_stress.py"), "--out-dir", str(out_dir), *dry_flag],
    }

    timings: Dict[str, float] = {}
    t_all = time.time()
    for st in args.stages:
        verify_v5_frozen_checksums()
        timings[st] = round(_run_stage(st, cmds[st]), 2)
    verify_v5_frozen_checksums()
    logger.info(
        f"All selected P0 closure stages ({args.stages}) finished cleanly in {time.time() - t_all:.1f}s. "
        f"Timings={timings}"
    )


if __name__ == "__main__":
    main()
