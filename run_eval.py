"""Wrapper to fix WORK_DIR permission issue and run experiments."""
import sys
import argparse
from pathlib import Path
import harness.evaluate as ev

ev.WORK_DIR = Path("/home/researcher/autobench_work/work_tmp")
ev.WORK_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    args = parser.parse_args()
    if args.baseline_only:
        ev.generate_dense_baseline(args.model, args.mode)
    else:
        ev.run_experiment(eval_mode=args.mode, model_id=args.model)
