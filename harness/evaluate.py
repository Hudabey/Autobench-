"""
Evaluate — orchestrates one complete experiment cycle.

This is the equivalent of Karpathy's training loop, but for benchmarking.
One call to run_experiment() = one data point on the Pareto frontier.

Flow:
  1. Load + validate config
  2. Load agent's attention patch
  3. Apply patch to model
  4. Generate videos (fixed prompts, fixed seeds)
  5. Compute all metrics against dense baseline
  6. Log results to results.jsonl
  7. Update Pareto frontier
  8. Restore model to dense attention

🔒 FIXED — agent cannot modify this file.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import traceback
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import torch
import numpy as np

# Harness imports
from harness.config_schema import load_config, ExperimentConfig
from harness.model import (
    load_pipeline, restore_attention, reset_attention_timing,
    get_attention_timing, generate_videos, get_attention_info,
    cleanup_gpu,
)
from harness.metrics import evaluate_experiment, AggregateMetrics


# ──────────────────────────────────────────────────────────────
# Paths (relative to project root)
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "experiments" / "config.yaml"
PATCH_PATH = PROJECT_ROOT / "experiments" / "attention_patch.py"
RESULTS_LOG = PROJECT_ROOT / "results" / "results.jsonl"
PARETO_PATH = PROJECT_ROOT / "results" / "pareto.json"
PROMPTS_PATH = PROJECT_ROOT / "harness" / "prompts.json"
REFERENCE_DIR = PROJECT_ROOT / "reference_videos"
WORK_DIR = Path("/tmp/autobench_work")


# ──────────────────────────────────────────────────────────────
# Prompt loading
# ──────────────────────────────────────────────────────────────

def load_prompts(mode: str = "full") -> list[dict]:
    """Load evaluation prompts. mode='fast' uses 9-prompt subset."""
    with open(PROMPTS_PATH) as f:
        data = json.load(f)

    all_prompts = data["prompts"]

    if mode == "fast":
        fast_ids = set(data["metadata"]["fast_eval_subset"])
        return [p for p in all_prompts if p["id"] in fast_ids]
    return all_prompts


# ──────────────────────────────────────────────────────────────
# Dense baseline generation
# ──────────────────────────────────────────────────────────────

def generate_dense_baseline(
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    mode: str = "full",
):
    """Generate reference videos with dense (vanilla) attention.
    Run this ONCE before any experiments.

    Always generates in 'full' mode regardless of argument, to ensure
    all prompt IDs have references. This prevents the fast/full mismatch bug.
    """
    if REFERENCE_DIR.exists() and any(REFERENCE_DIR.glob("video_p*.pt")):
        n_existing = len(list(REFERENCE_DIR.glob("video_p*.pt")))
        print(f"[evaluate] Reference videos already exist ({n_existing} files)")
        print(f"[evaluate] Delete {REFERENCE_DIR} to regenerate")
        return

    # Always generate full baseline — fast experiments will use a subset
    baseline_mode = "full"

    print("=" * 60)
    print("[evaluate] GENERATING DENSE BASELINE")
    print(f"[evaluate] Mode forced to '{baseline_mode}' (all prompts need references)")
    print("[evaluate] This runs ONCE and creates the reference videos")
    print("[evaluate] All sparse methods will be compared against these")
    print("=" * 60)

    pipe = load_pipeline(model_id)
    prompts_data = load_prompts(baseline_mode)

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    video_paths, timing = generate_videos(
        pipe,
        prompts_data,  # pass full dicts, not just strings
        num_inference_steps=50,
        guidance_scale=7.5,
        video_length=81,
        resolution=(480, 832),
        seed=42,
        output_dir=str(REFERENCE_DIR),
    )

    # Save baseline timing info
    baseline_info = {
        "model_id": model_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_videos": len(video_paths),
        "timing": timing,
        "attention_info": get_attention_info(pipe),
        "mode": baseline_mode,
    }
    with open(REFERENCE_DIR / "baseline_info.json", "w") as f:
        json.dump(baseline_info, f, indent=2)

    # Compute standalone quality metrics (ImageReward, HPSv2) for calibration
    # These are saved so compute_composite_score can normalize relative to baseline
    print("[evaluate] Computing baseline quality metrics for calibration...")
    from harness.metrics import compute_image_reward_frames, compute_hpsv2_frames
    ir_scores = []
    hpsv2_scores = []
    for vp, pd in zip(video_paths, prompts_data):
        frames = torch.load(vp, weights_only=True)
        ir = compute_image_reward_frames(frames, pd["prompt"])
        ir_scores.append(ir)
        hp = compute_hpsv2_frames(frames, pd["prompt"])
        hpsv2_scores.append(hp)

    baseline_quality = {
        "image_reward": float(np.mean(ir_scores)),
        "hpsv2": float(np.mean(hpsv2_scores)),
        # Reference-comparison metrics have known ideal values for baseline
        "fid": 0.0,       # self-reference
        "ssim": 1.0,      # self-reference
        "psnr": 50.0,     # effectively perfect
        "lpips": 0.0,     # self-reference
    }
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "baseline_quality.json", "w") as f:
        json.dump(baseline_quality, f, indent=2)

    print(f"[evaluate] Baseline ImageReward: {baseline_quality['image_reward']:.4f}")
    print(f"[evaluate] Baseline HPSv2: {baseline_quality['hpsv2']:.4f}")
    print(f"[evaluate] Dense baseline complete: {len(video_paths)} videos")
    print(f"[evaluate] Total generation time: {timing['total_generation_s']:.1f}s")


# ──────────────────────────────────────────────────────────────
# Load agent's attention patch
# ──────────────────────────────────────────────────────────────

def load_attention_patch(patch_path: Path):
    """Dynamically load the agent's attention_patch.py module.

    The patch module must define:
        def create_patch(config: ExperimentConfig) -> callable
    Where the callable has signature:
        patch_fn(original_forward, layer_name, module) -> new_forward
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("attention_patch", str(patch_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "create_patch"):
        raise AttributeError(
            f"attention_patch.py must define create_patch(config) -> patch_fn\n"
            f"  patch_fn(original_forward, layer_name, module) -> new_forward"
        )

    return module.create_patch


# ──────────────────────────────────────────────────────────────
# Pareto frontier tracking
# ──────────────────────────────────────────────────────────────

def update_pareto_frontier(new_result: dict) -> bool:
    """Update the Pareto frontier with a new result.
    Returns True if the new result is Pareto-optimal.

    Pareto-optimal = not dominated on (latency, quality) axes.
    A result is dominated if another result is both faster AND higher quality.
    """
    frontier = []
    if PARETO_PATH.exists():
        with open(PARETO_PATH) as f:
            frontier = json.load(f)

    new_latency = new_result.get("latency", {}).get("total_generation_s", float("inf"))
    new_quality = new_result.get("quality", {}).get("composite_score", 0.0)

    # Check if new result is dominated by any existing point
    is_dominated = False
    for point in frontier:
        p_latency = point.get("latency", {}).get("total_generation_s", float("inf"))
        p_quality = point.get("quality", {}).get("composite_score", 0.0)
        # Dominated if existing point is both faster and better quality
        if p_latency <= new_latency and p_quality >= new_quality:
            if p_latency < new_latency or p_quality > new_quality:
                is_dominated = True
                break

    if not is_dominated:
        # Remove points dominated by new result
        frontier = [
            p for p in frontier
            if not (new_latency <= p.get("latency", {}).get("total_generation_s", float("inf"))
                    and new_quality >= p.get("quality", {}).get("composite_score", 0.0)
                    and (new_latency < p.get("latency", {}).get("total_generation_s", float("inf"))
                         or new_quality > p.get("quality", {}).get("composite_score", 0.0)))
        ]
        frontier.append(new_result)

        with open(PARETO_PATH, "w") as f:
            json.dump(frontier, f, indent=2)

    return not is_dominated


# ──────────────────────────────────────────────────────────────
# Generate experiment ID
# ──────────────────────────────────────────────────────────────

def get_experiment_count() -> int:
    """Count existing experiments."""
    if not RESULTS_LOG.exists():
        return 0
    with open(RESULTS_LOG) as f:
        return sum(1 for line in f if line.strip())


# ──────────────────────────────────────────────────────────────
# Main experiment runner
# ──────────────────────────────────────────────────────────────

def run_experiment(
    eval_mode: str = "full",      # "fast" or "full"
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
) -> dict:
    """Run one complete experiment cycle.

    This is the function the agent triggers after editing
    config.yaml and attention_patch.py.

    Returns the result dict (also written to results.jsonl).
    """
    exp_num = get_experiment_count() + 1
    print("\n" + "=" * 60)
    print(f"[evaluate] EXPERIMENT #{exp_num}")
    print("=" * 60)

    # ── Step 1: Load and validate config ──
    print("\n[Step 1] Loading config...")
    try:
        cfg = load_config(str(CONFIG_PATH))
    except Exception as e:
        return _log_failure(exp_num, "config_invalid", str(e))

    print(f"  Method:     {cfg.method}")
    print(f"  Name:       {cfg.experiment_name}")
    print(f"  Hypothesis: {cfg.hypothesis}")
    print(f"  Phase:      {cfg.phase}")

    # ── Step 2: Ensure baseline exists ──
    if not REFERENCE_DIR.exists() or not any(REFERENCE_DIR.glob("video_p*.pt")):
        print("\n[Step 2] No baseline found — generating dense baseline first...")
        generate_dense_baseline(model_id, eval_mode)

    # ── Step 3: Load pipeline ──
    print("\n[Step 3] Loading pipeline...")
    pipe = load_pipeline(model_id)
    restore_attention(pipe)  # always start clean

    # ── Step 4: Apply attention patch (if not dense baseline) ──
    if cfg.method != "dense":
        print(f"\n[Step 4] Loading attention patch for '{cfg.method}'...")
        if not PATCH_PATH.exists():
            return _log_failure(exp_num, "no_patch_file",
                                f"experiments/attention_patch.py not found", cfg)
        try:
            create_patch = load_attention_patch(PATCH_PATH)
            patch_fn = create_patch(cfg)
            from harness.model import patch_attention
            patch_attention(pipe, patch_fn, cfg.params.apply_to_layers)
        except Exception as e:
            restore_attention(pipe)
            return _log_failure(exp_num, "patch_failed", traceback.format_exc(), cfg)
    else:
        print("\n[Step 4] Dense baseline — no patch applied")

    # ── Step 5: Generate videos ──
    print("\n[Step 5] Generating videos...")
    prompts_data = load_prompts(eval_mode)
    prompt_texts = [p["prompt"] for p in prompts_data]
    prompt_ids = [p["id"] for p in prompts_data]

    gen_dir = WORK_DIR / f"exp_{exp_num:04d}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    reset_attention_timing()

    try:
        video_paths, gen_timing = generate_videos(
            pipe,
            prompts_data,  # pass full dicts (not just strings) for ID-keyed filenames
            num_inference_steps=cfg.inference.num_inference_steps,
            guidance_scale=cfg.inference.guidance_scale,
            video_length=cfg.inference.video_length,
            resolution=cfg.inference.resolution,
            seed=cfg.inference.seed,
            output_dir=str(gen_dir),
        )
    except Exception as e:
        restore_attention(pipe)
        cleanup_gpu()
        return _log_failure(exp_num, "generation_failed", traceback.format_exc(), cfg)

    attn_timing = get_attention_timing()

    # ── Step 6: Compute metrics ──
    print("\n[Step 6] Computing metrics...")

    # Match generated videos to references BY PROMPT ID (not by position)
    # Generated: video_p003.pt, video_p005.pt, ...
    # Reference: video_p000.pt, video_p001.pt, ..., video_p024.pt
    matched_gen = []
    matched_ref = []
    matched_prompts = []
    for pid, gen_path in zip(prompt_ids, video_paths):
        ref_path = REFERENCE_DIR / f"video_p{pid:03d}.pt"
        if ref_path.exists():
            matched_gen.append(gen_path)
            matched_ref.append(str(ref_path))
            idx = next(i for i, p in enumerate(prompts_data) if p["id"] == pid)
            matched_prompts.append(prompt_texts[idx])
        else:
            print(f"[evaluate] WARNING: No reference for prompt {pid}, skipping")

    n = len(matched_gen)
    if n == 0:
        restore_attention(pipe)
        cleanup_gpu()
        return _log_failure(exp_num, "no_references",
                            "No matching reference videos found. Run baseline first.",
                            cfg)

    try:
        metrics = evaluate_experiment(
            video_paths=matched_gen,
            reference_paths=matched_ref,
            prompts=matched_prompts,
            work_dir=str(WORK_DIR / "eval_tmp"),
            mode=eval_mode,
        )
    except Exception as e:
        restore_attention(pipe)
        cleanup_gpu()
        return _log_failure(exp_num, "metrics_failed", traceback.format_exc(), cfg)

    # ── Step 7: Restore attention ──
    restore_attention(pipe)

    # ── Step 8: Load dense baseline timing for speedup calc ──
    baseline_info_path = REFERENCE_DIR / "baseline_info.json"
    dense_gen_time = None
    if baseline_info_path.exists():
        with open(baseline_info_path) as f:
            bi = json.load(f)
            dense_gen_time = bi.get("timing", {}).get("total_generation_s")

    speedup = None
    if dense_gen_time and gen_timing["total_generation_s"] > 0:
        speedup = dense_gen_time / gen_timing["total_generation_s"]

    # ── Step 9: Build result dict ──
    result = {
        "experiment_id": f"exp_{exp_num:04d}",
        "experiment_name": cfg.experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": cfg.method,
        "config": {
            "params": {k: v for k, v in cfg.params.__dict__.items() if v is not None},
            "inference": cfg.inference.__dict__,
        },
        "hypothesis": cfg.hypothesis,
        "phase": cfg.phase,

        "latency": {
            "total_generation_s": gen_timing["total_generation_s"],
            "avg_per_video_s": gen_timing["avg_per_video_s"],
            "ms_per_frame": gen_timing["ms_per_frame"],
            "attention_total_ms": attn_timing.total_attention_ms,
            "attention_avg_ms_per_call": attn_timing.avg_ms_per_call,
            "attention_num_calls": attn_timing.num_calls,
            "speedup_vs_dense": speedup,
        },

        "quality": {
            "fid": metrics.fid,
            "ssim": metrics.ssim_mean,
            "psnr": metrics.psnr_mean,
            "lpips": metrics.lpips_mean,
            "image_reward": metrics.image_reward_mean,
            "hpsv2": metrics.hpsv2_mean,
            "composite_score": metrics.composite_score,
        },

        "quality_std": {
            "ssim_std": metrics.ssim_std,
            "psnr_std": metrics.psnr_std,
            "lpips_std": metrics.lpips_std,
        },

        "meta": {
            "eval_mode": eval_mode,
            "num_videos": n,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "vram_peak_gb": (torch.cuda.max_memory_allocated() / 1e9
                            if torch.cuda.is_available() else 0),
        },

        "status": "success",
    }

    # ── Step 10: Update Pareto frontier ──
    is_pareto = update_pareto_frontier(result)
    result["pareto_optimal"] = is_pareto

    # ── Step 11: Log result ──
    RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_LOG, "a") as f:
        f.write(json.dumps(result) + "\n")

    # ── Step 12: Print summary ──
    print("\n" + "=" * 60)
    print(f"[evaluate] EXPERIMENT #{exp_num} COMPLETE")
    print(f"  Method:          {cfg.method}")
    print(f"  Latency:         {gen_timing['total_generation_s']:.1f}s total")
    if speedup:
        print(f"  Speedup:         {speedup:.2f}x vs dense")
    print(f"  FID:             {metrics.fid:.2f}")
    print(f"  SSIM:            {metrics.ssim_mean:.4f}")
    print(f"  LPIPS:           {metrics.lpips_mean:.4f}")
    print(f"  ImageReward:     {metrics.image_reward_mean:.4f}")
    print(f"  HPSv2:           {metrics.hpsv2_mean:.4f}")
    print(f"  Composite:       {metrics.composite_score:.4f}")
    print(f"  Pareto-optimal:  {'YES ★' if is_pareto else 'no'}")
    print("=" * 60)

    # Cleanup temp files
    if (WORK_DIR / "eval_tmp").exists():
        shutil.rmtree(WORK_DIR / "eval_tmp", ignore_errors=True)
    cleanup_gpu()

    return result


def _log_failure(exp_num: int, error_type: str, error_msg: str,
                 cfg: ExperimentConfig = None) -> dict:
    """Log a failed experiment with full context for debugging."""
    result = {
        "experiment_id": f"exp_{exp_num:04d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "error_type": error_type,
        "error_message": error_msg[:2000],  # truncate huge tracebacks
    }

    # Include config context if available (Bug 16 fix)
    if cfg is not None:
        result["method"] = cfg.method
        result["experiment_name"] = cfg.experiment_name
        result["hypothesis"] = cfg.hypothesis
        result["phase"] = cfg.phase
        result["config"] = {
            "params": {k: v for k, v in cfg.params.__dict__.items() if v is not None},
            "inference": cfg.inference.__dict__,
        }

    RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_LOG, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\n[evaluate] EXPERIMENT #{exp_num} FAILED: {error_type}")
    if cfg:
        print(f"  Method: {cfg.method} | Name: {cfg.experiment_name}")
    print(f"  {error_msg[:200]}")

    return result


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run one autobench experiment")
    parser.add_argument("--mode", choices=["fast", "full"], default="full",
                        help="Evaluation mode: 'fast' (9 prompts, FID+LPIPS) or 'full' (25 prompts, all metrics)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only generate dense baseline, then exit")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B",
                        help="Model ID")
    args = parser.parse_args()

    if args.baseline_only:
        generate_dense_baseline(args.model, args.mode)
    else:
        run_experiment(eval_mode=args.mode, model_id=args.model)
