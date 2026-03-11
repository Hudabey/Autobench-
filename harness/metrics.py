"""
Metrics — computes all quality and latency measurements.

Metric categories:
  Semantic & Preference:  FID↓, ImageReward↑, MPS↑, HPSv2↑
  Fidelity:               SSIM↑, PSNR↑, LPIPS↓

All metrics are computed frame-by-frame against dense-baseline reference
videos, then averaged. This gives per-video and aggregate scores.

🔒 FIXED — agent cannot modify this file.
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class VideoMetrics:
    """Metrics for a single video."""
    prompt_idx: int = -1
    prompt: str = ""
    # Fidelity (vs reference)
    ssim: float = 0.0          # ↑ better
    psnr: float = 0.0          # ↑ better
    lpips: float = 0.0         # ↓ better
    # Semantic / Preference (standalone quality)
    fid: float = 0.0           # ↓ better (computed per-video is approximate)
    image_reward: float = 0.0  # ↑ better
    hpsv2: float = 0.0         # ↑ better


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all videos in an experiment."""
    # Fidelity
    ssim_mean: float = 0.0
    ssim_std: float = 0.0
    psnr_mean: float = 0.0
    psnr_std: float = 0.0
    lpips_mean: float = 0.0
    lpips_std: float = 0.0
    # Semantic
    fid: float = 0.0           # FID computed across all frames
    image_reward_mean: float = 0.0
    hpsv2_mean: float = 0.0
    # Composite
    composite_score: float = 0.0
    # Per-video detail
    per_video: list = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Metric implementations
# ──────────────────────────────────────────────────────────────

# Lazy-loaded metric models (heavy, load once)
_lpips_model = None
_image_reward_model = None
_hpsv2_model = None


def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").cuda().eval()
    return _lpips_model


def _get_image_reward():
    global _image_reward_model
    if _image_reward_model is None:
        try:
            import ImageReward as IR
            _image_reward_model = IR.load("ImageReward-v1.0")
        except (ImportError, Exception) as e:
            print(f"[metrics] WARNING: ImageReward unavailable ({e}), scores will be 0.0")
            _image_reward_model = "unavailable"
    return _image_reward_model


def compute_ssim_frames(frames_a: torch.Tensor, frames_b: torch.Tensor) -> float:
    """Compute mean SSIM between two video tensors.
    frames shape: (T, C, H, W) in [0, 1]"""
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(frames_a.device)

    scores = []
    T = min(len(frames_a), len(frames_b))
    for t in range(T):
        score = ssim_fn(frames_a[t:t+1], frames_b[t:t+1])
        scores.append(score.item())
    return float(np.mean(scores))


def compute_psnr_frames(frames_a: torch.Tensor, frames_b: torch.Tensor) -> float:
    """Compute mean PSNR between two video tensors."""
    from torchmetrics.image import PeakSignalNoiseRatio
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(frames_a.device)

    scores = []
    T = min(len(frames_a), len(frames_b))
    for t in range(T):
        score = psnr_fn(frames_a[t:t+1], frames_b[t:t+1])
        scores.append(score.item())
    return float(np.mean(scores))


def compute_lpips_frames(frames_a: torch.Tensor, frames_b: torch.Tensor) -> float:
    """Compute mean LPIPS between two video tensors.
    frames shape: (T, C, H, W) in [0, 1]"""
    model = _get_lpips()

    scores = []
    T = min(len(frames_a), len(frames_b))
    with torch.no_grad():
        for t in range(T):
            # LPIPS expects [-1, 1]
            a = frames_a[t:t+1].cuda() * 2 - 1
            b = frames_b[t:t+1].cuda() * 2 - 1
            score = model(a, b)
            scores.append(score.item())
    return float(np.mean(scores))


def compute_fid(
    generated_frames_dir: str,
    reference_frames_dir: str,
    device: str = "cuda",
) -> float:
    """Compute FID between generated and reference frame distributions.

    Both dirs should contain .png frames extracted from videos.
    Uses clean-fid for more accurate computation.
    """
    try:
        from cleanfid import fid as cleanfid
        score = cleanfid.compute_fid(
            generated_frames_dir,
            reference_frames_dir,
            device=torch.device(device),
        )
        return float(score)
    except ImportError:
        # Fallback: pytorch-fid
        from pytorch_fid import fid_score
        score = fid_score.calculate_fid_given_paths(
            [generated_frames_dir, reference_frames_dir],
            batch_size=50,
            device=device,
            dims=2048,
        )
        return float(score)


def compute_image_reward_frames(
    frames: torch.Tensor,
    prompt: str,
    sample_every: int = 10,
) -> float:
    """Compute mean ImageReward score for sampled frames.
    Samples every N frames to keep computation manageable."""
    model = _get_image_reward()
    if model == "unavailable":
        return 0.0
    from torchvision.transforms.functional import to_pil_image

    scores = []
    indices = list(range(0, len(frames), sample_every))
    if not indices:
        indices = [0]

    for t in indices:
        pil_img = to_pil_image(frames[t].cpu().clamp(0, 1))
        with torch.no_grad():
            score = model.score(prompt, pil_img)
        scores.append(float(score))

    return float(np.mean(scores))


def compute_hpsv2_frames(
    frames: torch.Tensor,
    prompt: str,
    sample_every: int = 10,
) -> float:
    """Compute mean HPSv2 score for sampled frames."""
    global _hpsv2_model
    try:
        import hpsv2
    except ImportError:
        print("[metrics] WARNING: hpsv2 not installed, returning 0.0")
        return 0.0

    # Initialize model once (Bug 9 fix)
    if _hpsv2_model is None:
        _hpsv2_model = hpsv2  # hpsv2 module handles internal model caching

    from torchvision.transforms.functional import to_pil_image

    scores = []
    indices = list(range(0, len(frames), sample_every))
    if not indices:
        indices = [0]

    for t in indices:
        pil_img = to_pil_image(frames[t].cpu().clamp(0, 1))
        score = _hpsv2_model.score(pil_img, prompt)
        scores.append(float(score))

    return float(np.mean(scores))


# ──────────────────────────────────────────────────────────────
# Frame extraction for FID
# ──────────────────────────────────────────────────────────────

def extract_frames_to_dir(
    video_paths: list[str],
    output_dir: str,
    sample_every: int = 4,
) -> str:
    """Extract frames from video tensors to PNG files for FID computation."""
    from torchvision.utils import save_image
    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    for vp in video_paths:
        frames = torch.load(vp, weights_only=True)  # (T, C, H, W)
        for t in range(0, len(frames), sample_every):
            save_image(
                frames[t].clamp(0, 1),
                os.path.join(output_dir, f"frame_{frame_idx:06d}.png"),
            )
            frame_idx += 1

    print(f"[metrics] Extracted {frame_idx} frames to {output_dir}")
    return output_dir


# ──────────────────────────────────────────────────────────────
# Composite score
# ──────────────────────────────────────────────────────────────

# Normalization references — calibrated from dense baseline when available.
# These are fallback ranges used only before a baseline exists.
# After baseline, compute_composite_score uses baseline-relative normalization.
FALLBACK_NORM_RANGES = {
    "fid":           (0.0, 200.0),    # lower is better, 0=perfect
    "image_reward":  (-2.0, 2.0),     # higher is better
    "hpsv2":         (0.0, 1.0),      # higher is better
    "ssim":          (0.0, 1.0),      # higher is better
    "psnr":          (10.0, 50.0),    # higher is better
    "lpips":         (0.0, 0.7),      # lower is better
}

# Weights encode research priority — agent can see but NOT change
# Bug 7 fix: these now sum to 1.0 (was 0.90 with "reserved" 0.10)
QUALITY_WEIGHTS = {
    "fid":           0.22,    # semantic fidelity
    "image_reward":  0.17,    # human preference
    "hpsv2":         0.11,    # human preference v2
    "ssim":          0.17,    # structural similarity
    "psnr":          0.11,    # signal fidelity
    "lpips":         0.22,    # perceptual distance
}


def _normalize(value: float, low: float, high: float, higher_is_better: bool) -> float:
    """Normalize a metric to [0, 1] where 1 = best."""
    clamped = max(low, min(high, value))
    normalized = (clamped - low) / (high - low + 1e-8)
    if not higher_is_better:
        normalized = 1.0 - normalized
    return normalized


def _load_baseline_metrics() -> dict | None:
    """Try to load baseline metrics for calibrated normalization."""
    baseline_path = Path(__file__).parent.parent / "results" / "baseline_quality.json"
    if baseline_path.exists():
        import json
        with open(baseline_path) as f:
            return json.load(f)
    return None


def compute_composite_score(metrics: dict) -> float:
    """Compute weighted composite quality score in [0, 1].

    If baseline metrics exist (from dense baseline run), uses baseline-relative
    normalization so scores are meaningful. Otherwise falls back to heuristic ranges.
    """
    score = 0.0
    higher_is_better = {
        "fid": False, "image_reward": True, "hpsv2": True,
        "ssim": True, "psnr": True, "lpips": False,
    }

    baseline = _load_baseline_metrics()

    # Reference-comparison metrics (FID, SSIM, PSNR, LPIPS) have trivial baseline
    # values (0.0 or 1.0) because the baseline compares against itself.
    # Baseline-relative normalization only works for standalone metrics
    # (ImageReward, HPSv2) where the baseline value is meaningful.
    STANDALONE_METRICS = {"image_reward", "hpsv2"}

    for metric_name, weight in QUALITY_WEIGHTS.items():
        if metric_name not in metrics:
            continue

        value = metrics[metric_name]

        if baseline and metric_name in baseline and metric_name in STANDALONE_METRICS:
            bval = baseline[metric_name]
            hib = higher_is_better[metric_name]

            if abs(bval) < 1e-6:
                # Baseline value is ~0, can't normalize relative to it
                lo, hi = FALLBACK_NORM_RANGES.get(metric_name, (0.0, 1.0))
                norm = _normalize(value, lo, hi, hib)
            elif hib:
                norm = min(1.0, max(0.0, 0.8 * value / (bval + 1e-8)))
            else:
                norm = min(1.0, max(0.0, 1.0 - 0.8 * value / (2 * bval + 1e-8)))
        else:
            # Reference-comparison metrics always use heuristic ranges
            lo, hi = FALLBACK_NORM_RANGES.get(metric_name, (0.0, 1.0))
            norm = _normalize(value, lo, hi, higher_is_better[metric_name])

        score += weight * norm

    return score


# ──────────────────────────────────────────────────────────────
# Main evaluation function
# ──────────────────────────────────────────────────────────────

def evaluate_experiment(
    video_paths: list[str],
    reference_paths: list[str],
    prompts: list[str],
    work_dir: str = "/tmp/autobench_eval",
    mode: str = "full",           # "fast" = FID + LPIPS only, "full" = all metrics
) -> AggregateMetrics:
    """Compute all metrics for an experiment.

    Args:
        video_paths: Paths to generated video tensors (.pt files)
        reference_paths: Paths to dense-baseline reference video tensors
        prompts: Text prompts used for generation
        work_dir: Temp directory for frame extraction
        mode: "fast" for screening, "full" for complete evaluation
    """
    print(f"[metrics] Evaluating {len(video_paths)} videos (mode={mode})")
    t0 = time.time()

    per_video = []
    all_ssim, all_psnr, all_lpips = [], [], []
    all_ir, all_hpsv2 = [], []

    for i, (gen_path, ref_path) in enumerate(zip(video_paths, reference_paths)):
        gen_frames = torch.load(gen_path, weights_only=True)   # (T, C, H, W)
        ref_frames = torch.load(ref_path, weights_only=True)

        # Ensure same length
        T = min(len(gen_frames), len(ref_frames))
        gen_frames = gen_frames[:T].float()
        ref_frames = ref_frames[:T].float()

        vm = VideoMetrics(prompt_idx=i, prompt=prompts[i] if i < len(prompts) else "")

        # Always compute LPIPS (fast, informative)
        vm.lpips = compute_lpips_frames(gen_frames, ref_frames)
        all_lpips.append(vm.lpips)

        if mode == "full":
            # Fidelity metrics
            vm.ssim = compute_ssim_frames(gen_frames.cuda(), ref_frames.cuda())
            vm.psnr = compute_psnr_frames(gen_frames.cuda(), ref_frames.cuda())
            all_ssim.append(vm.ssim)
            all_psnr.append(vm.psnr)

            # Preference metrics (expensive — sample frames)
            if i < len(prompts):
                vm.image_reward = compute_image_reward_frames(gen_frames, prompts[i])
                vm.hpsv2 = compute_hpsv2_frames(gen_frames, prompts[i])
                all_ir.append(vm.image_reward)
                all_hpsv2.append(vm.hpsv2)

        per_video.append(vm)

    # FID: computed across all frames collectively
    gen_frames_dir = os.path.join(work_dir, "gen_frames")
    ref_frames_dir = os.path.join(work_dir, "ref_frames")
    extract_frames_to_dir(video_paths, gen_frames_dir)
    extract_frames_to_dir(reference_paths, ref_frames_dir)
    fid_score = compute_fid(gen_frames_dir, ref_frames_dir)

    # Aggregate
    agg = AggregateMetrics(
        fid=fid_score,
        lpips_mean=float(np.mean(all_lpips)) if all_lpips else 0.0,
        lpips_std=float(np.std(all_lpips)) if all_lpips else 0.0,
    )

    if mode == "full":
        agg.ssim_mean = float(np.mean(all_ssim)) if all_ssim else 0.0
        agg.ssim_std = float(np.std(all_ssim)) if all_ssim else 0.0
        agg.psnr_mean = float(np.mean(all_psnr)) if all_psnr else 0.0
        agg.psnr_std = float(np.std(all_psnr)) if all_psnr else 0.0
        agg.image_reward_mean = float(np.mean(all_ir)) if all_ir else 0.0
        agg.hpsv2_mean = float(np.mean(all_hpsv2)) if all_hpsv2 else 0.0

    # Composite score
    metrics_dict = {
        "fid": agg.fid,
        "ssim": agg.ssim_mean,
        "psnr": agg.psnr_mean,
        "lpips": agg.lpips_mean,
        "image_reward": agg.image_reward_mean,
        "hpsv2": agg.hpsv2_mean,
    }
    agg.composite_score = compute_composite_score(metrics_dict)
    agg.per_video = [asdict(v) for v in per_video]

    elapsed = time.time() - t0
    print(f"[metrics] Evaluation complete in {elapsed:.1f}s")
    print(f"[metrics] FID={agg.fid:.2f} SSIM={agg.ssim_mean:.4f} "
          f"LPIPS={agg.lpips_mean:.4f} IR={agg.image_reward_mean:.4f} "
          f"Composite={agg.composite_score:.4f}")

    return agg
