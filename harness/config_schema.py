"""
Config Schema — defines the contract between agent and harness.

The agent produces a config.yaml. This module validates it.
If validation fails, the experiment is skipped and the failure is logged.

🔒 FIXED — agent cannot modify this file.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml


# ──────────────────────────────────────────────────────────────
# Hard bounds — agent configs are clamped/rejected outside these
# ──────────────────────────────────────────────────────────────

VALID_METHODS = [
    "dense",                  # baseline (no sparse attention)
    "sla1", "sla2",           # Sparse-Linear Attention family
    "vorta",                  # Dynamic routing sparse kernels
    "nabla",                  # Neighborhood Adaptive Block-Level
    "vsa",                    # Two-stage (FastVideo)
    "vmoba",                  # Recurrent 1D-2D-3D partitioning
    "sparse_vdit",            # Pattern-optimized kernels
    "sliding_tile",           # Sliding Tile Attention (FastVideo)
    "deepseek_sparse",        # DeepSeek sparse attention
    "native_sparse",          # DeepSeek FlashMLA
    "video_sna",              # Video Neighborhood Sparse
    "pisa",                   # Piecewise Sparse Attention
    "salad",                  # Gated parallel linear + sparse
    "monarch_rt",             # MonarchRT (research target)
    "custom",                 # agent-defined hybrid / composition
]

# Generation parameter bounds — agent cannot go outside these
BOUNDS = {
    "num_inference_steps": (10, 50),
    "guidance_scale":      (1.0, 15.0),
    "video_length":        (17, 81),        # frames, Wan2.1 supports 1-81
    "resolution":          [(480, 832)],     # fixed for comparability
}


@dataclass
class MethodParams:
    """Method-specific parameters. Loosely validated — each method
    defines its own expected params. The harness checks types only."""
    window_size: Optional[int] = None
    top_k: Optional[int] = None
    block_size: Optional[int] = None
    sparsity_ratio: Optional[float] = None
    apply_to_layers: Optional[str] = "all"
    temporal_mode: Optional[str] = None
    spatial_mode: Optional[str] = None
    num_routes: Optional[int] = None
    kernel_type: Optional[str] = None
    pattern_type: Optional[str] = None
    stripe_width: Optional[int] = None
    tile_size_t: Optional[int] = None
    tile_size_h: Optional[int] = None
    tile_size_w: Optional[int] = None
    compression_ratio: Optional[float] = None
    num_pieces: Optional[int] = None
    gate_threshold: Optional[float] = None
    linear_dim: Optional[int] = None
    extra: dict = field(default_factory=dict)  # catch-all for unknowns


@dataclass
class InferenceConfig:
    """Generation settings — bounded to prevent agent from gaming metrics."""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    video_length: int = 81               # frames
    resolution: tuple = (480, 832)
    seed: int = 42                        # fixed for reproducibility


@dataclass
class ExperimentConfig:
    """Full experiment configuration produced by the agent."""
    experiment_name: str = ""
    method: str = "dense"
    method_repo: Optional[str] = None
    params: MethodParams = field(default_factory=MethodParams)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    hypothesis: str = ""
    phase: int = 1
    hybrid: Optional[dict] = None  # Phase 4: per-step method assignment


def validate_config(cfg: ExperimentConfig) -> tuple[bool, list[str]]:
    """Validate experiment config against hard bounds.
    Returns (is_valid, list_of_errors)."""
    errors = []

    if cfg.method not in VALID_METHODS:
        errors.append(f"Unknown method '{cfg.method}'. Valid: {VALID_METHODS}")

    lo, hi = BOUNDS["num_inference_steps"]
    if not (lo <= cfg.inference.num_inference_steps <= hi):
        errors.append(f"num_inference_steps={cfg.inference.num_inference_steps} outside [{lo}, {hi}]")

    lo, hi = BOUNDS["guidance_scale"]
    if not (lo <= cfg.inference.guidance_scale <= hi):
        errors.append(f"guidance_scale={cfg.inference.guidance_scale} outside [{lo}, {hi}]")

    lo, hi = BOUNDS["video_length"]
    if not (lo <= cfg.inference.video_length <= hi):
        errors.append(f"video_length={cfg.inference.video_length} outside [{lo}, {hi}]")

    if cfg.inference.resolution not in BOUNDS["resolution"]:
        errors.append(f"resolution={cfg.inference.resolution} not in {BOUNDS['resolution']}")

    if not cfg.experiment_name.strip():
        errors.append("experiment_name is empty")

    if not cfg.hypothesis.strip():
        errors.append("hypothesis is empty — agent must document what it's testing")

    if cfg.params.sparsity_ratio is not None:
        if not (0.0 < cfg.params.sparsity_ratio < 1.0):
            errors.append(f"sparsity_ratio={cfg.params.sparsity_ratio} must be in (0, 1)")

    return (len(errors) == 0, errors)


def load_config(path: str) -> ExperimentConfig:
    """Load and validate a YAML config file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    params_raw = raw.get("params", {})
    known_fields = {f.name for f in MethodParams.__dataclass_fields__.values() if f.name != "extra"}
    known_params = {k: v for k, v in params_raw.items() if k in known_fields}
    extra_params = {k: v for k, v in params_raw.items() if k not in known_fields}
    params = MethodParams(**known_params, extra=extra_params)

    inference_raw = raw.get("inference", {})
    if "resolution" in inference_raw and isinstance(inference_raw["resolution"], list):
        inference_raw["resolution"] = tuple(inference_raw["resolution"])
    inference = InferenceConfig(**inference_raw)

    cfg = ExperimentConfig(
        experiment_name=raw.get("experiment_name", ""),
        method=raw.get("method", "dense"),
        method_repo=raw.get("method_repo"),
        params=params,
        inference=inference,
        hypothesis=raw.get("hypothesis", ""),
        phase=raw.get("phase", 1),
        hybrid=raw.get("hybrid"),
    )

    is_valid, errors = validate_config(cfg)
    if not is_valid:
        raise ValueError(f"Invalid config:\n" + "\n".join(f"  - {e}" for e in errors))

    return cfg
