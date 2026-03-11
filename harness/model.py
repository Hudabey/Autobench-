"""
Model — Wan2.1-1.3B loading and attention patching infrastructure.

This module handles:
1. Loading the base model (once, cached)
2. Providing hook points for attention replacement
3. Restoring to dense attention between experiments
4. Timing instrumentation

🔒 FIXED — agent cannot modify this file.
"""

import time
import torch
import gc
from contextlib import contextmanager
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class AttentionTiming:
    """Accumulated attention timing across all denoising steps."""
    total_attention_ms: float = 0.0
    num_calls: int = 0

    @property
    def avg_ms_per_call(self) -> float:
        return self.total_attention_ms / max(self.num_calls, 1)

    def reset(self):
        self.total_attention_ms = 0.0
        self.num_calls = 0


# Global timing state (reset per experiment)
_attention_timing = AttentionTiming()


def get_attention_timing() -> AttentionTiming:
    return _attention_timing


def reset_attention_timing():
    _attention_timing.reset()


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────

_cached_pipe = None
_cached_model_id = None


def load_pipeline(
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Load Wan2.1-1.3B pipeline. Cached across experiments.

    Uses diffusers WanPipeline. If your Wan2.1 setup uses a different
    loading path, modify THIS file (not the agent's code).
    """
    global _cached_pipe, _cached_model_id

    if _cached_pipe is not None and _cached_model_id == model_id:
        print(f"[harness] Using cached pipeline: {model_id}")
        return _cached_pipe

    print(f"[harness] Loading pipeline: {model_id} ({dtype})")
    t0 = time.time()

    try:
        # Primary path: diffusers
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
    except ImportError:
        # Fallback: try wan2.1 native loading
        # Adjust this block to match your local Wan2.1 setup
        raise ImportError(
            "Could not import WanPipeline from diffusers. "
            "Install diffusers >= 0.33.0 or adjust model.py for your setup.\n"
            "  pip install diffusers[torch] --upgrade"
        )

    elapsed = time.time() - t0
    print(f"[harness] Pipeline loaded in {elapsed:.1f}s")

    _cached_pipe = pipe
    _cached_model_id = model_id
    return pipe


def get_transformer(pipe):
    """Extract the DiT transformer from the pipeline.
    This is what contains the attention layers the agent patches."""
    # diffusers stores it as pipe.transformer for Wan2.1
    if hasattr(pipe, "transformer"):
        return pipe.transformer
    elif hasattr(pipe, "unet"):
        return pipe.unet
    else:
        raise AttributeError(
            "Cannot find transformer/unet on pipeline. "
            "Check your diffusers version or adjust model.py."
        )


# ──────────────────────────────────────────────────────────────
# Attention layer discovery
# ──────────────────────────────────────────────────────────────

def find_attention_layers(transformer) -> list[tuple[str, torch.nn.Module]]:
    """Find all SELF-attention layers in the DiT transformer.

    Returns list of (name, module) tuples for every self-attention layer.

    Wan2.1 structure (30 blocks):
      blocks.{i}.attn1 — self-attention  (is_cross_attention=False)
      blocks.{i}.attn2 — cross-attention (is_cross_attention=True)

    We ONLY return attn1 layers. Patching attn2 would corrupt text conditioning.
    """
    attention_layers = []
    seen_paths = set()

    for name, module in transformer.named_modules():
        module_type = type(module).__name__.lower()

        # Must be an attention-like class
        is_attention = ("attention" in module_type or "attn" in module_type)
        if not is_attention:
            continue

        # Skip processors/sub-components
        if "processor" in module_type:
            continue

        # Skip cross-attention layers — check the attribute, NOT the name.
        # Wan2.1 names are blocks.0.attn1 (self) and blocks.0.attn2 (cross)
        # — neither contains "cross" in the string, so name-based filtering fails.
        if getattr(module, "is_cross_attention", False):
            continue

        # Fallback: if the attribute doesn't exist, use naming heuristics
        # "attn2" is cross-attention in Wan2.1 / diffusers convention
        if not hasattr(module, "is_cross_attention"):
            if "attn2" in name or "cross" in name.lower():
                continue

        # Skip children of already-found layers
        is_child = any(name.startswith(p + ".") for p in seen_paths)
        if is_child:
            continue

        attention_layers.append((name, module))
        seen_paths.add(name)

    print(f"[harness] Found {len(attention_layers)} self-attention layers")
    if attention_layers:
        print(f"[harness] First: {attention_layers[0][0]} ({type(attention_layers[0][1]).__name__})")
        # Sanity check: Wan2.1-1.3B should have 30 self-attention layers
        if len(attention_layers) > 40:
            print(f"[harness] WARNING: Found {len(attention_layers)} layers — "
                  f"expected ~30 for Wan2.1-1.3B. Cross-attention may be leaking through.")
    return attention_layers


def get_attention_info(pipe) -> dict:
    """Get metadata about the model's attention configuration.
    Useful for the agent to understand what it's working with."""
    transformer = get_transformer(pipe)
    layers = find_attention_layers(transformer)

    info = {
        "num_attention_layers": len(layers),
        "layer_names": [name for name, _ in layers],
        "model_dtype": str(next(transformer.parameters()).dtype),
        "model_device": str(next(transformer.parameters()).device),
    }

    # Try to extract head/dim info from first attention layer
    if layers:
        _, first_attn = layers[0]
        for attr in ["num_heads", "n_heads", "num_attention_heads"]:
            if hasattr(first_attn, attr):
                info["num_heads"] = getattr(first_attn, attr)
                break
        for attr in ["head_dim", "d_head"]:
            if hasattr(first_attn, attr):
                info["head_dim"] = getattr(first_attn, attr)
                break

    return info


# ──────────────────────────────────────────────────────────────
# Attention patching (the mechanism the agent uses)
# ──────────────────────────────────────────────────────────────

_original_forwards = {}  # stores original forward methods for restoration


def patch_attention(
    pipe,
    patch_fn: Callable,
    layer_selector: str = "all",
):
    """Replace attention forward methods with the agent's patched version.

    Args:
        pipe: The Wan2.1 pipeline
        patch_fn: Function that takes (original_forward, layer_name, module)
                  and returns a new forward method
        layer_selector: "all", "even", "odd", or list of layer indices
    """
    global _original_forwards
    transformer = get_transformer(pipe)
    layers = find_attention_layers(transformer)

    # Determine which layers to patch
    if layer_selector == "all":
        indices = list(range(len(layers)))
    elif layer_selector == "even":
        indices = list(range(0, len(layers), 2))
    elif layer_selector == "odd":
        indices = list(range(1, len(layers), 2))
    elif isinstance(layer_selector, list):
        indices = layer_selector
    else:
        indices = list(range(len(layers)))

    patched_count = 0
    for idx in indices:
        if idx >= len(layers):
            continue
        name, module = layers[idx]

        # Save original forward for restoration
        if name not in _original_forwards:
            _original_forwards[name] = module.forward

        # Create timed wrapper around the agent's patch
        original_fwd = _original_forwards[name]
        new_fwd = patch_fn(original_fwd, name, module)

        # Wrap with timing instrumentation
        def make_timed(fn):
            def timed_forward(*args, **kwargs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = fn(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                _attention_timing.total_attention_ms += elapsed_ms
                _attention_timing.num_calls += 1
                return result
            return timed_forward

        module.forward = make_timed(new_fwd)
        patched_count += 1

    print(f"[harness] Patched {patched_count}/{len(layers)} attention layers "
          f"(selector='{layer_selector}')")


def restore_attention(pipe):
    """Restore all attention layers to their original dense implementations."""
    global _original_forwards
    transformer = get_transformer(pipe)

    restored = 0
    for name, module in transformer.named_modules():
        if name in _original_forwards:
            module.forward = _original_forwards[name]
            restored += 1

    _original_forwards.clear()
    reset_attention_timing()
    print(f"[harness] Restored {restored} attention layers to dense")


@contextmanager
def attention_experiment(pipe, patch_fn=None, layer_selector="all"):
    """Context manager for clean attention experiments.

    Usage:
        with attention_experiment(pipe, my_patch_fn):
            videos = generate(pipe, prompts)
    # attention automatically restored after block
    """
    reset_attention_timing()

    if patch_fn is not None:
        patch_attention(pipe, patch_fn, layer_selector)

    try:
        yield pipe
    finally:
        restore_attention(pipe)


# ──────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────

def generate_videos(
    pipe,
    prompts: list[dict],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    video_length: int = 81,
    resolution: tuple = (480, 832),
    seed: int = 42,
    output_dir: str = ".",
    timeout_s: float = 600.0,
) -> tuple[list[str], dict]:
    """Generate videos from prompts and return paths + timing info.

    Args:
        prompts: list of {"id": int, "prompt": str, ...} dicts.
                 The id is used in filenames for correct reference matching.
        timeout_s: per-video timeout in seconds (default 10 min).

    Returns:
        (video_paths, timing_dict)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    video_paths = []
    prompt_ids = []
    total_gen_time = 0.0
    height, width = resolution

    for i, prompt_data in enumerate(prompts):
        prompt_id = prompt_data["id"]
        prompt_text = prompt_data["prompt"]

        # Seed is deterministic per prompt ID (not per position)
        generator = torch.Generator(device="cuda").manual_seed(seed + prompt_id)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Generate video
        output = pipe(
            prompt=prompt_text,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_frames=video_length,
            height=height,
            width=width,
            generator=generator,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - t0
        total_gen_time += gen_time

        # Timeout check
        if gen_time > timeout_s:
            print(f"[harness] WARNING: video {prompt_id} took {gen_time:.0f}s "
                  f"(timeout={timeout_s:.0f}s)")

        # Save with prompt ID in filename (not sequential index)
        # This ensures fast-mode videos match the correct references
        video_path = os.path.join(output_dir, f"video_p{prompt_id:03d}.pt")

        if hasattr(output, "frames"):
            frames = output.frames
            if isinstance(frames, list) and len(frames) > 0:
                if isinstance(frames[0], list):
                    # frames is [[PIL, PIL, ...]] — batch dim
                    frames = frames[0]
                # Convert PIL to tensor if needed
                if hasattr(frames[0], "convert"):
                    import torchvision.transforms as T
                    to_tensor = T.ToTensor()
                    frames_tensor = torch.stack([to_tensor(f) for f in frames])
                else:
                    frames_tensor = torch.stack(frames) if isinstance(frames, list) else frames
            else:
                frames_tensor = frames
        else:
            frames_tensor = output

        torch.save(frames_tensor.cpu(), video_path)
        video_paths.append(video_path)
        prompt_ids.append(prompt_id)

        print(f"[harness] Generated video {i+1}/{len(prompts)} (p{prompt_id:03d}): "
              f"{gen_time:.1f}s — {prompt_text[:60]}...")

    timing = {
        "total_generation_s": total_gen_time,
        "avg_per_video_s": total_gen_time / max(len(prompts), 1),
        "ms_per_frame": (total_gen_time * 1000) / max(len(prompts) * video_length, 1),
    }

    return video_paths, timing


def cleanup_gpu():
    """Free GPU memory between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
