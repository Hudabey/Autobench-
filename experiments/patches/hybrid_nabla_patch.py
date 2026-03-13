"""
Hybrid Dense→NABLA attention for Wan2.1-1.3B.
Uses dense attention for the first N denoising steps (global context)
and NABLA/STA 3D sliding window for the remaining steps (fast local attention).

With 30 attn layers, each denoising step = 30 forward calls.
50 steps × 30 layers = 1500 calls per video. Uses modulo for multi-video.
"""

import torch
import math

# ── Lazy imports and caches ──
_mask_cache = {}
_flex_compiled = None

# Global step counter shared across all layers
_global_call_count = [0]
_num_layers = [30]
_num_steps = [50]


def _get_compiled_flex():
    global _flex_compiled
    if _flex_compiled is None:
        from torch.nn.attention.flex_attention import flex_attention
        _flex_compiled = torch.compile(
            flex_attention, mode="max-autotune-no-cudagraphs", dynamic=False
        )
    return _flex_compiled


def get_video_dims(seq_len, video_length=81, resolution=(480, 832)):
    h_lat = resolution[0] // 8
    w_lat = resolution[1] // 8
    h_post = h_lat // 2
    w_post = w_lat // 2
    for t_latent in [21, 17, 13, 9, 5]:
        t_post = t_latent
        if t_post * h_post * w_post == seq_len:
            return t_post, h_post, w_post
    return None, None, None


def create_3d_window_mask(T, H, W, wT, wH, wW, device="cuda"):
    """Create a 3D sliding window BlockMask for flex_attention."""
    from torch.nn.attention.flex_attention import create_block_mask
    HW = H * W

    def mask_mod(b, h, q_idx, kv_idx):
        q_t = q_idx // HW
        q_r = q_idx % HW
        q_h = q_r // W
        q_w = q_r % W
        kv_t = kv_idx // HW
        kv_r = kv_idx % HW
        kv_h = kv_r // W
        kv_w = kv_r % W
        return (
            ((q_t - kv_t).abs() <= wT)
            & ((q_h - kv_h).abs() <= wH)
            & ((q_w - kv_w).abs() <= wW)
        )

    seq_len = T * H * W
    block_mask = create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
        device=device, _compile=True,
    )
    return block_mask


class HybridNABLAProcessor:
    """Dense for early steps, NABLA/STA 3D sliding window for later steps."""

    def __init__(self, dense_steps=10, wT=5, wH=4, wW=4,
                 video_length=81, resolution=(480, 832)):
        self.dense_steps = dense_steps
        self.wT = wT
        self.wH = wH
        self.wW = wW
        self.video_length = video_length
        self.resolution = resolution

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        batch_size, seq_len, _ = hidden_states.shape

        # Determine current denoising step via modulo (wraps per video)
        calls_per_video = _num_steps[0] * _num_layers[0]
        current_step = (_global_call_count[0] % calls_per_video) // _num_layers[0]
        _global_call_count[0] += 1
        use_sparse = current_step >= self.dense_steps

        # Step 1: Q/K/V projection
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Step 3: QK normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Step 2: Head reshape
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Step 4: RoPE
        if rotary_emb is not None:
            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(x)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(x)
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # Transpose for attention: (B, seq, heads, hd) → (B, heads, seq, hd)
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        # Step 5: Attention (dense or NABLA sparse)
        T, H, W = get_video_dims(seq_len, self.video_length, self.resolution)

        if use_sparse and T is not None:
            cache_key = (T, H, W, self.wT, self.wH, self.wW, str(query.device))
            if cache_key not in _mask_cache:
                _mask_cache[cache_key] = create_3d_window_mask(
                    T, H, W, self.wT, self.wH, self.wW, device=query.device
                )
            block_mask = _mask_cache[cache_key]
            compiled_flex = _get_compiled_flex()
            out = compiled_flex(query, key, value, block_mask=block_mask)
        else:
            # Dense SDPA for early steps or unknown layout
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask
            )

        # Reshape back: (B, heads, seq, hd) → (B, seq, dim)
        out = out.transpose(1, 2).flatten(2)
        out = out.type_as(hidden_states)

        # Step 6: Output projection
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out


def create_patch(config):
    """Hybrid dense→NABLA: dense for early steps, 3D sliding window for later."""
    params = config.params

    wT = params.tile_size_t if params.tile_size_t is not None else 5
    wH = params.tile_size_h if params.tile_size_h is not None else 4
    wW = params.tile_size_w if params.tile_size_w is not None else 4
    # dense_steps stored in block_size param (repurposed)
    dense_steps = params.block_size if hasattr(params, 'block_size') and params.block_size is not None else 10

    video_length = config.inference.video_length
    resolution = tuple(config.inference.resolution)

    # Set num_steps for modulo counter
    _num_steps[0] = config.inference.num_inference_steps
    # Reset counter at patch time
    _global_call_count[0] = 0

    proc = HybridNABLAProcessor(
        dense_steps=dense_steps,
        wT=wT, wH=wH, wW=wW,
        video_length=video_length, resolution=resolution,
    )

    def patch_fn(original_forward, layer_name, module):
        module.set_processor(proc)
        return original_forward

    return patch_fn
