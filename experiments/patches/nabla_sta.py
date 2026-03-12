"""
NABLA/STA 3D Sliding Window Sparse Attention for Wan2.1-1.3B.
Uses flex_attention with a 3D sliding window BlockMask.

Tokens at (t,h,w) attend to neighbors within ±wT frames, ±wH rows, ±wW cols.
This exploits temporal + spatial locality in video diffusion.

Wan2.1-1.3B: 12 heads, 128 head_dim, patch_size=(1,2,2)
At 81 frames, 480x832: T=21, H=30, W=52, seq_len=32760
"""

import torch
import math

# ── Lazy imports and caches ──
_mask_cache = {}
_flex_compiled = None


def _get_compiled_flex():
    global _flex_compiled
    if _flex_compiled is None:
        from torch.nn.attention.flex_attention import flex_attention
        _flex_compiled = torch.compile(
            flex_attention, mode="max-autotune-no-cudagraphs", dynamic=False
        )
    return _flex_compiled


def get_video_dims(seq_len, video_length=81, resolution=(480, 832)):
    """Infer T, H, W from sequence length and video config.
    Wan2.1 VAE: temporal ~4x, spatial 8x. Patch embedding: (1, 2, 2)."""
    h_lat = resolution[0] // 8
    w_lat = resolution[1] // 8
    h_post = h_lat // 2
    w_post = w_lat // 2
    # Try common temporal latent sizes
    for t_latent in [21, 17, 13, 9, 5]:
        t_post = t_latent  # patch_t = 1
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


class SparseSTAProcessor:
    """Drop-in WanAttnProcessor replacement with 3D sliding window attention.

    Steps 1-4 and 6 are identical to WanAttnProcessor.
    Step 5 replaces dense SDPA with flex_attention + BlockMask.
    """

    def __init__(self, wT=5, wH=4, wW=4, video_length=81, resolution=(480, 832)):
        self.wT = wT
        self.wH = wH
        self.wW = wW
        self.video_length = video_length
        self.resolution = resolution

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        # Self-attention only (harness never patches cross-attention)
        batch_size, seq_len, _ = hidden_states.shape

        # Step 1: Q/K/V projection
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Step 3 (before reshape): QK normalization (RMSNorm)
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Step 2: Head reshape: (B, seq, dim) → (B, seq, heads, head_dim)
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

        # ═══════════════════════════════════════════════
        # Step 5: Sparse attention via flex_attention
        # ═══════════════════════════════════════════════
        T, H, W = get_video_dims(seq_len, self.video_length, self.resolution)

        if T is not None:
            cache_key = (T, H, W, self.wT, self.wH, self.wW, str(query.device))
            if cache_key not in _mask_cache:
                _mask_cache[cache_key] = create_3d_window_mask(
                    T, H, W, self.wT, self.wH, self.wW, device=query.device
                )
            block_mask = _mask_cache[cache_key]
            compiled_flex = _get_compiled_flex()
            out = compiled_flex(query, key, value, block_mask=block_mask)
        else:
            # Fallback: dense SDPA (unknown seq_len layout)
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
    """NABLA/STA sparse attention — 3D sliding window via flex_attention."""
    params = config.params

    # Window half-sizes (tokens attend ±wT frames, ±wH rows, ±wW cols)
    wT = params.tile_size_t if params.tile_size_t is not None else 5
    wH = params.tile_size_h if params.tile_size_h is not None else 4
    wW = params.tile_size_w if params.tile_size_w is not None else 4

    video_length = config.inference.video_length
    resolution = tuple(config.inference.resolution)

    proc = SparseSTAProcessor(
        wT=wT, wH=wH, wW=wW,
        video_length=video_length, resolution=resolution,
    )

    def patch_fn(original_forward, layer_name, module):
        module.set_processor(proc)
        return original_forward

    return patch_fn
