"""
PISA (Piecewise Sparse Attention) for Wan2.1-1.3B.
Training-free sparse attention using Triton kernels.

API: piecewise_sparse_attention(q, k, v, density=0.15, block_size=64)
Input: (B, H, T, D) — same as SDPA format
Output: (B, H, T, D)

PISA works by:
1. Reducing Q/K/V to block-level summaries (chunk_reduce)
2. Computing block-level attention scores
3. Selecting top-k blocks per query block (density controls k)
4. Computing exact attention only for selected blocks
"""

import sys
import torch

# Fix Triton 3.4.0 allocator requirement before importing PISA
import triton
from triton.runtime._allocation import Allocator

class TorchCUDAAllocator(Allocator):
    def __call__(self, size: int, alignment: int, stream: int):
        return torch.cuda.caching_allocator_alloc(size, stream)

triton.set_allocator(TorchCUDAAllocator())

# Add PISA to path
sys.path.insert(0, "/home/researcher/autobench_work/methods/pisa")


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


class PISAProcessor:
    """WanAttnProcessor replacement using PISA sparse attention."""

    def __init__(self, density=0.15, block_size=64,
                 video_length=81, resolution=(480, 832)):
        self.density = density
        self.block_size = block_size
        self.video_length = video_length
        self.resolution = resolution
        self._pisa_fn = None

    def _get_pisa(self):
        if self._pisa_fn is None:
            from piecewise_attn import piecewise_sparse_attention
            self._pisa_fn = piecewise_sparse_attention
        return self._pisa_fn

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        batch_size, seq_len, _ = hidden_states.shape

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

        # Transpose: (B, seq, heads, hd) → (B, heads, seq, hd)
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        # Step 5: PISA sparse attention
        try:
            pisa_fn = self._get_pisa()
            out = pisa_fn(
                query, key, value,
                density=self.density,
                block_size=self.block_size,
            )
        except Exception as e:
            # Log the error so we know PISA failed (not silent!)
            import traceback
            print(f"[PISA] ERROR in sparse attention: {e}")
            traceback.print_exc()
            # Fallback to dense
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask
            )

        # Reshape: (B, heads, seq, hd) → (B, seq, dim)
        out = out.transpose(1, 2).flatten(2)
        out = out.type_as(hidden_states)

        # Step 6: Output projection
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out


def create_patch(config):
    """PISA piecewise sparse attention patch."""
    params = config.params

    density = params.sparsity_ratio if params.sparsity_ratio is not None else 0.15
    block_size = params.block_size if params.block_size is not None else 64

    video_length = config.inference.video_length
    resolution = tuple(config.inference.resolution)

    proc = PISAProcessor(
        density=density, block_size=block_size,
        video_length=video_length, resolution=resolution,
    )

    def patch_fn(original_forward, layer_name, module):
        module.set_processor(proc)
        return original_forward

    return patch_fn
