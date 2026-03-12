"""
VMoBA (Mixture-of-Block Attention) for Wan2.1-1.3B.
Uses flash_attn varlen with block-level top-k selection.

VMoBA splits KV into chunks, selects top-k relevant chunks per query,
and computes exact attention only on selected chunks.

Input to moba_attn_varlen: (S, H, D) with cu_seqlens
Wan2.1: T=21, H=30, W=52, seq_len=32760, 12 heads, 128 head_dim
"""

import sys
import torch
from einops import rearrange

# Fix flash-attn 2.8.3 API change: returns 4 values instead of 8.
# VMoBA expects (q, k, v, out, out_padded, softmax_lse, S_dmask, rng_state)
# but 2.8.3 returns (out, softmax_lse, rng_state, S_dmask).
# Monkey-patch before importing vmoba.
import flash_attn.flash_attn_interface as _fai
_orig_varlen_fwd = _fai._flash_attn_varlen_forward
def _compat_varlen_fwd(*args, **kwargs):
    result = _orig_varlen_fwd(*args, **kwargs)
    if len(result) == 4:
        out, softmax_lse, rng_state, S_dmask = result
        q = kwargs.get('q', args[0] if len(args) > 0 else torch.empty(0))
        k = kwargs.get('k', args[1] if len(args) > 1 else torch.empty(0))
        v = kwargs.get('v', args[2] if len(args) > 2 else torch.empty(0))
        return (q, k, v, out, out, softmax_lse, S_dmask, rng_state)
    return result
_fai._flash_attn_varlen_forward = _compat_varlen_fwd

sys.path.insert(0, "/home/researcher/autobench_work/methods/vmoba/src")


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


class VMoBAProcessor:
    """WanAttnProcessor replacement using VMoBA sparse attention."""

    def __init__(self, chunk_size=3, moba_topk=3, simsum_threshold=0.25,
                 video_length=81, resolution=(480, 832)):
        self.chunk_size = chunk_size  # temporal chunk size (1D: int = t frames * H * W)
        self.moba_topk = moba_topk
        self.simsum_threshold = simsum_threshold
        self.video_length = video_length
        self.resolution = resolution
        self._moba_fn = None

    def _get_moba(self):
        if self._moba_fn is None:
            from vmoba import moba_attn_varlen
            self._moba_fn = moba_attn_varlen
        return self._moba_fn

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

        # Step 2: Head reshape: (B, seq, dim) -> (B, seq, heads, head_dim)
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

        # Step 5: VMoBA sparse attention
        # moba_attn_varlen expects (total_seq, heads, head_dim) with cu_seqlens
        T, H, W = get_video_dims(seq_len, self.video_length, self.resolution)

        if T is not None:
            try:
                moba_fn = self._get_moba()

                # Compute chunk size in tokens
                # chunk_size as int = number of temporal frames per chunk
                # Each chunk = chunk_size * H * W tokens
                moba_chunk_size = self.chunk_size * H * W

                # Flatten batch into varlen format: (B, S, H, D) -> (B*S, H, D)
                q_flat = rearrange(query, "b s h d -> (b s) h d")
                k_flat = rearrange(key, "b s h d -> (b s) h d")
                v_flat = rearrange(value, "b s h d -> (b s) h d")

                # Create cu_seqlens for batch boundaries
                cu_seqlens = torch.arange(
                    0, batch_size * seq_len + 1, seq_len,
                    dtype=torch.int32, device=query.device
                )

                out_flat = moba_fn(
                    q_flat, k_flat, v_flat,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=seq_len,
                    moba_chunk_size=moba_chunk_size,
                    moba_topk=self.moba_topk,
                    select_mode='threshold',
                    simsum_threshold=self.simsum_threshold,
                    threshold_type='query_head',
                )

                # Reshape back: (B*S, H, D) -> (B, S, dim)
                out = rearrange(out_flat, "(b s) h d -> b s (h d)", b=batch_size)

            except Exception as e:
                # Fallback to dense SDPA
                query_t = query.transpose(1, 2).contiguous()
                key_t = key.transpose(1, 2).contiguous()
                value_t = value.transpose(1, 2).contiguous()
                out = torch.nn.functional.scaled_dot_product_attention(
                    query_t, key_t, value_t, attn_mask=attention_mask
                )
                out = out.transpose(1, 2).flatten(2)
        else:
            # Unknown layout, fallback to dense
            query_t = query.transpose(1, 2).contiguous()
            key_t = key.transpose(1, 2).contiguous()
            value_t = value.transpose(1, 2).contiguous()
            out = torch.nn.functional.scaled_dot_product_attention(
                query_t, key_t, value_t, attn_mask=attention_mask
            )
            out = out.transpose(1, 2).flatten(2)

        out = out.type_as(hidden_states)

        # Step 6: Output projection
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out


def create_patch(config):
    """VMoBA mixture-of-block sparse attention patch."""
    params = config.params

    # chunk_size: number of temporal frames per chunk
    # With T=21, chunk_size=3 gives 7 chunks of 3*30*52=4680 tokens each
    chunk_size = params.tile_size_t if params.tile_size_t is not None else 3
    moba_topk = params.block_size if params.block_size is not None else 3
    simsum_threshold = params.sparsity_ratio if params.sparsity_ratio is not None else 0.25

    video_length = config.inference.video_length
    resolution = tuple(config.inference.resolution)

    proc = VMoBAProcessor(
        chunk_size=chunk_size, moba_topk=moba_topk,
        simsum_threshold=simsum_threshold,
        video_length=video_length, resolution=resolution,
    )

    def patch_fn(original_forward, layer_name, module):
        module.set_processor(proc)
        return original_forward

    return patch_fn
