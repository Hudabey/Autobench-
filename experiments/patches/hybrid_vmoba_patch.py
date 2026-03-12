"""
Hybrid Dense→VMoBA attention for Wan2.1-1.3B.
Uses dense attention for the first N denoising steps (where global context
matters most) and VMoBA sparse attention for the remaining steps.

With 30 attn layers, each denoising step = 30 forward calls.
50 steps × 30 layers = 1500 calls per video. Uses modulo for multi-video.
"""

import sys
import torch
from einops import rearrange

# Fix flash-attn 2.8.3 API change: returns 4 values instead of old 8.
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


# Global step counter shared across all layers
_global_call_count = [0]
_num_layers = [30]
_num_steps = [50]


class HybridVMoBAProcessor:
    """Dense for early steps, VMoBA sparse for later steps.

    Uses modulo arithmetic so the counter wraps correctly across videos:
    calls_per_video = num_steps * num_layers = 50 * 30 = 1500
    current_step = (call_count % calls_per_video) // num_layers
    """

    def __init__(self, dense_steps=10, chunk_size=3, moba_topk=3,
                 simsum_threshold=0.25, video_length=81, resolution=(480, 832)):
        self.dense_steps = dense_steps
        self.chunk_size = chunk_size
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

        # Step 5: Attention (dense or VMoBA sparse)
        T, H, W = get_video_dims(seq_len, self.video_length, self.resolution)

        if use_sparse and T is not None:
            try:
                moba_fn = self._get_moba()
                moba_chunk_size = self.chunk_size * H * W

                q_flat = rearrange(query, "b s h d -> (b s) h d")
                k_flat = rearrange(key, "b s h d -> (b s) h d")
                v_flat = rearrange(value, "b s h d -> (b s) h d")

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

                out = rearrange(out_flat, "(b s) h d -> b s (h d)", b=batch_size)

            except Exception as e:
                import traceback
                print(f"[HybridVMoBA] ERROR in sparse attention: {e}")
                traceback.print_exc()
                # Fallback to dense
                query_t = query.transpose(1, 2).contiguous()
                key_t = key.transpose(1, 2).contiguous()
                value_t = value.transpose(1, 2).contiguous()
                out = torch.nn.functional.scaled_dot_product_attention(
                    query_t, key_t, value_t, attn_mask=attention_mask
                )
                out = out.transpose(1, 2).flatten(2)
        else:
            # Dense SDPA for early steps or unknown layout
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
    """Hybrid dense→VMoBA: dense for early steps, sparse for later."""
    params = config.params

    chunk_size = params.tile_size_t if params.tile_size_t is not None else 3
    moba_topk = params.block_size if params.block_size is not None else 3
    simsum_threshold = params.sparsity_ratio if params.sparsity_ratio is not None else 0.25
    # dense_steps: how many initial denoising steps use dense attention
    dense_steps = params.tile_size_h if hasattr(params, 'tile_size_h') and params.tile_size_h is not None else 10

    video_length = config.inference.video_length
    resolution = tuple(config.inference.resolution)

    # Set num_steps for modulo counter
    _num_steps[0] = config.inference.num_inference_steps
    # Reset counter at patch time
    _global_call_count[0] = 0

    proc = HybridVMoBAProcessor(
        dense_steps=dense_steps,
        chunk_size=chunk_size, moba_topk=moba_topk,
        simsum_threshold=simsum_threshold,
        video_length=video_length, resolution=resolution,
    )

    def patch_fn(original_forward, layer_name, module):
        module.set_processor(proc)
        return original_forward

    return patch_fn
