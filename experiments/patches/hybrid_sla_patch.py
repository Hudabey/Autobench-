"""
Hybrid Dense→SLA for Wan2.1-1.3B.
Dense SDPA for first N denoising steps, SLA sparse for remaining.

Same hybrid approach that worked brilliantly for VMoBA:
- Dense early steps establish global structure (full context)
- SLA sparse later steps refine details (block-selected context sufficient)

30 layers, 50 steps, modulo counter for multi-video.
"""

import sys
import torch

sys.path.insert(0, "/home/researcher/autobench_work/methods/sla1")


# Global step counter
_global_call_count = [0]
_num_layers = [30]
_num_steps = [50]


class HybridSLAProcessor:
    """Dense for early steps; SLA sparse for later steps."""

    def __init__(self, dense_steps=10, topk_ratio=0.3, block_size=64,
                 video_length=81, resolution=(480, 832)):
        self.dense_steps = dense_steps
        self.topk_ratio = topk_ratio
        self.block_size = block_size
        self.video_length = video_length
        self.resolution = resolution
        self._sla_module = None

    def _get_sla(self, device):
        if self._sla_module is None:
            from sparse_linear_attention import SparseLinearAttention
            self._sla_module = SparseLinearAttention(
                head_dim=128,
                topk=self.topk_ratio,
                feature_map='softmax',
                BLKQ=self.block_size,
                BLKK=self.block_size,
                use_bf16=True,
            ).to(device)
            self._sla_module.eval()
        return self._sla_module

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        batch_size, seq_len, _ = hidden_states.shape

        # Determine current denoising step
        calls_per_video = _num_steps[0] * _num_layers[0]
        current_step = (_global_call_count[0] % calls_per_video) // _num_layers[0]
        _global_call_count[0] += 1

        use_sparse = (current_step >= self.dense_steps)

        # Step 1: Q/K/V projection
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Step 2: QK normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Step 3: Head reshape -> (B, seq, heads, head_dim)
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

        # Step 5: Attention
        # Transpose: (B, seq, heads, hd) -> (B, heads, seq, hd)
        query_t = query.transpose(1, 2).contiguous()
        key_t = key.transpose(1, 2).contiguous()
        value_t = value.transpose(1, 2).contiguous()

        if use_sparse:
            try:
                sla = self._get_sla(query_t.device)
                with torch.no_grad():
                    out = sla(query_t, key_t, value_t)
            except Exception as e:
                import traceback
                print(f"[HybridSLA] ERROR: {e}")
                traceback.print_exc()
                out = torch.nn.functional.scaled_dot_product_attention(
                    query_t, key_t, value_t, attn_mask=attention_mask
                )
        else:
            # Dense SDPA
            out = torch.nn.functional.scaled_dot_product_attention(
                query_t, key_t, value_t, attn_mask=attention_mask
            )

        # Reshape: (B, heads, seq, hd) -> (B, seq, dim)
        out = out.transpose(1, 2).flatten(2)
        out = out.type_as(hidden_states)

        # Step 6: Output projection
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out


def create_patch(config):
    """Hybrid dense→SLA sparse attention patch."""
    params = config.params

    topk_ratio = params.sparsity_ratio if params.sparsity_ratio is not None else 0.3
    block_size = params.block_size if params.block_size is not None else 64
    dense_steps = params.tile_size_h if hasattr(params, 'tile_size_h') and params.tile_size_h is not None else 10

    video_length = config.inference.video_length
    resolution = tuple(config.inference.resolution)

    _num_steps[0] = config.inference.num_inference_steps
    _global_call_count[0] = 0

    proc = HybridSLAProcessor(
        dense_steps=dense_steps,
        topk_ratio=topk_ratio, block_size=block_size,
        video_length=video_length, resolution=resolution,
    )

    def patch_fn(original_forward, layer_name, module):
        module.set_processor(proc)
        return original_forward

    return patch_fn
