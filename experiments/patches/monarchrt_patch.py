"""
MonarchRT sparse attention for Wan2.1-1.3B (training-free).

MonarchRT decomposes attention using Monarch matrices, exploiting the 3D
(T, H, W) structure of video tokens. With f_tied=1, h_reduce=1, w_reduce=1,
it creates a two-level hierarchical attention:
- Level R: attention across 21 frame blocks, each of size (30, 52)
- Level L: attention within frames, across 30 "columns" of size 52

Input format: (B, seq_len, num_heads, head_dim)
Our dims: T=21, H=30, W=52 -> seq_len=32,760

30 layers, 50 steps, modulo counter for multi-video.
"""

import sys
import importlib.util
import torch

# Direct import of monarch_attn to avoid wan package dependency issues
_spec = importlib.util.spec_from_file_location(
    'monarch_attn',
    '/home/researcher/autobench_work/methods/monarchrt/wan/modules/monarch_attn.py'
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
monarch_attn_fn = _mod.monarch_attn

# Global step counter
_global_call_count = [0]
_num_layers = [30]
_num_steps = [50]


class MonarchRTProcessor:
    """Pure MonarchRT attention processor for Wan2.1-1.3B."""

    def __init__(self, f_tied=1, h_reduce=1, w_reduce=1,
                 grid_h=30, grid_w=52, num_iters=1):
        self.f_tied = f_tied
        self.h_reduce = h_reduce
        self.w_reduce = w_reduce
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_iters = num_iters

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        batch_size, seq_len, _ = hidden_states.shape

        _global_call_count[0] += 1

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

        # Step 5: MonarchRT attention
        # monarch_attn expects (B, seq, heads, head_dim)
        # query/key/value are already in that format after unflatten
        try:
            with torch.no_grad():
                out = monarch_attn_fn(
                    query.contiguous(), key.contiguous(), value.contiguous(),
                    f_tied=self.f_tied,
                    h_reduce=self.h_reduce,
                    w_reduce=self.w_reduce,
                    h=self.grid_h,
                    w=self.grid_w,
                    num_iters=self.num_iters,
                )
        except Exception as e:
            import traceback
            print(f"[MonarchRT] ERROR: {e}")
            traceback.print_exc()
            # Fallback to dense SDPA
            qt = query.transpose(1, 2).contiguous()
            kt = key.transpose(1, 2).contiguous()
            vt = value.transpose(1, 2).contiguous()
            out = torch.nn.functional.scaled_dot_product_attention(
                qt, kt, vt, attn_mask=attention_mask
            )
            out = out.transpose(1, 2)

        # Reshape: (B, seq, heads, hd) -> (B, seq, dim)
        out = out.flatten(2)
        out = out.type_as(hidden_states)

        # Step 6: Output projection
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out


class CombinedMonarchRTProcessor:
    """Dense for early steps; after that, dense on boundary layers, MonarchRT on middle."""

    def __init__(self, dense_steps=10, layer_idx=0, dense_layer_boundary=5,
                 num_total_layers=30, f_tied=1, h_reduce=1, w_reduce=1,
                 grid_h=30, grid_w=52, num_iters=1):
        self.dense_steps = dense_steps
        self.layer_idx = layer_idx
        self.dense_layer_boundary = dense_layer_boundary
        self.num_total_layers = num_total_layers
        self.is_boundary = (layer_idx < dense_layer_boundary) or \
                           (layer_idx >= num_total_layers - dense_layer_boundary)
        self.f_tied = f_tied
        self.h_reduce = h_reduce
        self.w_reduce = w_reduce
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_iters = num_iters

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        batch_size, seq_len, _ = hidden_states.shape

        # Determine current denoising step
        calls_per_video = _num_steps[0] * _num_layers[0]
        current_step = (_global_call_count[0] % calls_per_video) // _num_layers[0]
        _global_call_count[0] += 1

        # Use sparse only if: past dense_steps AND not a boundary layer
        use_sparse = (current_step >= self.dense_steps) and (not self.is_boundary)

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
        if use_sparse:
            try:
                with torch.no_grad():
                    out = monarch_attn_fn(
                        query.contiguous(), key.contiguous(), value.contiguous(),
                        f_tied=self.f_tied,
                        h_reduce=self.h_reduce,
                        w_reduce=self.w_reduce,
                        h=self.grid_h,
                        w=self.grid_w,
                        num_iters=self.num_iters,
                    )
            except Exception as e:
                import traceback
                print(f"[CombinedMonarchRT] ERROR: {e}")
                traceback.print_exc()
                qt = query.transpose(1, 2).contiguous()
                kt = key.transpose(1, 2).contiguous()
                vt = value.transpose(1, 2).contiguous()
                out = torch.nn.functional.scaled_dot_product_attention(
                    qt, kt, vt, attn_mask=attention_mask
                )
                out = out.transpose(1, 2)
        else:
            # Dense SDPA
            qt = query.transpose(1, 2).contiguous()
            kt = key.transpose(1, 2).contiguous()
            vt = value.transpose(1, 2).contiguous()
            out = torch.nn.functional.scaled_dot_product_attention(
                qt, kt, vt, attn_mask=attention_mask
            )
            out = out.transpose(1, 2)

        # Reshape: (B, seq, heads, hd) -> (B, seq, dim)
        out = out.flatten(2)
        out = out.type_as(hidden_states)

        # Step 6: Output projection
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out


def create_patch(config):
    """MonarchRT attention patch for Wan2.1-1.3B."""
    params = config.params

    f_tied = params.tile_size_t if hasattr(params, 'tile_size_t') and params.tile_size_t is not None else 1
    dense_steps = params.tile_size_h if hasattr(params, 'tile_size_h') and params.tile_size_h is not None else 0
    dense_layer_boundary = params.tile_size_w if hasattr(params, 'tile_size_w') and params.tile_size_w is not None else 0
    num_iters = params.block_size if hasattr(params, 'block_size') and params.block_size is not None else 1

    # Grid dimensions for Wan2.1 at 81 frames, 480x832
    video_length = config.inference.video_length
    resolution = tuple(config.inference.resolution)
    grid_h = resolution[0] // 16  # VAE downscale: 480/16 = 30
    grid_w = resolution[1] // 16  # VAE downscale: 832/16 = 52

    num_total_layers = 30
    _num_steps[0] = config.inference.num_inference_steps
    _global_call_count[0] = 0

    use_combined = (dense_steps > 0 or dense_layer_boundary > 0)

    def patch_fn(original_forward, layer_name, module):
        if use_combined:
            # Extract layer index
            try:
                parts = layer_name.split('.')
                layer_idx = int(parts[1])
            except (IndexError, ValueError):
                layer_idx = 0

            proc = CombinedMonarchRTProcessor(
                dense_steps=dense_steps,
                layer_idx=layer_idx,
                dense_layer_boundary=dense_layer_boundary,
                num_total_layers=num_total_layers,
                f_tied=f_tied,
                grid_h=grid_h,
                grid_w=grid_w,
                num_iters=num_iters,
            )
        else:
            proc = MonarchRTProcessor(
                f_tied=f_tied,
                grid_h=grid_h,
                grid_w=grid_w,
                num_iters=num_iters,
            )
        module.set_processor(proc)
        return original_forward

    return patch_fn
