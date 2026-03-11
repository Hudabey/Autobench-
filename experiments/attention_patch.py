"""
Attention Patch — wires sparse attention into Wan2.1-1.3B.
✏️ AGENT EDITS THIS FILE

Must define: create_patch(config) -> patch_fn
  patch_fn(original_forward, layer_name, module) -> new_forward

──────────────────────────────────────────────────────────────
Wan2.1 Self-Attention Pipeline (WanAttnProcessor2_0)
──────────────────────────────────────────────────────────────

The REAL pipeline inside each self-attention layer is:

  1. Q/K/V projection:  query = attn.to_q(x), key = attn.to_k(x), value = attn.to_v(x)
  2. Head reshape:       (B, seq, dim) → (B, seq, heads, head_dim)
  3. QK normalization:   query = attn.norm_q(query), key = attn.norm_k(key)   ← RMSNorm
  4. RoPE:               query, key = apply_rotary_emb(query, key, freqs)
  5. Attention kernel:   out = F.scaled_dot_product_attention(q, k, v)
  6. Output projection:  out = attn.to_out[0](out), out = attn.to_out[1](out)

YOUR sparse method replaces ONLY step 5. Steps 1-4 and 6 MUST be preserved
exactly, or you'll get garbage (wrong positions, wrong norms, wrong text).

NOTE: The harness only patches self-attention (blocks.{i}.attn1).
Cross-attention (blocks.{i}.attn2) is never touched — text conditioning
is always preserved.

RECOMMENDED: Pattern A (processor replacement)
ADVANCED:    Pattern B (full forward replacement)
──────────────────────────────────────────────────────────────

CURRENT: Dense baseline (pass-through)
"""

import torch


def create_patch(config):
    """Dense baseline — returns original forward unchanged."""

    def patch_fn(original_forward, layer_name, module):
        return original_forward

    return patch_fn


# ═══════════════════════════════════════════════════════════════════
# PATTERN A — Processor replacement (RECOMMENDED)
#
# Replace the processor object on the attention module.
# You get full control over the attention kernel while the harness
# handles layer selection and timing.
#
# CRITICAL: You MUST include QK normalization and RoPE.
# Skipping either produces garbage outputs.
#
# from diffusers.models.attention_processor import Attention
#
# class SparseProcessor:
#     """Drop-in replacement for WanAttnProcessor2_0.
#
#     Steps 1-4 and 6 are copied exactly from the real processor.
#     Only step 5 (the attention kernel) is replaced with your method.
#     """
#
#     def __init__(self, block_size=64):
#         self.block_size = block_size
#
#     def __call__(self, attn, hidden_states, *args,
#                  encoder_hidden_states=None, attention_mask=None,
#                  rotary_emb=None, **kwargs):
#
#         input_states = (encoder_hidden_states
#                         if encoder_hidden_states is not None
#                         else hidden_states)
#         batch_size, seq_len, _ = hidden_states.shape
#
#         # Step 1: Q/K/V projection
#         query = attn.to_q(hidden_states)
#         key = attn.to_k(input_states)
#         value = attn.to_v(input_states)
#
#         # Step 2: Head reshape — (B, seq, dim) → (B, seq, heads, head_dim)
#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads
#         query = query.view(batch_size, -1, attn.heads, head_dim)
#         key = key.view(batch_size, -1, attn.heads, head_dim)
#         value = value.view(batch_size, -1, attn.heads, head_dim)
#
#         # Step 3: QK normalization (RMSNorm) ── DO NOT SKIP ──
#         if hasattr(attn, "norm_q") and attn.norm_q is not None:
#             query = attn.norm_q(query)
#         if hasattr(attn, "norm_k") and attn.norm_k is not None:
#             key = attn.norm_k(key)
#
#         # Step 4: Rotary Position Embeddings ── DO NOT SKIP ──
#         if rotary_emb is not None:
#             from diffusers.models.embeddings import apply_rotary_emb
#             query = apply_rotary_emb(query, rotary_emb)
#             key = apply_rotary_emb(key, rotary_emb)
#
#         # Transpose for attention: (B, seq, heads, hd) → (B, heads, seq, hd)
#         query = query.transpose(1, 2)
#         key = key.transpose(1, 2)
#         value = value.transpose(1, 2)
#
#         # ════════════════════════════════════════════
#         # Step 5: YOUR SPARSE ATTENTION KERNEL HERE
#         #
#         # Replace this with your method:
#         #   out = my_sparse_attn(query, key, value,
#         #                        block_size=self.block_size)
#         #
#         # Shapes at this point:
#         #   query: (B, heads, seq_len, head_dim)
#         #   key:   (B, heads, seq_len, head_dim)
#         #   value: (B, heads, seq_len, head_dim)
#         #
#         # Must return: (B, heads, seq_len, head_dim)
#         #
#         # Default: standard SDPA (dense, for reference)
#         out = torch.nn.functional.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask
#         )
#         # ════════════════════════════════════════════
#
#         # Reshape back: (B, heads, seq, hd) → (B, seq, dim)
#         out = out.transpose(1, 2).reshape(batch_size, seq_len, inner_dim)
#
#         # Step 6: Output projection
#         out = attn.to_out[0](out)   # Linear
#         out = attn.to_out[1](out)   # Dropout
#
#         return out
#
#
# def create_patch(config):
#     block_size = config.params.block_size or 64
#     proc = SparseProcessor(block_size=block_size)
#
#     def patch_fn(original_forward, layer_name, module):
#         module.set_processor(proc)
#         return original_forward  # forward unchanged, processor swapped
#
#     return patch_fn
#
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# PATTERN B — Full forward replacement (ADVANCED)
#
# Use when you need control flow around attention (e.g., step-adaptive:
# dense for early denoising steps, sparse for late steps).
#
# def create_patch(config):
#     switch_step = config.params.extra.get("switch_step", 20)
#
#     def patch_fn(original_forward, layer_name, module):
#         step_counter = [0]
#
#         def new_forward(hidden_states, *args, **kwargs):
#             if step_counter[0] < switch_step:
#                 step_counter[0] += 1
#                 return original_forward(hidden_states, *args, **kwargs)
#             step_counter[0] += 1
#             # ... sparse implementation (must include QK norm + RoPE) ...
#             return result
#
#         return new_forward
#     return patch_fn
# ═══════════════════════════════════════════════════════════════════
