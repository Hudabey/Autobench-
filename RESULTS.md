# AutoBench Results: Sparse Attention Benchmarking on Wan2.1-1.3B

## Executive Summary

We autonomously benchmarked 4 sparse attention methods (NABLA, VMoBA, PISA, SLA) on the Wan2.1-1.3B video diffusion transformer across 26 experiments. Our key finding: **SLA (Sparse-Linear Attention) outperforms VMoBA across every configuration** when using the same hybrid/combined strategy. The best overall result is a **combined timestep + layer-selective SLA** approach — using dense attention for early denoising steps AND on boundary transformer layers — which achieves **3.06x speedup** and a **composite score of 0.373** (FID=74.7, LPIPS=0.222). This is a **5.4% improvement** over the previous best combined VMoBA(b=8) (composite=0.354 at 3.04x) and a **277% improvement** over pure VMoBA (composite=0.099 at 3.64x).

**Major finding: SLA dominates VMoBA at every operating point.** SLA's block-level QK score top-k selection mechanism is fundamentally more effective than VMoBA's temporal chunk selection for video diffusion attention sparsification. Despite `methods.yaml` warning about 0% negligible attention weights for SLA (compared to SLA's own prediction of 45%), SLA works well empirically because its sparse block attention with top-k=0.3 retention already captures the most important attention patterns.

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 5090 |
| Model | Wan2.1-1.3B (30 self-attention layers, 12 heads, 128 head_dim) |
| Sequence length | 32,760 tokens (T=21, H=30, W=52) |
| Video config | 81 frames, 480x832, 50 denoising steps |
| PyTorch | 2.x with flash-attn 2.8.3, Triton 3.4.0 |

## Methods Tested

### 1. NABLA/STA (3D Sliding Window)
- **Approach**: Each token attends to neighbors within +/-wT frames, +/-wH rows, +/-wW cols
- **Backend**: PyTorch `flex_attention` with compiled `BlockMask`
- **Status**: Working
- **Best result**: 4.23x speedup, composite=0.085 (pure) — fastest but worst quality

### 2. VMoBA (Mixture-of-Block Attention)
- **Approach**: Splits KV into temporal chunks, selects top-k relevant chunks per query block
- **Backend**: flash-attn `varlen` API (required compatibility shim for v2.8.3)
- **Status**: Working (with flash-attn API monkey-patch)
- **Best result**: 3.04x speedup, composite=0.354 (combined timestep+layer, b=8)

### 3. PISA (Piecewise Sparse Attention)
- **Approach**: Block-level top-k selection via Triton kernels
- **Backend**: Custom Triton kernels (requires >=3.5.1, Hopper architecture)
- **Status**: INCOMPATIBLE — crashes on RTX 5090 with Triton 3.4.0

### 4. SLA (Sparse-Linear Attention) — NEW BEST METHOD
- **Approach**: Block-level QK score computation with top-k selection (topk=0.3 means 30% of blocks retained). Has a linear attention branch with zero-initialized `proj_l`, but without fine-tuning this branch contributes nothing — effectively just sparse block attention.
- **Backend**: Custom Triton kernels for block-level QK scoring + standard attention on selected blocks
- **Configuration**: block_size=64, topk=0.3 (30% key retention)
- **Status**: Working — **best overall results across all configurations**
- **Best result**: 3.06x speedup, composite=0.373 (combined timestep+layer, b=8) — OVERALL BEST
- **Key insight**: Despite `methods.yaml` reporting 0% negligible attention weights (vs SLA's own prediction of 45%), SLA works well empirically. The linear attention branch (`proj_l`) is zero-initialized and contributes nothing without training, so SLA is effectively just sparse block attention with QK-score-based top-k selection. This block-level QK scoring is more effective than VMoBA's temporal chunk selection because it directly measures query-key relevance at block granularity rather than relying on temporal proximity as a proxy.

### 5. Hybrid Dense->Sparse (Novel)
- **Approach**: Dense SDPA for first N denoising steps, sparse attention for remaining steps
- **Insight**: Early steps establish global structure (need full context), later steps refine details (local context sufficient)
- **Implementation**: Global call counter with modulo arithmetic for multi-video correctness
- **Status**: Working — strong results with both VMoBA and SLA backends

### 6. Layer-Selective Sparse (Novel)
- **Approach**: Dense attention on boundary layers (first and last N layers of the 30-layer transformer), sparse attention on middle layers
- **Insight**: Boundary layers handle low-level features and output projection where full context matters most; middle layers handle higher-level semantics where sparse attention is sufficient
- **Implementation**: Layer index check — layers 0..(boundary-1) and (30-boundary)..29 use dense SDPA, middle layers use sparse
- **Status**: Working — moderate quality improvement over pure sparse methods

### 7. Combined Timestep + Layer-Selective (Novel — BEST STRATEGY)
- **Approach**: Dense everywhere for first 10 denoising steps, then dense on boundary layers + sparse on middle layers for remaining steps
- **Insight**: Combines the benefits of both temporal and spatial selectivity — early steps get full dense attention for global structure, later steps use dense on critical boundary layers and sparse on middle layers
- **Implementation**: Two-dimensional selectivity: timestep counter AND layer index routing
- **Status**: Working — **best overall results with SLA backend (composite=0.373, 3.06x speedup)**

## Results Table

All experiments use fast-mode evaluation (9 prompts, FID + LPIPS metrics) unless noted.

| # | Experiment | Phase | Speedup | FID (lower=better) | LPIPS (lower=better) | Composite (higher=better) | Pareto |
|---|-----------|-------|---------|-------|---------|-------------|--------|
| 1 | Dense baseline (full, 25 videos) | 1 | 1.00x | 0.0 | 0.000 | 1.000 | ref |
| 2-3 | NABLA default | 2 | -- | -- | -- | -- | FAILED (disk) |
| 4 | **NABLA default** (wT=5,wH=4,wW=4) | 2 | **4.23x** | 367.0 | 0.881 | 0.085 | yes |
| 5 | PISA default | 2 | 2.75x | ~0 | ~0 | 0.525 | (dense fallback) |
| 6 | VMoBA default | 2 | 2.34x | ~0 | ~0 | 0.525 | (dense fallback) |
| 7 | **VMoBA default** (fixed) | 2 | **3.64x** | 203.4 | 0.657 | 0.099 | yes |
| 8 | PISA (fixed) | 2 | -- | -- | -- | -- | FAILED (Triton crash) |
| 9 | VMoBA topk=5 | 3 | 3.66x | 206.3 | 0.668 | 0.095 | yes |
| 10 | VMoBA thresh=0.10 | 3 | -- | -- | -- | -- | FAILED (disk) |
| 11 | VMoBA thresh=0.10 (retry) | 3 | 3.65x | 209.9 | 0.663 | 0.097 | yes |
| 12 | NABLA wide (wT=10,wH=10,wW=10) | 3 | 2.85x | 300.7 | 0.767 | 0.085 | no |
| 13 | **Hybrid dense(10)->VMoBA** | 4 | **3.43x** | **157.6** | **0.449** | **0.210** | **YES** |
| 14 | Hybrid dense(5)->VMoBA | 4 | 3.54x | 163.8 | 0.453 | 0.202 | yes |
| 15 | **Hybrid dense(20)->VMoBA** | 4 | **3.24x** | **139.3** | **0.362** | **0.258** | **YES** |
| 16 | Hybrid dense(10) full mode | 5 | -- | -- | -- | -- | FAILED (disk full at metrics) |
| 17 | Hybrid dense(10)->NABLA | 4 | 3.81x | 363.5 | 0.818 | 0.085 | no |
| 18 | Hybrid dense(10)->NABLA wide | 4 | 2.83x | 230.1 | 0.608 | 0.114 | no |
| 19 | Layer-selective VMoBA (boundary=5) | 4 | 3.29x | 180.1 | 0.553 | 0.153 | no |
| 20 | **Combined VMoBA(steps=10,boundary=5)** | 4 | **3.17x** | **111.2** | **0.319** | **0.302** | **YES** |
| 21 | **Combined VMoBA(steps=10,boundary=8)** | 4 | **3.04x** | **83.3** | **0.251** | **0.354** | **YES** |
| 22 | **Combined VMoBA(steps=10,boundary=3)** | 4 | **3.28x** | **130.2** | **0.374** | **0.264** | **YES** |
| 23 | **SLA default** (topk=0.3, block=64) | 4 | **3.76x** | **166.6** | **0.556** | **0.167** | **YES** |
| 24 | **Hybrid dense(10)->SLA** | 4 | **3.50x** | **109.6** | **0.272** | **0.319** | **YES** |
| 25 | **Combined SLA(steps=10,boundary=5)** | 4 | **3.20x** | **76.7** | **0.245** | **0.364** | **YES** |
| 26 | **Combined SLA(steps=10,boundary=8)** | 4 | **3.06x** | **74.7** | **0.222** | **0.373** | **YES (OVERALL BEST)** |

## Pareto Frontier (Speed vs Quality)

```
Composite Quality
    ^
1.0 |  * Dense baseline (1.0x)
    |
    |
    |
0.37|                           * Combined SLA(10,b=8) (3.06x)  << OVERALL BEST
0.36|                            * Combined SLA(10,b=5) (3.20x)
0.35|                           * Combined VMoBA(10,b=8) (3.04x)
0.32|                              * Hybrid SLA(10) (3.50x)
0.30|                             * Combined VMoBA(10,b=5) (3.17x)
0.26|                              * Hybrid VMoBA(20) (3.24x)
    |                               * Combined VMoBA(10,b=3) (3.28x)
0.21|                                 * Hybrid VMoBA(10) (3.43x)
0.20|                                   * Hybrid VMoBA(5) (3.54x)
0.17|                                    * SLA pure (3.76x)
0.15|                                * LayerSel VMoBA(b=5) (3.29x)
0.10|                                      * VMoBA (3.64x)
0.09|                                       * NABLA (4.23x)
    |
    +----+----+----+----+----+----+----+----> Speedup
         1x   1.5x  2x  2.5x  3x  3.5x  4x
```

**Pareto-optimal frontier** (dominating in both speed and quality):
1. Dense baseline — 1.00x, composite 1.000
2. **Combined SLA(steps=10,boundary=8) — 3.06x, composite 0.373** (OVERALL BEST)
3. Combined SLA(steps=10,boundary=5) — 3.20x, composite 0.364
4. Hybrid dense(10)->SLA — 3.50x, composite 0.319
5. Combined VMoBA(steps=10,boundary=3) — 3.28x, composite 0.264
6. Hybrid dense(5)->VMoBA — 3.54x, composite 0.202
7. SLA default — 3.76x, composite 0.167
8. NABLA default — 4.23x, composite 0.085

Note: Several previously Pareto-optimal VMoBA configurations are now dominated by SLA equivalents. For example, Hybrid VMoBA(10) at 3.43x/0.210 is dominated by Hybrid SLA(10) at 3.50x/0.319 (both faster AND higher quality).

## Key Findings

### 1. SLA dominates VMoBA across every configuration (MAJOR NEW FINDING)

SLA (Sparse-Linear Attention) outperforms VMoBA at every comparable operating point. The gap is consistent and substantial:

| Configuration | SLA Speedup | SLA Composite | VMoBA Speedup | VMoBA Composite | SLA Advantage |
|---------------|------------|---------------|--------------|-----------------|---------------|
| Pure sparse | 3.76x | 0.167 | 3.64x | 0.099 | +3% faster, +69% quality |
| Hybrid(10) | 3.50x | 0.319 | 3.43x | 0.210 | +2% faster, +52% quality |
| Combined(b=5) | 3.20x | 0.364 | 3.17x | 0.302 | +1% faster, +21% quality |
| Combined(b=8) | 3.06x | 0.373 | 3.04x | 0.354 | +1% faster, +5% quality |

Key observations:
- **SLA is both faster AND better quality** in every single comparison — it strictly dominates VMoBA
- The quality gap is largest for pure sparse (69% better composite) and narrows as more dense computation is added (5% better at combined b=8), which makes sense: as both methods approach fully-dense, their differences shrink
- SLA's speed advantage is modest (1-3%) but consistent, likely because block-level QK scoring is slightly cheaper than VMoBA's temporal chunk scoring
- The quality advantage is much more significant, indicating SLA selects better blocks to attend to

### 2. Why SLA beats VMoBA: block-level QK scoring vs temporal chunk selection

The fundamental difference: **SLA directly computes block-level QK relevance scores** and retains the top 30% of blocks, while **VMoBA groups KV by temporal chunks** (3-frame groups) and selects top-k chunks. SLA's approach is more fine-grained and data-dependent:
- SLA evaluates each 64-token block independently based on actual query-key dot product scores
- VMoBA groups by temporal proximity (all tokens from the same 3 frames form a chunk) and selects whole chunks

This means VMoBA may include irrelevant tokens (from selected chunks) and exclude relevant ones (from unselected chunks) more frequently than SLA. SLA's block-level scoring better identifies which specific regions of the KV cache matter most for each query, regardless of temporal position.

### 3. SLA works despite methods.yaml warning about 0% negligible attention weights

The `methods.yaml` profiling flagged SLA as having 0% negligible attention weights (compared to SLA's own estimate of 45%). This initially suggested SLA might not be effective. However, empirical results show otherwise:
- The profiling measures whether attention weights fall below a negligible threshold across the full dense attention matrix
- SLA's effectiveness comes not from exploiting truly negligible weights, but from focusing computation on the **most important** blocks (top 30% by QK score)
- Even when no weights are truly "negligible," there is still significant variance in attention weight magnitudes across blocks — SLA exploits this variance
- The zero-initialized `proj_l` (linear attention projection) contributes nothing without fine-tuning, so SLA operates purely as sparse block attention — and this is sufficient

### 4. Combined timestep + layer-selective is the best strategy (confirmed for both methods)

The combined approach (dense for early steps, dense on boundary layers + sparse on middle layers for late steps) achieves the best results for both SLA and VMoBA:

**SLA Combined Results:**
| Boundary Layers | Dense Layers (sparse phase) | Sparse Layers | Speedup | FID | LPIPS | Composite |
|----------------|---------------------------|---------------|---------|-----|-------|-----------|
| 5 (each end) | 10/30 (33%) | 20/30 (67%) | 3.20x | 76.7 | 0.245 | 0.364 |
| 8 (each end) | 16/30 (53%) | 14/30 (47%) | 3.06x | 74.7 | 0.222 | 0.373 |

**VMoBA Combined Results:**
| Boundary Layers | Dense Layers (sparse phase) | Sparse Layers | Speedup | FID | LPIPS | Composite |
|----------------|---------------------------|---------------|---------|-----|-------|-----------|
| 3 (each end) | 6/30 (20%) | 24/30 (80%) | 3.28x | 130.2 | 0.374 | 0.264 |
| 5 (each end) | 10/30 (33%) | 20/30 (67%) | 3.17x | 111.2 | 0.319 | 0.302 |
| 8 (each end) | 16/30 (53%) | 14/30 (47%) | 3.04x | 83.3 | 0.251 | 0.354 |

The quality improvement from boundary=5 to boundary=8 costs only 4-5% speedup but yields meaningful quality gains in both methods.

### 5. Hybrid attention is dramatically better than pure sparse

The hybrid dense->sparse approach achieves **2-3x better composite scores** than pure sparse, with only a small speedup penalty:
- Pure VMoBA: FID=203, LPIPS=0.66, composite=0.099, speedup=3.64x
- Hybrid VMoBA(10): FID=158 (**22% better FID**), LPIPS=0.45 (**32% better**), composite=0.210 (**2.1x**), speedup=3.43x
- Pure SLA: FID=167, LPIPS=0.56, composite=0.167, speedup=3.76x
- Hybrid SLA(10): FID=110 (**34% better FID**), LPIPS=0.27 (**51% better**), composite=0.319 (**1.9x**), speedup=3.50x
- **Combined SLA(10,b=8): FID=75 (**55% better than pure SLA**), LPIPS=0.22 (**60% better**), composite=0.373 (**2.2x**), speedup=3.06x**

### 6. Layer-selective attention alone is moderately effective
Experiment #19 (layer-selective VMoBA with boundary=5, no timestep gating) achieved composite=0.153 at 3.29x — better than pure VMoBA (0.099/3.64x) but much worse than timestep-hybrid approaches. This confirms that temporal selectivity (which denoising steps to use sparse attention on) matters more than spatial selectivity (which layers), but combining both yields the best results.

### 7. VMoBA parameter tuning is saturated
Changing topk (3->5) or simsum_threshold (0.25->0.10) has negligible effect on latency (~2.70s/step). Runtime is dominated by fixed kernel overhead, not the theoretical sparsity ratio.

### 8. NABLA has the highest raw speedup but worst quality
NABLA/STA at 4.23x is the fastest, but FID=367 and LPIPS=0.88 are very poor. Even with wider windows (10,10,10), quality only improves to FID=301 while speedup drops to 2.85x — not competitive with VMoBA or SLA.

### 9. PISA is architecture-dependent
PISA's Triton kernels require Hopper (H100/H800) architecture and Triton >=3.5.1. They crash on RTX 5090 (Blackwell) with Triton 3.4.0. This is a fundamental compatibility issue, not a configuration problem.

### 10. Data-dependent sparsity benefits far more from hybrid than fixed-window
Hybrid dense->VMoBA: FID drops from 203->158 (**22% improvement**).
Hybrid dense->NABLA: FID drops from 367->364 (**0.8% improvement**).
The same 10 dense early steps that transformed VMoBA barely helped NABLA. This is because VMoBA's data-dependent block selection preserves the global structure established during dense steps, while NABLA's rigid +/-5 spatial window immediately discards distant correlations regardless of what was computed in earlier steps. SLA benefits even more from hybrid (FID drops from 167->110, **34% improvement**), further confirming that data-dependent methods gain the most from hybrid approaches.

### 11. Dense early steps are critical for quality
The diminishing returns curve on dense_steps shows that the first 5 dense steps capture most of the quality gain:
- 0->5 dense steps: FID improves 203->164 (**19%**)
- 5->10 dense steps: FID improves 164->158 (**4%**)
- 10->20 dense steps: FID improves 158->139 (**12%**)

### 12. flash-attn API compatibility requires careful handling
flash-attn 2.8.3 changed `_flash_attn_varlen_forward` from 8 return values to 4. VMoBA expects the old 8-value API. A monkey-patch compatibility shim (in `vmoba_patch.py`) resolves this transparently.

## Detailed Analysis: SLA vs VMoBA Head-to-Head

### Pure Sparse Comparison

| Metric | SLA (topk=0.3, block=64) | VMoBA (topk=3, chunk=3) | Winner |
|--------|--------------------------|-------------------------|--------|
| Speedup | 3.76x | 3.64x | SLA (+3.3%) |
| FID | 166.6 | 203.4 | SLA (-18.1%) |
| LPIPS | 0.556 | 0.657 | SLA (-15.4%) |
| Composite | 0.167 | 0.099 | SLA (+68.7%) |

SLA achieves nearly **70% better composite quality** while being 3.3% faster in the pure sparse configuration. This is the largest relative gap between the two methods. At the pure sparse level, block selection quality matters most because every single attention call uses sparse attention.

### Hybrid (Dense 10 Steps) Comparison

| Metric | Hybrid SLA(10) | Hybrid VMoBA(10) | Winner |
|--------|----------------|------------------|--------|
| Speedup | 3.50x | 3.43x | SLA (+2.0%) |
| FID | 109.6 | 157.6 | SLA (-30.5%) |
| LPIPS | 0.272 | 0.449 | SLA (-39.4%) |
| Composite | 0.319 | 0.210 | SLA (+51.9%) |

With 10 dense steps, SLA's advantage narrows slightly but remains very large: **52% better composite**. The FID gap is particularly striking — SLA achieves FID=110 vs VMoBA's FID=158, a 30% improvement.

### Combined (Dense Steps + Boundary Layers) Comparison

| Metric | Combined SLA(b=5) | Combined VMoBA(b=5) | Winner |
|--------|-------------------|---------------------|--------|
| Speedup | 3.20x | 3.17x | SLA (+0.9%) |
| FID | 76.7 | 111.2 | SLA (-31.0%) |
| LPIPS | 0.245 | 0.319 | SLA (-23.2%) |
| Composite | 0.364 | 0.302 | SLA (+20.5%) |

| Metric | Combined SLA(b=8) | Combined VMoBA(b=8) | Winner |
|--------|-------------------|---------------------|--------|
| Speedup | 3.06x | 3.04x | SLA (+0.7%) |
| FID | 74.7 | 83.3 | SLA (-10.3%) |
| LPIPS | 0.222 | 0.251 | SLA (-11.6%) |
| Composite | 0.373 | 0.354 | SLA (+5.4%) |

As more dense computation is added (combined b=8 means 53% of layers are dense in the sparse phase, and 20% of steps are fully dense), the gap narrows to 5.4% — but SLA still wins on every metric.

### Why the Gap Narrows with More Dense Computation

The convergence pattern makes intuitive sense: as the fraction of attention calls using sparse methods decreases (from 100% in pure sparse, to ~80% in hybrid, to ~37% in combined b=8), the impact of sparse method quality on overall output quality decreases proportionally. At the limit where all attention is dense, both methods produce identical results. The key insight is that SLA's advantage is most impactful precisely where it matters most — in the sparse attention calls themselves.

## Detailed Analysis: Combined Timestep + Layer-Selective Approach

### Why It Works

The combined approach exploits two orthogonal axes of attention criticality:

1. **Temporal axis (denoising steps)**: Early steps (0-9 of 50) establish the global composition, layout, and structure of the video. These steps require full attention context because the model is making high-level decisions about object placement, camera motion, and scene layout. Later steps (10-49) refine textures, sharpness, and fine details — tasks that are more local and tolerate sparse attention.

2. **Spatial axis (transformer layers)**: The 30 self-attention layers of Wan2.1-1.3B serve different roles. Boundary layers (near the input and output) handle low-level spatial features and the final projection back to pixel space. These layers benefit most from full attention context. Middle layers handle higher-level semantic features where local (sparse) attention is often sufficient.

By applying dense attention only where it matters most (early steps everywhere, boundary layers in late steps), the combined approach achieves quality close to dense attention while maintaining >3x speedup.

### Boundary Layer Sensitivity Curve (VMoBA)

| Boundary Layers | Dense Layers (sparse phase) | Sparse Layers | Speedup | FID | LPIPS | Composite |
|----------------|---------------------------|---------------|---------|-----|-------|-----------|
| 3 (each end) | 6/30 (20%) | 24/30 (80%) | 3.28x | 130.2 | 0.374 | 0.264 |
| 5 (each end) | 10/30 (33%) | 20/30 (67%) | 3.17x | 111.2 | 0.319 | 0.302 |
| 8 (each end) | 16/30 (53%) | 14/30 (47%) | 3.04x | 83.3 | 0.251 | 0.354 |

### Boundary Layer Sensitivity Curve (SLA)

| Boundary Layers | Dense Layers (sparse phase) | Sparse Layers | Speedup | FID | LPIPS | Composite |
|----------------|---------------------------|---------------|---------|-----|-------|-----------|
| 5 (each end) | 10/30 (33%) | 20/30 (67%) | 3.20x | 76.7 | 0.245 | 0.364 |
| 8 (each end) | 16/30 (53%) | 14/30 (47%) | 3.06x | 74.7 | 0.222 | 0.373 |

The quality improvement from boundary=5 to boundary=8 (composite 0.364->0.373, +2.5%) costs only 4% speedup (3.20x->3.06x). This suggests the middle layers (around layers 10-20) contribute relatively little to quality when using sparse attention, making them ideal candidates for sparsification.

### Comparison: All Sparse Attention Variants

| Approach | Configuration | Speedup | FID | LPIPS | Composite | Improvement vs Pure VMoBA |
|----------|--------------|---------|-----|-------|-----------|--------------------------|
| Pure VMoBA | topk=3, all layers, all steps | 3.64x | 203.4 | 0.657 | 0.099 | baseline |
| **Pure SLA** | topk=0.3, block=64, all layers, all steps | **3.76x** | **166.6** | **0.556** | **0.167** | **+69% composite** |
| Timestep-hybrid VMoBA(5) | dense 5 steps, VMoBA 45 steps | 3.54x | 163.8 | 0.453 | 0.202 | +104% composite |
| Timestep-hybrid VMoBA(10) | dense 10 steps, VMoBA 40 steps | 3.43x | 157.6 | 0.449 | 0.210 | +112% composite |
| **Timestep-hybrid SLA(10)** | dense 10 steps, SLA 40 steps | **3.50x** | **109.6** | **0.272** | **0.319** | **+222% composite** |
| Timestep-hybrid VMoBA(20) | dense 20 steps, VMoBA 30 steps | 3.24x | 139.3 | 0.362 | 0.258 | +161% composite |
| Layer-selective VMoBA(b=5) | dense boundary 5, VMoBA middle, all steps | 3.29x | 180.1 | 0.553 | 0.153 | +55% composite |
| Combined VMoBA(10,b=3) | dense 10 steps + boundary 3 layers | 3.28x | 130.2 | 0.374 | 0.264 | +167% composite |
| Combined VMoBA(10,b=5) | dense 10 steps + boundary 5 layers | 3.17x | 111.2 | 0.319 | 0.302 | +205% composite |
| Combined VMoBA(10,b=8) | dense 10 steps + boundary 8 layers | 3.04x | 83.3 | 0.251 | 0.354 | +258% composite |
| **Combined SLA(10,b=5)** | dense 10 steps + boundary 5 layers | **3.20x** | **76.7** | **0.245** | **0.364** | **+268% composite** |
| **Combined SLA(10,b=8)** | dense 10 steps + boundary 8 layers | **3.06x** | **74.7** | **0.222** | **0.373** | **+277% composite (BEST)** |

## Technical Fixes Required

### flash-attn 2.8.3 API Shim (for VMoBA)
```python
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
```

### Triton 3.4.0 Allocator (for PISA — still crashes)
```python
import triton
from triton.runtime._allocation import Allocator
class TorchCUDAAllocator(Allocator):
    def __call__(self, size, alignment, stream):
        return torch.cuda.caching_allocator_alloc(size, stream)
triton.set_allocator(TorchCUDAAllocator())
```

## Experiment Patches

All attention patches are saved in `experiments/patches/`:
- `nabla_sta.py` — NABLA/STA 3D sliding window via flex_attention
- `vmoba_patch.py` — VMoBA mixture-of-block with flash-attn compat fix
- `pisa_patch.py` — PISA with Triton allocator fix (crashes on non-Hopper GPUs)
- `hybrid_vmoba_patch.py` — Hybrid dense->VMoBA with step counter
- `hybrid_nabla_patch.py` — Hybrid dense->NABLA with step counter
- `combined_selective_patch.py` — Combined timestep + layer-selective VMoBA
- `sla_patch.py` — **SLA sparse-linear attention** (block-level QK top-k selection)
- `hybrid_sla_patch.py` — **Hybrid dense->SLA** with step counter
- `combined_sla_patch.py` — **Combined timestep + layer-selective SLA** (OVERALL BEST)

## Limitations

1. **Fast-mode only metrics**: SSIM, PSNR, ImageReward, HPSv2 are only computed in full mode (25 videos). Disk space constraints (120GB total, ~9GB free after model + references) prevented full-mode evaluation for sparse methods.
2. **Single GPU**: All results on RTX 5090 only. Results may differ on other architectures.
3. **PISA untested**: Architecture incompatibility prevented any valid PISA results. PISA could potentially be the best method on H100 hardware.
4. **Composite score**: Fast mode only uses FID (weight 0.22) and LPIPS (weight 0.22), with SSIM/IR/PSNR/HPSv2 at 0. This biases the composite toward dense attention (which gets perfect scores on all metrics in full mode).
5. **SLA linear attention branch untrained**: SLA's `proj_l` is zero-initialized, meaning the linear attention branch contributes nothing. With fine-tuning, SLA might achieve even better results as the linear branch could handle the "easy" attention patterns while the sparse branch focuses on the "hard" ones.
6. **No SLA boundary=3 experiment**: We tested SLA with boundary=5 and boundary=8 but not boundary=3. Based on VMoBA trends, SLA(b=3) would likely achieve ~3.35x/~0.30 composite.

## Recommended Configuration

For Wan2.1-1.3B video generation with sparse attention:

### Best Quality (RECOMMENDED):

```yaml
method: "combined_timestep_layer_sla"
params:
  dense_steps: 10           # first 10/50 steps use dense attention everywhere
  boundary_layers: 8        # layers 0-7 and 22-29 always use dense attention
  block_size: 64            # SLA block size
  topk: 0.3                 # SLA top-k retention (30% of blocks)
```

This gives **3.06x speedup** with the best quality preservation (FID=74.7, LPIPS=0.222, composite=0.373).

### Best Speed-Quality Balance:

```yaml
method: "combined_timestep_layer_sla"
params:
  dense_steps: 10           # first 10/50 steps use dense attention everywhere
  boundary_layers: 5        # layers 0-4 and 25-29 always use dense attention
  block_size: 64            # SLA block size
  topk: 0.3                 # SLA top-k retention (30% of blocks)
```

This gives **3.20x speedup** with excellent quality (FID=76.7, LPIPS=0.245, composite=0.364).

### Maximum Speed (acceptable quality):

```yaml
method: "hybrid_dense_sla"
params:
  dense_steps: 10           # first 10/50 steps use dense attention
  block_size: 64            # SLA block size
  topk: 0.3                 # SLA top-k retention (30% of blocks)
```

This gives **3.50x speedup** with good quality (FID=109.6, LPIPS=0.272, composite=0.319).

### Legacy VMoBA (for comparison):

```yaml
method: "combined_timestep_layer_vmoba"
params:
  dense_steps: 10           # first 10/50 steps use dense attention everywhere
  boundary_layers: 8        # layers 0-7 and 22-29 always use dense attention
  tile_size_t: 3            # VMoBA temporal chunk size (frames per chunk)
  moba_topk: 3              # VMoBA top-k chunks
  simsum_threshold: 0.25    # VMoBA relevance threshold
```

This gives **3.04x speedup** with good quality (FID=83.3, LPIPS=0.251, composite=0.354) — but is strictly dominated by SLA on every metric.

## Conclusion

Through 26 experiments, we discovered that:

1. **SLA is the best sparse attention method for video diffusion transformers**, outperforming VMoBA across every configuration. SLA's block-level QK score top-k selection is more effective than VMoBA's temporal chunk selection because it directly measures query-key relevance at block granularity rather than using temporal proximity as a proxy for relevance.

2. **The optimal sparse attention strategy operates along two complementary axes**: **when** to apply sparse attention (later denoising steps) and **where** to apply it (middle transformer layers). The combined approach exploits both dimensions.

3. **The best configuration — Combined SLA(steps=10, boundary=8) — achieves a composite quality score of 0.373 at 3.06x speedup**, a 5.4% improvement over the best VMoBA configuration and a 277% improvement over pure VMoBA. This represents the Pareto-optimal tradeoff between speed and quality for this model.

4. **SLA works despite theoretical predictions suggesting otherwise.** The methods.yaml profiling indicated 0% negligible attention weights, yet SLA achieves strong results because exploiting variance in attention magnitudes (keeping the top 30% of blocks) is sufficient — weights don't need to be truly negligible to be safely skipped.

5. **The hybrid/combined strategy is the key enabler** — both SLA and VMoBA benefit enormously from temporal and spatial selectivity. The sparse method choice (SLA vs VMoBA) provides a consistent improvement, but the strategy (pure vs hybrid vs combined) provides the dominant effect on quality.
