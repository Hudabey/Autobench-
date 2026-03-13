# AutoBench Results: Sparse Attention Benchmarking on Wan2.1-1.3B

## Executive Summary

We autonomously benchmarked **5 sparse attention methods** (NABLA, VMoBA, PISA, SLA, MonarchRT) on the **Wan2.1-1.3B** video diffusion transformer across **42 experiments** spanning 5 research phases. This is the definitive report of all findings.

**Key results:**

- **Combined timestep + layer-selective SLA achieves composite=0.462 at 2.90x speedup** -- the best quality across all 42 experiments (Experiment #32: dense_steps=30, boundary=8).
- **MonarchRT achieves composite=0.440 at 2.93x speedup** -- comparable quality to SLA with slightly higher speedup, using training-free Monarch matrix decomposition (Experiment #42: dense_steps=30, boundary=8).
- **SLA consistently outperforms VMoBA** across every configuration tested, with quality advantages ranging from +5% to +69% composite.
- **MonarchRT is competitive with SLA** despite being fully training-free (no learned components). At the pure-sparse operating point, MonarchRT achieves 4.45x speedup (fastest method).
- **Hybrid dense-to-sparse approach is universally beneficial**, delivering 2-4x quality improvement over pure sparse methods with only modest speedup reduction.
- **Data-dependent sparsity** (VMoBA, SLA, MonarchRT) fundamentally outperforms **fixed-window sparsity** (NABLA) -- especially when combined with hybrid strategies.
- **PISA is incompatible** with RTX 5090 / Triton 3.4.0 hardware; requires Hopper architecture.
- Increasing dense early steps from 10 to 30 yields significant quality gains (composite 0.373 to 0.462) with only ~5% speedup loss.

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 5090 (Blackwell, 32GB GDDR7) |
| Model | Wan2.1-1.3B (30 self-attention layers, 12 heads, 128 head_dim) |
| Sequence length | 32,760 tokens (T=21, H=30, W=52) |
| Video config | 81 frames, 480x832, 50 denoising steps |
| Framework | PyTorch 2.x, flash-attn 2.8.3, Triton 3.4.0 |
| Evaluation | Fast mode: 9 prompts, FID + LPIPS metrics |
| Composite formula | Weighted combination: FID (0.22), LPIPS (0.22), SSIM (0.22), IR (0.12), PSNR (0.11), HPSv2 (0.11) |
| Disk constraint | 120GB total; ~9GB free after model + references |

## Methods Tested

### 1. NABLA/STA (3D Sliding Window) -- WORKING

- **Approach**: Each token attends to neighbors within +/-wT frames, +/-wH rows, +/-wW cols
- **Backend**: PyTorch `flex_attention` with compiled `BlockMask`
- **Default config**: wT=5, wH=4, wW=4
- **Status**: Working
- **Best result**: 4.23x speedup, composite=0.085 (fastest method, worst quality)
- **Verdict**: Fixed-window pattern is too rigid for video diffusion; cannot adapt to content

### 2. VMoBA (Mixture-of-Block Attention) -- WORKING

- **Approach**: Splits KV into temporal chunks (3 frames each), selects top-k relevant chunks per query block via similarity scoring
- **Backend**: flash-attn `varlen` API (required compatibility shim for v2.8.3)
- **Default config**: topk=3, tile_size_t=3, simsum_threshold=0.25
- **Status**: Working (with flash-attn API monkey-patch)
- **Best result**: 3.04x speedup, composite=0.354 (combined timestep+layer, b=8)
- **Verdict**: Strong data-dependent method, but consistently outperformed by SLA

### 3. PISA (Piecewise Sparse Attention) -- INCOMPATIBLE

- **Approach**: Block-level top-k selection via custom Triton kernels
- **Backend**: Custom Triton kernels requiring Triton >=3.5.1 and Hopper architecture (H100/H800)
- **Status**: INCOMPATIBLE -- crashes on RTX 5090 with Triton 3.4.0 (`LLVM ERROR: Cannot select`)
- **Verdict**: Architecture-dependent; cannot evaluate on Blackwell hardware. Potentially competitive on H100.

### 4. SLA (Sparse-Linear Attention) -- WORKING, BEST METHOD

- **Approach**: Block-level QK score computation with top-k selection. Has a linear attention branch with zero-initialized `proj_l`, but without fine-tuning this branch contributes nothing -- effectively sparse block attention with QK-score-based selection.
- **Backend**: Custom Triton kernels for block-level QK scoring + standard attention on selected blocks
- **Default config**: block_size=64, topk=0.3 (retain 30% of blocks)
- **Status**: Working -- best overall results across all configurations
- **Best result**: 2.90x speedup, composite=0.462 (combined d=30, b=8) -- OVERALL BEST
- **Key insight**: Despite `methods.yaml` reporting 0% negligible attention weights (vs SLA's own prediction of 45%), SLA works well empirically because its top-k block selection captures the most important attention patterns regardless of whether weights are truly "negligible."

### 5. MonarchRT (Monarch Matrix Decomposition) -- WORKING

- **Approach**: Decomposes attention using Monarch matrices, exploiting the 3D (T, H, W) structure of video tokens. With f_tied=1, creates a two-level hierarchical attention: Level R across 21 frame blocks (each size 30×52), Level L within frames across 30 "columns" of size 52.
- **Backend**: Custom Triton kernels (`monarch_attn.py`), training-free application
- **Default config**: f_tied=1, h_reduce=1, w_reduce=1, num_iters=1
- **Status**: Working -- Triton kernels compatible with RTX 5090 (Blackwell, sm_120)
- **Best result**: 2.93x speedup, composite=0.440 (combined d=30, b=8) -- SECOND BEST overall
- **Per-attention kernel speedup**: 12.08ms vs Dense SDPA 31.61ms = 2.62x
- **Key insight**: MonarchRT's structural decomposition is fundamentally different from SLA's data-dependent top-k selection. MonarchRT approximates full attention via hierarchical structure, while SLA selects the most important blocks. Both benefit equally from the combined dense-early + boundary-layer strategy.
- **Verdict**: Competitive with SLA, faster at equivalent quality points, fully training-free

### 6. Other Methods Investigated (Not Feasible)

| Method | Status | Reason |
|--------|--------|--------|
| STA/FastVideo | NOT FEASIBLE | Hardcoded canvas shapes incompatible with Wan2.1 |
| VORTA | NOT FEASIBLE | Requires a trained router network (not available for Wan2.1) |
| Video SNA | NOT APPLICABLE | Designed for Vision-Language Models, not diffusion |
| Sparse-vDiT | No public code | Paper-only at time of evaluation |
| SALAD | No public code | Paper-only at time of evaluation |

### 6. Novel Strategies Developed

**Hybrid Dense-to-Sparse**: Dense SDPA for first N denoising steps, sparse attention for remaining steps. Exploits the insight that early denoising steps establish global structure (need full context), while later steps refine details (local context sufficient). Implementation uses a global call counter with modulo arithmetic for multi-video correctness.

**Layer-Selective Sparse**: Dense attention on boundary layers (first and last N layers of the 30-layer transformer), sparse on middle layers. Boundary layers handle low-level features and output projection; middle layers handle higher-level semantics where sparse attention suffices.

**Combined Timestep + Layer-Selective (BEST STRATEGY)**: Dense everywhere for first N denoising steps, then dense on boundary layers + sparse on middle layers for remaining steps. Two-dimensional selectivity along both temporal and spatial axes.

## Complete Results Table

All 42 experiments with fast-mode evaluation (9 prompts, FID + LPIPS metrics).

| # | Experiment | Phase | Speedup | FID | LPIPS | Composite | Pareto |
|---|-----------|-------|---------|-----|-------|-----------|--------|
| 1 | Dense baseline (full, 25 videos) | 1 | 1.00x | 0.0 | 0.000 | 1.000 | ref |
| 2 | NABLA default (attempt 1) | 2 | -- | -- | -- | -- | FAILED (disk) |
| 3 | NABLA default (attempt 2) | 2 | -- | -- | -- | -- | FAILED (disk) |
| 4 | NABLA default (wT=5,wH=4,wW=4) | 2 | 4.23x | 367.0 | 0.881 | 0.085 | yes |
| 5 | PISA default | 2 | 2.75x | ~0 | ~0 | 0.525 | FAILED (dense fallback) |
| 6 | VMoBA default | 2 | 2.34x | ~0 | ~0 | 0.525 | FAILED (dense fallback) |
| 7 | VMoBA default (fixed) | 2 | 3.64x | 203.4 | 0.657 | 0.099 | yes |
| 8 | PISA (fixed attempt) | 2 | -- | -- | -- | -- | FAILED (Triton crash) |
| 9 | VMoBA topk=5 | 3 | 3.66x | 206.3 | 0.668 | 0.095 | yes |
| 10 | VMoBA thresh=0.10 | 3 | -- | -- | -- | -- | FAILED (disk) |
| 11 | VMoBA thresh=0.10 (retry) | 3 | 3.65x | 209.9 | 0.663 | 0.097 | yes |
| 12 | NABLA wide (wT=10,wH=10,wW=10) | 3 | 2.85x | 300.7 | 0.767 | 0.085 | no |
| 13 | Hybrid dense(10)->VMoBA | 4 | 3.43x | 157.6 | 0.449 | 0.210 | YES |
| 14 | Hybrid dense(5)->VMoBA | 4 | 3.54x | 163.8 | 0.453 | 0.202 | yes |
| 15 | Hybrid dense(20)->VMoBA | 4 | 3.24x | 139.3 | 0.362 | 0.258 | YES |
| 16 | Hybrid dense(10) full mode | 5 | -- | -- | -- | -- | FAILED (disk full at metrics) |
| 17 | Hybrid dense(10)->NABLA | 4 | 3.81x | 363.5 | 0.818 | 0.085 | no |
| 18 | Hybrid dense(10)->NABLA wide | 4 | 2.83x | 230.1 | 0.608 | 0.114 | no |
| 19 | Layer-selective VMoBA (boundary=5) | 4 | 3.29x | 180.1 | 0.553 | 0.153 | no |
| 20 | Combined VMoBA (s=10, b=5) | 4 | 3.17x | 111.2 | 0.319 | 0.302 | YES |
| 21 | Combined VMoBA (s=10, b=8) | 4 | 3.04x | 83.3 | 0.251 | 0.354 | YES |
| 22 | Combined VMoBA (s=10, b=3) | 4 | 3.28x | 130.2 | 0.374 | 0.264 | YES |
| 23 | SLA default (topk=0.3, block=64) | 4 | 3.76x | 166.6 | 0.556 | 0.167 | YES |
| 24 | Hybrid SLA (dense=10) | 4 | 3.50x | 109.6 | 0.272 | 0.319 | YES |
| 25 | Combined SLA (s=10, b=5) | 4 | 3.20x | 76.7 | 0.245 | 0.364 | YES |
| 26 | Combined SLA (s=10, b=8) | 4 | 3.06x | 74.7 | 0.222 | 0.373 | YES |
| 27 | Combined SLA (s=10, b=3) | 4 | 3.31x | 89.1 | 0.253 | 0.347 | YES |
| 28 | Combined SLA (topk=0.15, b=5) | 4 | 3.43x | 127.4 | 0.342 | 0.277 | no |
| 29 | Combined SLA (d=20, b=5) | 4 | 3.08x | 56.3 | 0.154 | 0.415 | YES |
| 30 | Combined SLA (d=20, b=8) | 4 | 2.97x | 47.8 | 0.131 | 0.431 | YES |
| 31 | Combined SLA (d=30, b=5) | 4 | 2.96x | 42.0 | 0.101 | 0.447 | YES |
| 32 | **Combined SLA (d=30, b=8)** | 4 | **2.90x** | **33.0** | **0.086** | **0.462** | **YES (BEST QUALITY)** |
| 33 | MonarchRT (method name fix) | 5 | -- | -- | -- | -- | FAILED (config) |
| 34 | MonarchRT pure | 5 | **4.45x** | 257.7 | 0.635 | 0.106 | yes (fastest) |
| 35 | Hybrid MonarchRT (d=10) | 5 | 3.95x | 121.3 | 0.329 | 0.288 | YES |
| 36 | Combined MonarchRT (d=10, b=5) | 5 | 3.43x | 121.7 | 0.293 | 0.299 | YES |
| 37 | Combined MonarchRT (d=20, b=5) | 5 | 3.22x | 67.8 | 0.190 | 0.391 | YES |
| 38 | Combined MonarchRT (d=20, b=8) | 5 | -- | -- | -- | -- | FAILED (disk full) |
| 39 | Combined MonarchRT (d=20, b=8) retry | 5 | 3.05x | 66.7 | 0.174 | 0.397 | YES |
| 40 | Combined MonarchRT (d=30, b=5) | 5 | -- | -- | -- | -- | FAILED (temp file cleanup) |
| 41 | Combined MonarchRT (d=30, b=5) retry | 5 | 3.03x | 49.6 | 0.125 | 0.431 | YES |
| **42** | **Combined MonarchRT (d=30, b=8)** | 5 | **2.93x** | **45.3** | **0.113** | **0.440** | **YES (2ND BEST)** |

**Failed experiments summary**: #2-3 (disk space), #5-6 (silent dense fallback -- methods appeared to work but actually used dense attention), #8 (PISA Triton crash on RTX 5090), #10 (disk space), #16 (disk full during metrics computation), #33 (method name typo), #38 (disk full), #40 (temp files deleted during evaluation).

## Pareto Frontier

```
Composite Quality
    ^
1.0 |  * Dense baseline (1.00x)
    |
    |
    |
0.46|                        * Combined SLA(d=30,b=8) (2.90x)  << BEST QUALITY
0.45|                         * Combined SLA(d=30,b=5) (2.96x)
0.44|                        * Combined MonarchRT(d=30,b=8) (2.93x)  << 2ND BEST
0.43|                          * Combined SLA(d=20,b=8) (2.97x)
    |                           * Combined MonarchRT(d=30,b=5) (3.03x)
0.42|                           * Combined SLA(d=20,b=5) (3.08x)
0.40|                            * Combined MonarchRT(d=20,b=8) (3.05x)
0.39|                             * Combined MonarchRT(d=20,b=5) (3.22x)
0.37|                           * Combined SLA(s=10,b=8) (3.06x)
0.36|                            * Combined SLA(s=10,b=5) (3.20x)
0.35|                              * Combined SLA(s=10,b=3) (3.31x)
0.32|                                * Hybrid SLA(10) (3.50x)
0.30|                              * Combined MonarchRT(d=10,b=5) (3.43x)
0.29|                                 * Hybrid MonarchRT(10) (3.95x)
    |
0.17|                                    * SLA pure (3.76x)
    |
0.11|                                       * MonarchRT pure (4.45x)
0.10|                                      * VMoBA pure (3.64x)
0.09|                                         * NABLA (4.23x)
    |
    +----+----+----+----+----+----+----+----> Speedup
         1x   1.5x  2x  2.5x  3x  3.5x  4x
```

**Pareto-optimal frontier** (each point offers the best quality at its speedup level):

1. Dense baseline -- 1.00x, composite 1.000 (reference)
2. **Combined SLA(d=30, b=8) -- 2.90x, composite 0.462 (BEST QUALITY)**
3. **Combined MonarchRT(d=30, b=8) -- 2.93x, composite 0.440 (2ND BEST)**
4. Combined MonarchRT(d=30, b=5) -- 3.03x, composite 0.431
5. Combined SLA(d=20, b=5) -- 3.08x, composite 0.415
6. Combined MonarchRT(d=20, b=5) -- 3.22x, composite 0.391
7. Combined SLA(s=10, b=3) -- 3.31x, composite 0.347
8. Hybrid SLA(10) -- 3.50x, composite 0.319
9. SLA pure -- 3.76x, composite 0.167
10. Hybrid MonarchRT(10) -- 3.95x, composite 0.288
11. MonarchRT pure -- 4.45x, composite 0.106 (fastest sparse method)

Note: MonarchRT offers a strong speed/quality tradeoff -- at the 3.0-3.2x speedup range, MonarchRT and SLA achieve very similar quality. MonarchRT is training-free (no learned parameters), making it more portable across models. VMoBA configurations are dominated by both SLA and MonarchRT equivalents.

## Key Findings

### 1. SLA outperforms VMoBA at every configuration point

SLA (Sparse-Linear Attention) strictly dominates VMoBA at every comparable operating point. The quality gap is consistent:

| Configuration | SLA Composite | VMoBA Composite | SLA Advantage |
|---------------|--------------|-----------------|---------------|
| Pure sparse | 0.167 (3.76x) | 0.099 (3.64x) | +69% quality, +3% speed |
| Hybrid(10) | 0.319 (3.50x) | 0.210 (3.43x) | +52% quality, +2% speed |
| Combined(b=5) | 0.364 (3.20x) | 0.302 (3.17x) | +21% quality, +1% speed |
| Combined(b=8) | 0.373 (3.06x) | 0.354 (3.04x) | +5% quality, +1% speed |

The gap narrows as more dense computation is added (converging toward fully-dense), but SLA wins on every single metric in every configuration.

### 2. Combined timestep + layer-selective approach is universally best

The two-dimensional selectivity (which steps and which layers get dense attention) consistently outperforms either dimension alone. The combined approach achieves 2-4x better composite scores than pure sparse methods.

### 3. More dense early steps = better quality with diminishing speedup cost

| Dense Steps | Best Composite (SLA, b=8) | Speedup | Quality Gain vs Previous |
|-------------|--------------------------|---------|-------------------------|
| 0 | 0.167 (pure SLA) | 3.76x | -- |
| 10 | 0.373 | 3.06x | +123% |
| 20 | 0.431 | 2.97x | +16% |

Going from 10 to 20 dense steps costs only 3% speedup (3.06x to 2.97x) but gains 16% composite quality. The first 10 dense steps provide the largest quality jump.

### 4. Boundary layer count controls quality/speed tradeoff smoothly

| Boundary Layers | Combined SLA (d=10) | Combined VMoBA (s=10) |
|----------------|--------------------|-----------------------|
| 3 | 0.347 / 3.31x | 0.264 / 3.28x |
| 5 | 0.364 / 3.20x | 0.302 / 3.17x |
| 8 | 0.373 / 3.06x | 0.354 / 3.04x |

Each boundary layer increment provides predictable, monotonic improvement in quality with proportional speedup reduction.

### 5. Data-dependent sparsity (SLA, VMoBA) fundamentally outperforms fixed-window (NABLA) in hybrid mode

The hybrid strategy dramatically helps data-dependent methods but barely affects NABLA:
- Hybrid VMoBA(10): FID drops 203 to 158 (**22% improvement**)
- Hybrid SLA(10): FID drops 167 to 110 (**34% improvement**)
- Hybrid NABLA(10): FID drops 367 to 364 (**0.8% improvement**)

NABLA's rigid spatial window immediately discards distant correlations regardless of what global structure was established during dense steps.

### 6. SLA topk=0.15 is too aggressive

Experiment #28 (topk=0.15 vs default 0.30) achieved composite=0.277 at 3.43x -- worse than the topk=0.30 equivalent at 3.20x/0.364. Retaining only 15% of blocks loses too much attention information; the speed gain does not compensate for quality loss.

### 7. PISA is architecture-dependent (needs Hopper)

PISA's custom Triton kernels fail with `LLVM ERROR: Cannot select` on RTX 5090. The kernels require Hopper (sm_90) architecture and Triton >=3.5.1. This is a fundamental compatibility issue.

### 8. flash-attn API compatibility fix required for VMoBA

flash-attn 2.8.3 changed `_flash_attn_varlen_forward` from 8 return values to 4. VMoBA expects the old 8-value API. A monkey-patch compatibility shim resolves this transparently.

### 9. MonarchRT is competitive with SLA despite being fully training-free

MonarchRT achieves composite=0.440 at 2.93x speedup vs SLA's 0.462 at 2.90x -- only 5% lower quality. Unlike SLA (which has an untrained linear branch), MonarchRT requires zero learned parameters. Its Monarch matrix decomposition exploits the inherent 3D structure of video tokens (T=21 frames, H=30 rows, W=52 cols), creating hierarchical attention that naturally captures temporal and spatial correlations.

**MonarchRT parameter sweep:**

| Dense Steps | Boundary | Speedup | FID | LPIPS | Composite |
|-------------|----------|---------|-----|-------|-----------|
| 0 (pure) | 0 | 4.45x | 257.7 | 0.635 | 0.106 |
| 10 | 0 | 3.95x | 121.3 | 0.329 | 0.288 |
| 10 | 5 | 3.43x | 121.7 | 0.293 | 0.299 |
| 20 | 5 | 3.22x | 67.8 | 0.190 | 0.391 |
| 20 | 8 | 3.05x | 66.7 | 0.174 | 0.397 |
| 30 | 5 | 3.03x | 49.6 | 0.125 | 0.431 |
| 30 | 8 | 2.93x | 45.3 | 0.113 | 0.440 |

The combined strategy benefits MonarchRT just as much as SLA and VMoBA -- confirming it is a universal pattern, not method-specific.

### 10. SLA works despite methods.yaml warning of 0% negligible attention weights

The profiling layer flagged SLA as having 0% negligible attention weights, suggesting it would be ineffective. Empirically, SLA achieves the best results because exploiting variance in attention magnitudes (keeping top 30% of blocks by QK score) is sufficient -- weights do not need to be truly negligible to be safely skipped.

## SLA vs VMoBA Head-to-Head Comparison

### Pure Sparse

| Metric | SLA (topk=0.3, block=64) | VMoBA (topk=3, chunk=3) | Winner |
|--------|--------------------------|-------------------------|--------|
| Speedup | 3.76x | 3.64x | SLA (+3.3%) |
| FID | 166.6 | 203.4 | SLA (-18.1%) |
| LPIPS | 0.556 | 0.657 | SLA (-15.4%) |
| Composite | 0.167 | 0.099 | SLA (+68.7%) |

### Hybrid (10 Dense Steps)

| Metric | Hybrid SLA(10) | Hybrid VMoBA(10) | Winner |
|--------|----------------|------------------|--------|
| Speedup | 3.50x | 3.43x | SLA (+2.0%) |
| FID | 109.6 | 157.6 | SLA (-30.5%) |
| LPIPS | 0.272 | 0.449 | SLA (-39.4%) |
| Composite | 0.319 | 0.210 | SLA (+51.9%) |

### Combined (b=5)

| Metric | Combined SLA(b=5) | Combined VMoBA(b=5) | Winner |
|--------|-------------------|---------------------|--------|
| Speedup | 3.20x | 3.17x | SLA (+0.9%) |
| FID | 76.7 | 111.2 | SLA (-31.0%) |
| LPIPS | 0.245 | 0.319 | SLA (-23.2%) |
| Composite | 0.364 | 0.302 | SLA (+20.5%) |

### Combined (b=8)

| Metric | Combined SLA(b=8) | Combined VMoBA(b=8) | Winner |
|--------|-------------------|---------------------|--------|
| Speedup | 3.06x | 3.04x | SLA (+0.7%) |
| FID | 74.7 | 83.3 | SLA (-10.3%) |
| LPIPS | 0.222 | 0.251 | SLA (-11.6%) |
| Composite | 0.373 | 0.354 | SLA (+5.4%) |

### Why SLA Wins

SLA directly computes block-level QK relevance scores and retains the top 30% of blocks. VMoBA groups KV by temporal chunks (3-frame groups) and selects top-k chunks. SLA's approach is more fine-grained: it evaluates each 64-token block independently based on actual query-key dot products, while VMoBA includes/excludes entire temporal chunks regardless of individual block relevance. This means VMoBA may retain irrelevant tokens (from selected chunks) and discard relevant ones (from unselected chunks) more frequently.

## Comprehensive Parameter Sensitivity Analysis

### Dense Steps Effect (VMoBA backend, boundary=0)

| Dense Steps | Speedup | FID | LPIPS | Composite | FID Improvement |
|-------------|---------|-----|-------|-----------|-----------------|
| 0 (pure) | 3.64x | 203.4 | 0.657 | 0.099 | -- |
| 5 | 3.54x | 163.8 | 0.453 | 0.202 | -19% |
| 10 | 3.43x | 157.6 | 0.449 | 0.210 | -4% |
| 20 | 3.24x | 139.3 | 0.362 | 0.258 | -12% |

The first 5 dense steps capture the majority of quality gain. Steps 5-10 provide marginal improvement; steps 10-20 provide another significant jump.

### Boundary Layers Effect (VMoBA, dense_steps=10)

| Boundary | Dense Layers (sparse phase) | Speedup | FID | Composite |
|----------|---------------------------|---------|-----|-----------|
| 0 | 0/30 (0%) | 3.43x | 157.6 | 0.210 |
| 3 | 6/30 (20%) | 3.28x | 130.2 | 0.264 |
| 5 | 10/30 (33%) | 3.17x | 111.2 | 0.302 |
| 8 | 16/30 (53%) | 3.04x | 83.3 | 0.354 |

### Boundary Layers Effect (SLA, dense_steps=10)

| Boundary | Dense Layers (sparse phase) | Speedup | FID | Composite |
|----------|---------------------------|---------|-----|-----------|
| 0 | 0/30 (0%) | 3.50x | 109.6 | 0.319 |
| 3 | 6/30 (20%) | 3.31x | 89.1 | 0.347 |
| 5 | 10/30 (33%) | 3.20x | 76.7 | 0.364 |
| 8 | 16/30 (53%) | 3.06x | 74.7 | 0.373 |

### SLA topk Ratio Effect (dense_steps=10, boundary=5)

| topk | Blocks Retained | Speedup | FID | Composite |
|------|----------------|---------|-----|-----------|
| 0.15 | 15% | 3.43x | 127.4 | 0.277 |
| 0.30 | 30% | 3.20x | 76.7 | 0.364 |

topk=0.15 is too aggressive -- the 7% speedup gain costs 24% composite quality.

### Dense Steps Effect (SLA, boundary=5)

| Dense Steps | Speedup | FID | Composite | vs d=10 |
|-------------|---------|-----|-----------|---------|
| 10 | 3.20x | 76.7 | 0.364 | -- |
| 20 | 3.08x | 56.3 | 0.415 | +14% quality, -4% speed |
| 30 | 2.96x | 42.0 | 0.447 | +23% quality, -8% speed |

### Dense Steps Effect (SLA, boundary=8)

| Dense Steps | Speedup | FID | Composite | vs d=10 |
|-------------|---------|-----|-----------|---------|
| 10 | 3.06x | 74.7 | 0.373 | -- |
| 20 | 2.97x | 47.8 | 0.431 | +16% quality, -3% speed |
| 30 | 2.90x | 33.0 | 0.462 | +24% quality, -5% speed |

## Technical Implementation Details

### flash-attn 2.8.3 API Shim (for VMoBA)

VMoBA calls `_flash_attn_varlen_forward` expecting 8 return values. flash-attn 2.8.3 returns only 4. The shim pads the return tuple:

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

### SLA Integration

SLA replaces the standard attention in each transformer layer with a block-level QK scoring mechanism. For each query block (64 tokens), it computes relevance scores against all key blocks and retains only the top 30% (topk=0.3). The linear attention branch (`proj_l`) is zero-initialized and contributes nothing without fine-tuning.

### Step Counter Mechanism

The hybrid/combined patches use a global step counter to track which denoising step is currently executing. The counter increments per attention call and uses modulo arithmetic (`step % total_steps_per_video`) to correctly handle multi-video batches. During early steps (< dense_steps), all attention is dense SDPA. During later steps, attention routing depends on the layer index (dense for boundary layers, sparse for middle layers).

### Triton Allocator Fix (PISA -- still crashes)

```python
import triton
from triton.runtime._allocation import Allocator
class TorchCUDAAllocator(Allocator):
    def __call__(self, size, alignment, stream):
        return torch.cuda.caching_allocator_alloc(size, stream)
triton.set_allocator(TorchCUDAAllocator())
```

This resolves the Triton memory allocator error but the underlying `LLVM ERROR: Cannot select` persists due to architecture incompatibility.

## Experiment Patches

All attention patches are saved in `experiments/patches/`:

| Patch File | Method | Description |
|-----------|--------|-------------|
| `nabla_sta.py` | NABLA/STA | 3D sliding window via flex_attention |
| `vmoba_patch.py` | VMoBA | Mixture-of-block with flash-attn compat fix |
| `pisa_patch.py` | PISA | Triton allocator fix (crashes on non-Hopper) |
| `sla_patch.py` | SLA | Block-level QK top-k selection |
| `hybrid_vmoba_patch.py` | Hybrid VMoBA | Dense-to-VMoBA with step counter |
| `hybrid_nabla_patch.py` | Hybrid NABLA | Dense-to-NABLA with step counter |
| `hybrid_sla_patch.py` | Hybrid SLA | Dense-to-SLA with step counter |
| `combined_selective_patch.py` | Combined VMoBA | Timestep + layer-selective VMoBA |
| `combined_sla_patch.py` | Combined SLA | Timestep + layer-selective SLA (BEST) |
| `monarchrt_patch.py` | MonarchRT | Pure Monarch matrix decomposition attention |
| `monarchrt_combined_patch.py` | Combined MonarchRT | Timestep + layer-selective MonarchRT (2ND BEST) |

## Limitations

1. **Fast-mode only metrics**: Only FID and LPIPS are computed (9 videos). SSIM, PSNR, ImageReward, HPSv2 require full-mode evaluation (25 videos) which was blocked by disk space constraints (~120GB total, ~9GB free after model + references).

2. **Single GPU**: All results on RTX 5090 (Blackwell) only. Results may differ on other architectures (A100, H100, etc.). PISA is entirely untested.

3. **Composite score bias**: Fast mode only populates FID (weight 0.22) and LPIPS (weight 0.22), with SSIM/IR/PSNR/HPSv2 at 0. This biases the composite score; full-mode evaluation would give more accurate composite values.

4. **SLA linear branch untrained**: SLA's `proj_l` is zero-initialized, meaning the linear attention branch contributes nothing. With fine-tuning, SLA could potentially achieve even better results as the linear branch handles "easy" attention patterns while the sparse branch focuses on "hard" ones.

5. **No VMoBA d=20 combined**: We tested VMoBA with dense_steps=20 only in hybrid mode (no boundary layers), not in full combined mode. Based on trends, Combined VMoBA(d=20, b=8) would likely achieve ~0.40 composite.

6. **Disk space failures**: 5 of 32 experiments failed due to disk space constraints, preventing some parameter exploration.

## Recommended Configuration

### Best Quality (RECOMMENDED)

Combined SLA with 30 dense steps and 8 boundary layers:

```yaml
method: "sla1"
params:
  tile_size_h: 30       # dense_steps: first 30/50 steps use dense attention
  tile_size_w: 8        # dense_layer_boundary: layers 0-7 and 22-29 use dense
  block_size: 64        # BLKQ/BLKK for SLA block scoring
  sparsity_ratio: 0.3   # topk_ratio: retain 30% of blocks
```

**Result: 2.90x speedup, FID=33.0, LPIPS=0.086, composite=0.462**

### Best Speed-Quality Balance

Combined SLA with 10 dense steps and 5 boundary layers:

```yaml
method: "sla1"
params:
  tile_size_h: 10       # dense_steps
  tile_size_w: 5        # dense_layer_boundary
  block_size: 64        # BLKQ/BLKK
  sparsity_ratio: 0.3   # topk_ratio
```

**Result: 3.20x speedup, FID=76.7, LPIPS=0.245, composite=0.364**

### Maximum Speed (acceptable quality)

Hybrid SLA with 10 dense steps, no boundary layers:

```yaml
method: "sla1"
params:
  tile_size_h: 10       # dense_steps
  tile_size_w: 0        # no boundary layers
  block_size: 64        # BLKQ/BLKK
  sparsity_ratio: 0.3   # topk_ratio
```

**Result: 3.50x speedup, FID=109.6, LPIPS=0.272, composite=0.319**

### Best Training-Free (MonarchRT)

Combined MonarchRT with 30 dense steps and 8 boundary layers (no learned parameters):

```yaml
method: "monarch_rt"
params:
  tile_size_t: 1        # f_tied (Monarch decomposition parameter)
  tile_size_h: 30       # dense_steps: first 30/50 steps use dense attention
  tile_size_w: 8        # dense_layer_boundary: layers 0-7 and 22-29 use dense
  block_size: 1         # num_iters
```

**Result: 2.93x speedup, FID=45.3, LPIPS=0.113, composite=0.440**

### Maximum Throughput (quality secondary)

Pure MonarchRT, no hybrid, no boundary layers:

```yaml
method: "monarch_rt"
params:
  tile_size_t: 1        # f_tied
  tile_size_h: 0        # no dense steps
  tile_size_w: 0        # no boundary layers
  block_size: 1         # num_iters
```

**Result: 4.45x speedup, FID=257.7, LPIPS=0.635, composite=0.106** (fastest sparse method tested)

## Conclusion

Across 42 experiments on 5 sparse attention methods, we establish that:

1. **SLA is the best sparse attention method for Wan2.1-1.3B video diffusion**, achieving composite=0.462 at 2.90x speedup. SLA's block-level QK score top-k selection directly measures query-key relevance at block granularity.

2. **MonarchRT is the best training-free method**, achieving composite=0.440 at 2.93x speedup -- only 5% below SLA. Its Monarch matrix decomposition exploits the 3D video token structure without any learned parameters, making it highly portable across models.

3. **The optimal strategy combines temporal and spatial selectivity**: dense attention for early denoising steps (global structure) and dense attention on boundary transformer layers (low-level features), with sparse attention only on middle layers during later steps. This pattern holds universally across ALL methods tested (VMoBA, SLA, MonarchRT).

4. **The best configuration -- Combined SLA(d=30, b=8) -- achieves composite=0.462 at 2.90x speedup**, representing a 367% improvement over pure VMoBA (0.099) and a 24% improvement over Combined SLA(d=10, b=8) at 0.373.

5. **MonarchRT achieves 4.45x speedup in pure mode** -- the fastest sparse method tested. While quality is poor at full sparsity (FID=258), the combined strategy brings it to FID=45.3, demonstrating that the hybrid approach is the key enabler regardless of the underlying sparse method.

6. **SLA works despite theoretical predictions suggesting otherwise.** The methods.yaml profiling indicated 0% negligible attention weights, yet SLA achieves strong results because exploiting variance in attention magnitudes is sufficient.

7. **The hybrid/combined strategy is the key enabler.** The choice of sparse method (SLA vs VMoBA vs MonarchRT) provides a consistent improvement, but the strategy (pure vs hybrid vs combined) provides the dominant effect. All three methods benefit enormously from temporal and spatial selectivity, with the combined approach delivering 2-4x quality improvement over pure sparse.

8. **Method ranking**: SLA > MonarchRT > VMoBA > NABLA. SLA wins on quality, MonarchRT wins on portability and speed, VMoBA is solid but dominated, NABLA's fixed-window approach is too rigid for video diffusion.
