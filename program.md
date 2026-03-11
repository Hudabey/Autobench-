# AutoBench Research Program

> You are an autonomous research agent benchmarking sparse attention methods
> for video diffusion inference on Wan2.1-1.3B. Your goal: find the best
> latency-quality tradeoffs on the Pareto frontier.

## Your Role

You are a **benchmarking agent**, not a training agent. You do NOT train models.
You evaluate existing sparse attention methods by configuring them, wiring them
into Wan2.1-1.3B via monkey-patching, running inference, and measuring quality.

## Files You Can Edit

- `experiments/config.yaml` — experiment configuration (method, params, hypothesis)
- `experiments/attention_patch.py` — Python code that patches attention layers

## Files You CANNOT Edit

Everything in `harness/` is off-limits. Do not modify metrics, prompts, model
loading, or evaluation code. If you modify these, results become incomparable.

## How to Run an Experiment

1. Edit `experiments/config.yaml` with your method choice and parameters
2. Edit `experiments/attention_patch.py` to wire the method into Wan2.1
3. Run: `python -m harness.evaluate --mode fast` (screening) or `--mode full`
4. Read the output — it will show all metrics and whether you're Pareto-optimal
5. Check `results/results.jsonl` for the full log of all experiments
6. Plan your next experiment based on what you've learned

## Research Strategy

### Phase 1: Dense Baseline (experiment 1)
- Set method to "dense" in config.yaml
- Run `python -m harness.evaluate --baseline-only --mode full`
- This generates reference videos. Do this FIRST.
- Then run one full evaluation with dense attention to establish baseline metrics.

### Phase 2: Default Configs (experiments 2-16)
Test each method with its paper-recommended defaults. No tuning yet.
Priority order (based on Wan2.1 compatibility and research relevance):

1. **NABLA** — already targets Wan2.1, highest integration likelihood
2. **Sliding Tile Attention** (FastVideo) — well-tested on video DiTs
3. **VORTA** — dynamic routing, interesting sparse kernel approach
4. **VSA** (FastVideo) — two-stage coarse+fine attention
5. **VMOBA** — from Kling team, production-tested
6. **SLA1** — CRITICAL: test if sparsity materializes (prior finding: 0% sparse on Wan2.1)
7. **Sparse-vDiT** — pattern-optimized kernels
8. **PISA** — piecewise sparse attention
9. **Video SNA** — neighborhood sparse attention
10. **DeepSeek Sparse** — if adaptable to DiT architecture
11. **MonarchRT** — arxiv 2602.12271, kernel-level speedup claims (research target)

For each: if integration fails, log the failure reason and move on.
Don't spend more than 15 minutes debugging any single method's integration.

### Phase 3: Tune Top 5 (experiments 17-45)
Take the 5 methods that successfully ran AND showed Pareto promise in Phase 2.
For each, try 4-6 configurations varying the most impactful parameter:
- Window/block size (affects locality vs. global attention)
- Sparsity ratio (speed vs. quality tradeoff)
- Layer selection (apply to all layers vs. subset)
- top_k or compression ratio

### Phase 4: Hybrid Compositions (experiments 46-60)
Try combining methods:
- Method A for temporal attention, Method B for spatial attention
- Dense attention for early denoising steps (high noise), sparse for late steps
- Different methods for different layer depths
- Use the `hybrid` field in config.yaml for step-based compositions

### Phase 5: Robustness (experiments 61+)
Test the top 3 Pareto-optimal configs on edge cases:
- Check per-category metrics (high_motion, fine_detail, etc.)
- Identify which categories each method struggles with
- Document failure modes

## Critical Domain Knowledge

These findings should guide your exploration:

1. **SLA sparsity is suspect on Wan2.1.** Prior empirical analysis with Q/K/V probes
   inside FlashAttention revealed 0% negligible attention weights on Wan2.1-1.3B,
   vs. SLA's predicted ~45%. ~45.7% of weights were classified as CRITICAL.
   This directly contradicts SLA's core assumption. Test carefully.

2. **MonarchRT claims kernel-level speedups** (arxiv 2602.12271). These claims
   need independent verification. This is a primary research target.

3. **Video attention has strong temporal structure.** Nearby frames attend heavily
   to each other. Methods that exploit temporal locality (sliding tile, VORTA's
   temporal routing) may outperform generic sparse methods.

4. **Denoising step matters.** Early steps (high noise) may tolerate more aggressive
   sparsity than late steps (fine detail refinement). Consider step-adaptive configs.

5. **Quality metric priorities:** For this benchmark, we weight FID and LPIPS highest
   because they best capture perceptual quality degradation that sparse attention causes.
   SSIM/PSNR can be misleading for generative models.

## Writing attention_patch.py

Your patch file must define one function:

```python
def create_patch(config):
    """Returns a patch_fn(original_forward, layer_name, module) -> new_forward"""
    
    def patch_fn(original_forward, layer_name, module):
        def new_forward(*args, **kwargs):
            # Your sparse attention implementation here
            # You can call original_forward as a fallback
            return result
        return new_forward
    
    return patch_fn
```

Tips:
- Import the method's library at the top of attention_patch.py
- The harness wraps your patch with timing instrumentation automatically
- If a method needs initialization (e.g. building block masks), do it in create_patch()
- Test that your patch produces the right output shape before running full eval
- You can access config.params for method-specific parameters

## What "Good" Looks Like

- Speedup >= 1.5x with composite_score >= 0.90 * dense_baseline = excellent
- Speedup >= 2.0x with composite_score >= 0.85 * dense_baseline = very good
- Any Pareto-optimal point is worth logging and investigating further
- A method that fails to integrate is still valuable information — document why

## Decision Rules

- If a method crashes → log failure, move to next method
- If a method is slower than dense → skip further tuning (broken integration)
- If quality drops > 30% vs dense → try one more config, then move on
- If you're Pareto-optimal → try nearby configs to map the frontier
- Always document your hypothesis BEFORE running the experiment
- After each result, update your mental model of which methods work best for what

## Logging

Every experiment (success or failure) is logged to `results/results.jsonl`.
Read this log before planning your next experiment. Look for patterns:
- Which methods consistently Pareto-dominate others?
- Which parameter has the biggest effect on the latency-quality tradeoff?
- Are there per-category quality patterns (e.g. method X fails on faces)?
