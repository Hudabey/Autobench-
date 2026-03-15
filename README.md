# AutoBench

Autonomous sparse attention benchmarking for Wan2.1-1.3B video generation.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), but adapted for **inference benchmarking** instead of training optimization. An AI agent systematically evaluates and configures multiple sparse attention methods against a fixed evaluation harness.

## Quick Start

```bash
# 1. Setup
chmod +x setup.sh && ./setup.sh

# 2. Generate dense baseline (run ONCE)
uv run python -m harness.evaluate --baseline-only --mode fast

# 3. Start the agent
# Point your AI agent at program.md:
#   "Read program.md and start the benchmarking loop."
```

## How It Works

```
Karpathy's loop:  edit train.py → train 5min → check val_bpb → keep/discard
AutoBench loop:   edit config  → generate videos → measure 7 metrics → update Pareto frontier
```

## Project Structure

```
autobench/
├── program.md              ← Human edits: research strategy for the agent
├── methods.yaml            ← Human edits: method registry
│
├── harness/                ← 🔒 FIXED — agent cannot touch
│   ├── config_schema.py    │   Config validation + bounds
│   ├── model.py            │   Wan2.1-1.3B loading + attention patching
│   ├── metrics.py          │   FID, SSIM, PSNR, LPIPS, IR, HPSv2
│   ├── evaluate.py         │   Main orchestration loop
│   └── prompts.json        │   Fixed evaluation prompts (25)
│
├── experiments/            ← ✏️ Agent edits these
│   ├── config.yaml         │   Current experiment config
│   └── attention_patch.py  │   Attention monkey-patch code
│
├── results/                ← 📊 Append-only log
│   ├── results.jsonl       │   All experiment results
│   └── pareto.json         │   Current Pareto frontier
│
├── reference_videos/       ← Generated once from dense baseline
└── methods/                ← Cloned sparse attention repos
```

## Metrics

| Metric | Direction | Category |
|--------|-----------|----------|
| FID | ↓ lower is better | Semantic fidelity |
| ImageReward | ↑ higher is better | Human preference |
| HPSv2 | ↑ higher is better | Human preference |
| SSIM | ↑ higher is better | Structural similarity |
| PSNR | ↑ higher is better | Signal fidelity |
| LPIPS | ↓ lower is better | Perceptual distance |
| Latency | ↓ lower is better | Speed |

## Key Design Decisions

- **Training-free**: We only do inference. No model weights are modified.
- **Multi-objective**: Pareto frontier over (latency, composite quality) instead of single scalar.
- **Fixed harness**: Prompts, metrics, seeds, and evaluation code are immutable.
- **Tiered evaluation**: Fast screening (9 prompts, 2 metrics) → Full eval (25 prompts, 7 metrics).
