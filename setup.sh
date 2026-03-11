#!/usr/bin/env bash
# AutoBench Setup Script
# Run this once on your GPU instance to get everything ready.
set -e

echo "═══════════════════════════════════════════════════"
echo "  AutoBench Setup — Sparse Attention Benchmarking"
echo "  for Wan2.1-1.3B"
echo "═══════════════════════════════════════════════════"

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[setup] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Install dependencies
echo "[setup] Installing Python dependencies..."
uv sync

# Optional: install full metric suite
echo "[setup] Installing optional metric dependencies..."
uv pip install hpsv2 matplotlib pandas jupyter 2>/dev/null || \
    echo "[setup] WARNING: Some optional deps failed. HPSv2 metrics may not be available."

# 3. Install flash-attn (Bug 12: was missing)
echo "[setup] Installing flash-attn (may take a few minutes to compile)..."
uv pip install flash-attn --no-build-isolation --no-cache-dir 2>/dev/null || {
    echo "[setup] WARNING: flash-attn failed to install. Wan2.1 will use SDPA fallback."
    echo "[setup]   For accurate latency measurements, install manually:"
    echo "[setup]   pip install flash-attn --no-build-isolation"
}

# 4. Verify GPU
echo "[setup] Checking GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')" || \
    echo "[setup] WARNING: No GPU detected. This will be very slow."

# 5. Verify diffusers has Wan2.1 support
echo "[setup] Checking Wan2.1 pipeline support..."
python -c "from diffusers import WanPipeline; print('WanPipeline available')" || \
    echo "[setup] WARNING: WanPipeline not found. You may need diffusers >= 0.33.0"

# 6. Clone sparse attention method repos (Bug 11: was missing entirely)
echo ""
echo "[setup] Cloning sparse attention method repositories..."
mkdir -p methods

clone_method() {
    local name=$1
    local url=$2
    if [ -d "methods/$name" ]; then
        echo "  [$name] Already cloned, skipping"
    else
        echo "  [$name] Cloning $url..."
        git clone --depth 1 "$url" "methods/$name" 2>/dev/null && \
            echo "  [$name] OK" || \
            echo "  [$name] FAILED (check URL or auth)"
    fi
}

clone_method nabla        "https://github.com/gen-ai-team/Wan2.1-NABLA.git"
clone_method vorta        "https://github.com/wenhao728/VORTA.git"
clone_method fastvideo    "https://github.com/hao-ai-lab/FastVideo.git"
clone_method sla          "https://github.com/thu-ml/SLA.git"
clone_method sparse_vdit  "https://github.com/Peyton-Chen/Sparse-vDiT.git"
clone_method vmoba        "https://github.com/KlingAIResearch/VMoBA.git"
clone_method pisa         "https://github.com/xie-lab-ml/piecewise-sparse-attention.git"
clone_method video_sna    "https://github.com/Espere-1119-Song/VideoNSA.git"

echo ""
echo "[setup] Cloned methods:"
ls -d methods/*/ 2>/dev/null | sed 's|methods/||;s|/||' | while read m; do echo "  ✓ $m"; done

# 7. Create output directories
mkdir -p results/plots reference_videos

# 8. Summary
echo ""
echo "═══════════════════════════════════════════════════"
echo "  SETUP COMPLETE"
echo ""
echo "  Next step: Generate dense baseline"
echo ""
echo "  This will:"
echo "    - Download Wan2.1-1.3B (~2.6 GB)"
echo "    - Generate 25 reference videos (~20-40 min)"
echo "    - Compute baseline metrics"
echo ""
echo "  Run:"
echo "    uv run python -m harness.evaluate --baseline-only --mode full"
echo ""
echo "  Then point your agent at program.md:"
echo "    'Read program.md and start the research loop.'"
echo "═══════════════════════════════════════════════════"
