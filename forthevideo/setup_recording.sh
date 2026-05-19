#!/bin/bash
# Quick setup script for BirdCLEF+ agent video recording
# Run this before recording to ensure everything is ready

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   BirdCLEF+ 2026 AGENT - VIDEO RECORDING SETUP                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python environment
echo "[1/6] Checking Python environment..."
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Virtual environment not activated. Running:"
    source ~/.venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "✅ Already in venv: $VIRTUAL_ENV"
fi
echo ""

# Check GPU
echo "[2/6] Checking GPU..."
GPU_OUTPUT=$(python -c "
import torch
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'✅ GPU: {device}')
    print(f'✅ VRAM: {vram:.1f}GB')
    print(f'✅ CUDA Version: {torch.version.cuda}')
else:
    print('❌ No GPU detected')
    exit(1)
" 2>&1)
echo "$GPU_OUTPUT"
echo ""

# Check Ollama
echo "[3/6] Checking Ollama..."
if python -c "from llm_client import LLMClient; c = LLMClient(); print('✅ Ollama is running')" 2>&1; then
    echo "✅ LLM client is ready"
else
    echo "⚠️  Warning: Ollama might not be running. Start it with: ollama serve"
fi
echo ""

# Generate results report
echo "[4/6] Generating results report..."
python results_reporter.py --output experiments/RESULTS_SUMMARY.md 2>&1 | grep -E "^✅|^Error" || echo "✅ Report generated"
echo ""

# Clear unnecessary files
echo "[5/6] Cleaning up temporary files..."
find experiments -name "*.pyc" -delete 2>/dev/null || true
find experiments -name "__pycache__" -type d -delete 2>/dev/null || true
echo "✅ Cleaned up"
echo ""

# Display directory structure
echo "[6/6] Ready to record!"
echo ""
echo "📊 Experiment Statistics:"
python -c "
import json
from pathlib import Path

state_file = Path('experiments/agent_state.json')
if state_file.exists():
    state = json.loads(state_file.read_text())
    print(f\"  • Total iterations: {state['iteration']}\")
    print(f\"  • Best AUC: {state['best_auc']:.4f}\")
    print(f\"  • Best iteration: {state['best_iteration']}\")
    successful = len([h for h in state['history'] if h['auc'] > 0.1])
    print(f\"  • Successful runs: {successful}/{len(state['history'])}\")
" || echo "  (No experiments yet)"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    READY FOR RECORDING!                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "💡 NEXT STEPS:"
echo ""
echo "1. RECOMMENDED: Pre-record the demo run"
echo "   python agent_demo.py --mode demo 2>&1 | tee demo_run.log"
echo ""
echo "2. Then record screen with narration overlay"
echo "   (Play back pre-recorded run while narrating)"
echo ""
echo "3. OR record live run (takes 10-15 minutes for 2 iterations)"
echo "   python agent_demo.py --mode demo"
echo ""
echo "4. Generate results summary to show at end"
echo "   python results_reporter.py -o RESULTS.md"
echo ""
echo "💻 Terminal Settings for Recording:"
echo "   • Font size: 16-18pt"
echo "   • Theme: Light background, dark text"
echo "   • No transparency"
echo "   • Full screen or at least 1280x720"
echo ""
echo "🎥 OBS Studio Settings:"
echo "   • Bitrate: 6000-8000 kbps"
echo "   • Resolution: 1920x1080 (or native)"
echo "   • FPS: 30 (sufficient for terminal)"
echo "   • Audio: Mic + System Audio (capture terminal)"
echo ""
