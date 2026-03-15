#!/bin/bash
# Download CHROMA model + dependencies for StratOS image generation
# Run once. Downloads ~18GB total.

set -e
COMFYUI_DIR=~/Downloads/StratOS/StratOS1/tools/ComfyUI

echo "=== StratOS CHROMA Model Setup ==="
echo ""

# 1. CHROMA model (Q8_0 GGUF, ~9GB)
echo "[1/5] Downloading CHROMA Q8_0 GGUF..."
if [ -f "$COMFYUI_DIR/models/unet/chroma-unlocked-v35-Q8_0.gguf" ]; then
    echo "  Already exists, skipping."
else
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='lodestones/Chroma',
    filename='chroma-unlocked-v35-Q8_0.gguf',
    local_dir='$COMFYUI_DIR/models/unet',
)
print('  Done.')
"
fi

# 2. T5-XXL Q8 text encoder (~5GB)
echo "[2/5] Downloading T5-XXL Q8 encoder..."
if [ -f "$COMFYUI_DIR/models/clip/t5-v1_1-xxl-encoder-Q8_0.gguf" ]; then
    echo "  Already exists, skipping."
else
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='city96/t5-v1_1-xxl-encoder-gguf',
    filename='t5-v1_1-xxl-encoder-Q8_0.gguf',
    local_dir='$COMFYUI_DIR/models/clip',
)
print('  Done.')
"
fi

# 3. CLIP-L text encoder (~250MB)
echo "[3/5] Downloading CLIP-L..."
if [ -f "$COMFYUI_DIR/models/clip/clip_l.safetensors" ]; then
    echo "  Already exists, skipping."
else
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='comfyanonymous/flux_text_encoders',
    filename='clip_l.safetensors',
    local_dir='$COMFYUI_DIR/models/clip',
)
print('  Done.')
"
fi

# 4. FLUX VAE (ae.safetensors, ~320MB) — shared with CHROMA
echo "[4/5] Downloading FLUX VAE..."
if [ -f "$COMFYUI_DIR/models/vae/ae.safetensors" ]; then
    echo "  Already exists, skipping."
else
    echo "  NOTE: ae.safetensors requires HuggingFace authentication (gated repo)."
    echo "  Run: huggingface-cli login"
    echo "  Then re-run this script."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='black-forest-labs/FLUX.1-schnell',
    filename='ae.safetensors',
    local_dir='$COMFYUI_DIR/models/vae',
)
print('  Done.')
" || echo "  FAILED — download manually from https://huggingface.co/black-forest-labs/FLUX.1-schnell"
fi

# 5. ComfyUI_FluxMod custom node (required for CHROMA)
echo "[5/5] Installing ComfyUI_FluxMod custom node..."
if [ -d "$COMFYUI_DIR/custom_nodes/ComfyUI_FluxMod" ]; then
    echo "  Already installed, pulling latest..."
    cd "$COMFYUI_DIR/custom_nodes/ComfyUI_FluxMod" && git pull
else
    cd "$COMFYUI_DIR/custom_nodes"
    git clone https://github.com/lodestone-rock/ComfyUI_FluxMod.git
fi

echo ""
echo "=== Setup Complete ==="
echo "Model files:"
ls -lh "$COMFYUI_DIR/models/unet/"*.gguf 2>/dev/null
ls -lh "$COMFYUI_DIR/models/clip/"*.{gguf,safetensors} 2>/dev/null
ls -lh "$COMFYUI_DIR/models/vae/"*.safetensors 2>/dev/null
echo ""
echo "Start ComfyUI: bash tools/start_comfyui.sh"
