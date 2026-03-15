#!/bin/bash
# Start ComfyUI in API mode for StratOS image generation (CHROMA model)
# Run ONLY when no training is running and Ollama is stopped

echo "Stopping Ollama to free GPU..."
ollama stop 2>/dev/null
sleep 2

# Ensure ComfyUI_FluxMod custom node is installed (required for CHROMA)
COMFYUI_DIR=~/Downloads/StratOS/StratOS1/tools/ComfyUI
if [ ! -d "$COMFYUI_DIR/custom_nodes/ComfyUI_FluxMod" ]; then
    echo "Installing ComfyUI_FluxMod custom node..."
    cd "$COMFYUI_DIR/custom_nodes"
    git clone https://github.com/lodestone-rock/ComfyUI_FluxMod.git
    cd "$COMFYUI_DIR"
fi

echo "Starting ComfyUI on http://127.0.0.1:8188"
cd "$COMFYUI_DIR"
python3 main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --preview-method auto
