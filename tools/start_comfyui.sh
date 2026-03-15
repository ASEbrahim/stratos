#!/bin/bash
# Start ComfyUI in API mode for StratOS image generation
# Run ONLY when no training is running and Ollama is stopped

echo "Stopping Ollama to free GPU..."
ollama stop 2>/dev/null
sleep 2

echo "Starting ComfyUI on http://127.0.0.1:8188"
cd ~/Downloads/StratOS/StratOS1/tools/ComfyUI
python3 main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --preview-method auto
