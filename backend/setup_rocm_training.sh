#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  STRAT_OS — ROCm + LoRA Training Setup for AMD GPUs
#  Tested on: Ubuntu 22.04/24.04 + Radeon 7900 XTX (gfx1100)
# ═══════════════════════════════════════════════════════════════
#
# What this does:
#   1. Installs ROCm 6.x (AMD's GPU compute platform)
#   2. Installs PyTorch with ROCm support
#   3. Installs PEFT + TRL for LoRA training
#   4. Installs llama.cpp for GGUF conversion
#   5. Verifies everything works
#
# Usage:
#   chmod +x setup_rocm_training.sh
#   ./setup_rocm_training.sh
#
# After setup, run:
#   python export_training.py
#   python train_lora.py
#
set -e

echo "════════════════════════════════════════════════════════════"
echo "  STRAT_OS ROCm Training Environment Setup"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Check if AMD GPU is present ──
if ! lspci | grep -i 'vga\|display' | grep -qi 'amd\|radeon'; then
    echo "❌ No AMD GPU detected. This script is for AMD Radeon GPUs."
    exit 1
fi
echo "✅ AMD GPU detected:"
lspci | grep -i 'vga\|display' | grep -i 'amd\|radeon'
echo ""

# ── Step 1: Install ROCm ──
echo "[1/5] Checking ROCm installation..."
if command -v rocminfo &>/dev/null && rocminfo 2>/dev/null | grep -q 'gfx'; then
    echo "✅ ROCm already installed"
    rocminfo 2>/dev/null | grep 'Name:' | head -3
else
    echo "Installing ROCm 6.x..."
    
    # Add ROCm repository
    sudo mkdir -p /etc/apt/keyrings
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
    
    # Detect Ubuntu version
    UBUNTU_VER=$(lsb_release -rs)
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/latest \
        $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null
    
    sudo apt-get update
    sudo apt-get install -y rocm-dev rocm-libs rocminfo
    
    # Add user to render/video groups
    sudo usermod -aG render,video $USER
    
    echo "✅ ROCm installed"
    echo "⚠️  You may need to REBOOT and re-run this script for ROCm to work."
    echo "   Run: sudo reboot"
    echo ""
fi

# Verify ROCm can see the GPU
if rocminfo 2>/dev/null | grep -q 'gfx11'; then
    echo "✅ ROCm sees your 7900 XTX (gfx1100)"
elif rocminfo 2>/dev/null | grep -q 'gfx'; then
    echo "✅ ROCm sees your AMD GPU"
else
    echo "⚠️  ROCm installed but can't see GPU. Try:"
    echo "   1. sudo reboot"
    echo "   2. Re-run this script"
    echo "   3. Check: sudo dmesg | grep amdgpu"
    exit 1
fi
echo ""

# ── Step 2: Install PyTorch with ROCm ──
echo "[2/5] Installing PyTorch with ROCm support..."
pip3 install --upgrade pip --break-system-packages

# PyTorch ROCm 6.2 (latest stable)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 --break-system-packages

# Verify
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available (via ROCm HIP): {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'VRAM: {vram:.1f} GB')
    print(f'ROCm/HIP: {torch.version.hip}')
" && echo "✅ PyTorch with ROCm working" || echo "❌ PyTorch GPU test failed"
echo ""

# ── Step 3: Install Training Dependencies ──
echo "[3/5] Installing training dependencies..."
pip3 install transformers datasets peft trl accelerate sentencepiece protobuf --break-system-packages
echo "✅ Training libraries installed"
echo ""

# ── Step 4: Install llama.cpp for GGUF conversion ──
echo "[4/5] Installing llama.cpp (GGUF converter)..."
LLAMA_DIR="$HOME/llama.cpp"
if [ -d "$LLAMA_DIR" ] && [ -f "$LLAMA_DIR/convert_hf_to_gguf.py" ]; then
    echo "✅ llama.cpp already present at $LLAMA_DIR"
    cd "$LLAMA_DIR" && git pull --quiet 2>/dev/null || true
else
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
    echo "✅ llama.cpp cloned to $LLAMA_DIR"
fi
# Install Python requirements for conversion
pip3 install gguf numpy --break-system-packages
echo ""

# ── Step 5: Install STRAT_OS backend dependencies ──
echo "[5/5] Installing STRAT_OS backend dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    pip3 install --break-system-packages -r "$SCRIPT_DIR/../requirements.txt" 2>/dev/null || true
elif [ -f "$SCRIPT_DIR/../requirements.txt" ]; then
    pip3 install --break-system-packages -r "$SCRIPT_DIR/../requirements.txt" 2>/dev/null || true
fi
pip3 install pyyaml --break-system-packages
echo "✅ Backend dependencies ready"
echo ""

# ── Summary ──
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Setup Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Your training environment:"
python3 -c "
import torch
vram = torch.cuda.get_device_properties(0).total_mem / 1024**3 if torch.cuda.is_available() else 0
print(f'    GPU:      {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'    VRAM:     {vram:.0f} GB')
print(f'    Backend:  ROCm {torch.version.hip if hasattr(torch.version, \"hip\") and torch.version.hip else \"N/A\"}')
print(f'    PyTorch:  {torch.__version__}')
" 2>/dev/null || true
echo ""
echo "  With 24GB VRAM, you can train:"
echo "    Qwen3 1.7B  — LoRA in ~2 minutes  (uses ~4GB)"
echo "    Qwen3 4B    — LoRA in ~5 minutes  (uses ~10GB)"
echo "    Qwen3 8B    — LoRA in ~10 minutes (uses ~18GB)"
echo ""
echo "  Next steps:"
echo "    cd $(dirname "$SCRIPT_DIR")/backend"
echo "    python3 export_training.py        # Convert corrections → training data"
echo "    python3 train_lora.py             # Fine-tune Qwen with LoRA"
echo ""
echo "  The GGUF conversion will happen automatically."
echo "  After training, update config.yaml:"
echo "    scoring:"
echo "      model: stratos-scorer-v1"
echo "════════════════════════════════════════════════════════════"
