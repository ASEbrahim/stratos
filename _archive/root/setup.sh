#!/bin/bash

echo ""
echo "============================================================"
echo "   STRAT_OS - Strategic Intelligence Operating System"
echo "   First-Time Setup"
echo "============================================================"
echo ""

# Check Python
echo "[1/4] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed."
    echo "Please install Python 3.11+ using your package manager."
    exit 1
fi
echo "[OK] Python found: $(python3 --version)"
echo ""

# Check Ollama
echo "[2/4] Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "[WARNING] Ollama is not installed."
    echo ""
    echo "Ollama is required for AI-powered scoring."
    echo "Install it from: https://ollama.ai/download"
    echo ""
    echo "Or on Linux: curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    read -p "Continue without Ollama? (y/n): " continue
    if [[ $continue != "y" ]]; then
        exit 1
    fi
    echo "[SKIP] Continuing without Ollama (will use fallback scoring)"
else
    echo "[OK] Ollama found."
    
    # Pull model
    echo ""
    echo "[3/4] Pulling AI model (llama3.2)..."
    echo "This may take 5-10 minutes on first run..."
    ollama pull llama3.2
    echo "[OK] Model ready."
fi
echo ""

# Install Python packages
echo "[4/4] Installing Python dependencies..."
cd "$(dirname "$0")"
pip3 install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies."
    exit 1
fi
echo "[OK] Dependencies installed."
echo ""

# Create output directory
mkdir -p backend/output

echo "============================================================"
echo "   Setup Complete!"
echo "============================================================"
echo ""
echo "To start STRAT_OS, run: ./start.sh"
echo ""
