#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# STRAT_OS Launcher
# Double-click this file or run: bash ~/Downloads/StratOS/StratOS1/backend/stratos.sh
# ═══════════════════════════════════════════════════════════════

BACKEND="$HOME/Downloads/StratOS/StratOS1/backend"
cd "$BACKEND" || { echo "ERROR: Backend directory not found at $BACKEND"; read -p "Press Enter to exit..."; exit 1; }

echo "════════════════════════════════════════════════"
echo "  STRAT_OS — Strategic Intelligence OS"
echo "════════════════════════════════════════════════"
echo ""

# ── 1. Check Ollama is running ──
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[!] Ollama is not running. Starting it..."
    ollama serve &
    sleep 3
fi

# ── 2. Check scoring model is registered in Ollama ──
SCORER_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$BACKEND/config.yaml'))['scoring']['model'])" 2>/dev/null || echo "stratos-scorer-v1")
echo "[→] Scoring model from config: $SCORER_MODEL"

if ! ollama list 2>/dev/null | grep -q "$SCORER_MODEL"; then
    echo "[!] $SCORER_MODEL not found in Ollama. Looking for GGUF..."
    
    # Find the latest GGUF in data/models/
    GGUF_PATH=$(find "$BACKEND/data/models" -name "*.gguf" -type f 2>/dev/null | sort -V | tail -1)
    if [ -z "$GGUF_PATH" ]; then
        echo "[!] No GGUF found. Scoring will fall back to base model."
    else
        echo "[→] Found: $GGUF_PATH"
        cat > /tmp/Modelfile << ENDOFFILE
FROM ${GGUF_PATH}

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
PARAMETER num_predict 128
PARAMETER repeat_penalty 1.3
PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>
PARAMETER stop <|im_start|>
PARAMETER stop <think>
PARAMETER stop </think>

SYSTEM """You are a relevance scorer for a personalized intelligence dashboard.
Score each article 0.0-10.0. Never score exactly 5.0.
9.0-10.0: Directly actionable (job match, breakthrough, money-saving deal)
7.0-8.9: Highly relevant (useful skills, regional growth, key trend)
5.0-6.9: Somewhat relevant
0.0-4.9: Noise (generic ads, clickbait, wrong field)
Respond with ONLY valid JSON: {"score": X.X, "reason": "brief explanation"}
Do not use thinking tags. Output only the JSON."""
ENDOFFILE
        ollama create "$SCORER_MODEL" -f /tmp/Modelfile
        echo "[✓] $SCORER_MODEL registered"
    fi
fi

echo ""
echo "[✓] Ollama ready"

# ── 3. Ensure inference model for agent/suggest/generate ──
if ! ollama list 2>/dev/null | grep -q "qwen3:30b-a3b"; then
    echo "[→] Pulling qwen3:30b-a3b for agent & inference features..."
    ollama pull qwen3:30b-a3b
fi

echo "[→] Starting STRAT_OS server + background scanning..."
echo "[→] Dashboard will open at http://localhost:8080"
echo "[→] Press Ctrl+C to stop"
echo ""

# ── 3. Launch StratOS with server + background refresh ──
python3 main.py --serve --background
