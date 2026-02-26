#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# STRAT_OS — Overnight Training Mode
# ═══════════════════════════════════════════════════════════════
# Run this before bed. It will:
#   1. Start the dashboard (accessible at http://localhost:8080)
#   2. Loop: Scan → Qwen scores → Opus re-scores → save corrections
#   3. Every 3 cycles, retrain Qwen with Opus's corrections
#   4. Stop when budget runs out or you press Ctrl+C
#
# Usage:
#   bash stratos_overnight.sh           # Default $5 budget
#   bash stratos_overnight.sh 10.00     # $10 budget
# ═══════════════════════════════════════════════════════════════

BACKEND="$HOME/Downloads/StratOS/StratOS1/backend"
BUDGET="${1:-8.16}"

cd "$BACKEND" || { echo "ERROR: Backend not found at $BACKEND"; exit 1; }

# ── 0. Prevent system sleep during overnight training ──
echo "[→] Disabling system sleep..."
SLEEP_INHIBITED=false
if command -v systemd-inhibit &> /dev/null; then
    # systemd-inhibit will be used to wrap the main process later
    SLEEP_INHIBITED=true
    echo "[✓] Will use systemd-inhibit to prevent sleep"
elif command -v caffeine &> /dev/null; then
    caffeine &
    CAFFEINE_PID=$!
    SLEEP_INHIBITED=true
    echo "[✓] Caffeine activated (PID: $CAFFEINE_PID)"
else
    # Fallback: disable via gsettings
    if command -v gsettings &> /dev/null; then
        OLD_IDLE_DELAY=$(gsettings get org.gnome.desktop.session idle-delay 2>/dev/null)
        OLD_SLEEP_TYPE=$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null)
        gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null
        gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null
        SLEEP_INHIBITED=true
        echo "[✓] GNOME sleep disabled via gsettings"
    else
        echo "[!] Could not disable sleep — install 'caffeine' or keep PC active manually"
        echo "    sudo apt install caffeine   OR   gsettings set org.gnome.desktop.session idle-delay 0"
    fi
fi

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  STRAT_OS — Overnight Autonomous Training"
echo "══════════════════════════════════════════════════════════"
echo "  Budget:    \$$BUDGET"
echo "  Dashboard: http://localhost:8080"
echo "  Log file:  $BACKEND/data/autopilot.log"
echo "  Stop:      Ctrl+C"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── 1. Load .env ──
if [ -f "$BACKEND/.env" ]; then
    export $(grep -v '^#' "$BACKEND/.env" | xargs)
    echo "[✓] API key loaded from .env"
else
    echo "[!] No .env file found — distillation will fail"
    echo "    Create one: echo 'ANTHROPIC_API_KEY=sk-ant-...' > $BACKEND/.env"
    exit 1
fi

# ── 2. Check Ollama ──
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[→] Starting Ollama..."
    ollama serve > /dev/null 2>&1 &
    sleep 4
fi

# Verify scorer model is loaded (reads from config.yaml)
SCORER_MODEL=$(python3 -c "import yaml; print(yaml.safe_load(open('$BACKEND/config.yaml'))['scoring']['model'])" 2>/dev/null || echo "stratos-scorer-v1")
if ! ollama list 2>/dev/null | grep -q "$SCORER_MODEL"; then
    echo "[→] Registering $SCORER_MODEL..."
    GGUF=$(find "$BACKEND/data/models" -name "*.gguf" -type f 2>/dev/null | sort -V | tail -1)
    if [ -n "$GGUF" ]; then
        cat > /tmp/Modelfile << ENDOFFILE
FROM ${GGUF}

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
    else
        echo "[!] No GGUF found — autopilot will use qwen3:8b"
    fi
fi

# Ensure inference model (qwen3:30b-a3b) is available for agent/suggest/generate
if ! ollama list 2>/dev/null | grep -q "qwen3:30b-a3b"; then
    echo "[→] Pulling qwen3:30b-a3b for inference (agent, suggest, generate)..."
    echo "    This may take a while on first run..."
    ollama pull qwen3:30b-a3b
fi

echo "[✓] Ollama ready (scorer + inference)"

# ── 3. Start dashboard server in background ──
echo "[→] Starting dashboard server..."
python3 main.py --serve &
SERVER_PID=$!
sleep 3

# Check server started
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[!] Server failed to start"
    exit 1
fi
echo "[✓] Dashboard running at http://localhost:8080 (PID: $SERVER_PID)"

# ── 4. Run autopilot (scan → distill → train loop) ──
echo ""
echo "[→] Starting autopilot loop..."
echo "    Each cycle: Scan (~2min) → Opus distill (~3min) → Wait 60s"
echo "    Training triggers every 3 cycles"
echo "    ~12 cycles per \$5 budget ≈ 1-2 hours of active learning"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "[→] Shutting down..."
    kill $SERVER_PID 2>/dev/null

    # Restore sleep settings
    if [ -n "$CAFFEINE_PID" ]; then
        kill $CAFFEINE_PID 2>/dev/null
        echo "[✓] Caffeine stopped"
    fi
    if [ -n "$OLD_IDLE_DELAY" ]; then
        gsettings set org.gnome.desktop.session idle-delay $OLD_IDLE_DELAY 2>/dev/null
        gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type $OLD_SLEEP_TYPE 2>/dev/null
        echo "[✓] Sleep settings restored"
    fi

    echo "[✓] Server stopped"
    echo ""
    echo "Check results:"
    echo "  Dashboard:  http://localhost:8080"
    echo "  Log:        $BACKEND/data/autopilot.log"
    echo "  Report:     $BACKEND/data/distillation_report.json"
    echo "  Training:   $BACKEND/data/training_data.jsonl"
    echo "  State:      $BACKEND/data/autopilot_state.json"
    exit 0
}
trap cleanup INT TERM

# Run autopilot — this blocks until budget exhausted or Ctrl+C
python3 autopilot.py \
    --budget "$BUDGET" \
    --train-every 3 \
    --interval 60

# When autopilot finishes (budget exhausted), keep server running
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Autopilot complete! Budget exhausted."
echo "  Dashboard still running — press Ctrl+C to stop."
echo "══════════════════════════════════════════════════════════"
wait $SERVER_PID
