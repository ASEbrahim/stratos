#!/bin/bash

echo ""
echo "============================================================"
echo "   STRAT_OS - Strategic Intelligence Operating System"
echo "============================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Start Ollama if not running
echo "Checking Ollama..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &> /dev/null &
    sleep 3
fi

# Start STRAT_OS
echo "Starting STRAT_OS..."
echo ""
echo "Press Ctrl+C to stop."
echo ""

cd backend
python3 main.py --serve
