# StratOS Complete Patch — Thumbs Feedback + Think Fix + Training Fix

## Drop-In Instructions (Ubuntu)

```bash
cd ~/Downloads/StratOS/StratOS1

# Backup everything first
for f in backend/train_lora.py backend/main.py backend/database.py backend/export_training.py \
         backend/processors/scorer_adaptive.py backend/processors/briefing.py backend/processors/profile_generator.py \
         backend/routes/agent.py backend/routes/generate.py \
         frontend/feed.js frontend/app.js; do
    cp "$f" "$f.bak"
done
```

Then replace each file with the downloaded version into the same path.

---

## What's Fixed

### 1. Qwen3 Think Block Bug (7 files)
Qwen3 outputs `<think>...reasoning...</think>` before the answer.
With tight token budgets (300-400), the think block eats all tokens → empty response.
This is what caused "Suggest returned empty" in the autopilot log.

**Fix:** Every Ollama call now has `"think": False` + strip fallback.

### 2. Thumbs Up/Down Feedback (6 files)  
👍/👎 on news cards → stored in DB → injected into scorer prompt → exported as training data.

### 3. Training Crash Fix (1 file)
Delete `hf_device_map` after PEFT wrapping + force `.to("cuda:0")`.
