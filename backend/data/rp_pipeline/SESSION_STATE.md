# RP Pipeline State

**Last updated:** 2026-03-15
**Current phase:** Phase 0 — Pull & Baseline
**Current task:** Model download in progress (huihui_ai/qwen3.5-abliterated:9b)
**Next step:** Once download completes: `ollama create stratos-rp-baseline -f data/rp_pipeline/Modelfile.baseline` then `python3 data/rp_pipeline/scripts/run_baseline.py`
**Blockers:** Model download ETA ~2 hours at current speed

## Completed
- [x] Branch created: `rp-model-v1`
- [x] Directory structure set up
- [x] .gitignore updated for RP pipeline artifacts
- [x] Living documents initialized (SESSION_STATE, TRAINING_LOG, EVAL_LOG, DATA_LOG, DECISIONS)
- [x] Baseline test suite created (10 tests, automated runner)
- [x] Modelfile.baseline created
- [ ] Phase 0: Pull abliterated model (downloading...)
- [ ] Phase 0: Register baseline model in Ollama
- [ ] Phase 0: Run baseline tests B-01 through B-10
- [ ] Phase 0: Score and write baseline report
- [ ] Phase 1: Data preparation
- [ ] Phase 2: DoRA Fine-Tune
- [ ] Phase 3: Evaluation
- [ ] Phase 4: Integration
- [ ] Phase 5: Conversation Memory

## Key Numbers
- Baseline average score: TBD
- Disk free: ~38GB (model download will use ~6.6GB)
- Existing Ollama models: scorer v2.2 (9.5GB), qwen3.5:9b (6.6GB)
- Abliterated model: downloading (ETA ~2h)

## How to Resume
```bash
git checkout rp-model-v1
cat data/rp_pipeline/SESSION_STATE.md
# Check if model finished downloading:
ollama list | grep abliterated
# If downloaded, register and run tests:
cd backend
ollama create stratos-rp-baseline -f data/rp_pipeline/Modelfile.baseline
python3 data/rp_pipeline/scripts/run_baseline.py
```
