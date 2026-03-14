# RP Pipeline State

**Last updated:** 2026-03-15
**Current phase:** Phase 0 — Pull & Baseline
**Current task:** Pull abliterated model, run baseline tests
**Next step:** Pull `huihui_ai/qwen3.5-abliterated:9b` via Ollama, then run B-01 through B-10
**Blockers:** Disk space tight (38GB free) — defer safetensors download until Phase 2

## Completed
- [x] Branch created: `rp-model-v1`
- [x] Directory structure set up
- [x] .gitignore updated for RP pipeline artifacts
- [x] Living documents initialized
- [ ] Phase 0: Baseline testing (in progress)
- [ ] Phase 1: Data preparation
- [ ] Phase 2: DoRA Fine-Tune
- [ ] Phase 3: Evaluation
- [ ] Phase 4: Integration
- [ ] Phase 5: Conversation Memory

## Key Numbers
- Baseline average score: TBD
- Disk free: 38GB (tight — watch model downloads)
- Existing Ollama models: scorer v2.2 (9.5GB), qwen3.5:9b (6.6GB)
