# RP Pipeline State

**Last updated:** 2026-03-15
**Current phase:** Phase 0 COMPLETE -- Phase 1 ready to begin
**Current task:** Phase 0 baseline testing complete (avg 4.2/5)
**Next step:** Begin Phase 1 -- download LimaRP and PIPPA datasets, run filtering pipeline
**Blockers:** Disk space tight (38GB free - 6.6GB model = ~31GB). Safetensors download (~18GB) will be tight. May need to clean old model artifacts.

## Completed
- [x] Branch created: `rp-model-v1`
- [x] Directory structure set up
- [x] .gitignore updated for RP pipeline artifacts
- [x] Living documents initialized
- [x] Phase 0: Model pulled (huihui_ai/qwen3.5-abliterated:9b, 6.6GB)
- [x] Phase 0: Baseline model registered (stratos-rp-baseline)
- [x] Phase 0: Baseline tests B-01 through B-10 complete
- [x] Phase 0: Scored and documented (avg 4.2/5, lowest 3/5 on multi-NPC)
- [x] Decision D-004: Proceed to training (multi-NPC and register-shifting gaps)
- [x] Decision D-005: Generate synthetic data for identified gaps
- [ ] Phase 1: Download LimaRP dataset
- [ ] Phase 1: Download PIPPA dataset
- [ ] Phase 1: Filter PIPPA (quality filter)
- [ ] Phase 1: Convert LimaRP format
- [ ] Phase 1: Create holdout set (200 conversations, SACRED)
- [ ] Phase 1: Generate synthetic gap-filling data
- [ ] Phase 1: Validate training data
- [ ] Phase 2: DoRA Fine-Tune (3 runs: A, B, C)
- [ ] Phase 3: Evaluation
- [ ] Phase 4: Integration
- [ ] Phase 5: Conversation Memory

## Key Numbers
- Baseline average score: 4.2/5
- Lowest baseline score: 3/5 (B-10: Multi-NPC)
- Primary gaps: multi-NPC voicing, register shifting
- Disk free: ~31GB (after model download)
- Ollama models loaded: scorer v2.2, qwen3.5:9b, stratos-rp-baseline

## How to Resume
```bash
git checkout rp-model-v1
cat backend/data/rp_pipeline/SESSION_STATE.md
# Begin Phase 1:
cd backend
# 1. Download datasets
huggingface-cli download lemonilia/LimaRP --local-dir data/rp_pipeline/raw_data/limarp/
# 2. Download PIPPA
python3 -c "from datasets import load_dataset; ds = load_dataset('PygmalionAI/PIPPA', 'pippa_deduped'); ds['train'].to_json('data/rp_pipeline/raw_data/pippa_deduped.jsonl')"
# 3. Run filtering scripts (create these first)
```
