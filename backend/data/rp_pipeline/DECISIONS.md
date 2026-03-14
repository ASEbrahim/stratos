# Decisions Log

## D-001: Base model selection
- **Date:** 2026-03-15
- **Options:** Qwen3.5-9B (abliterate ourselves) vs huihui-ai pre-abliterated vs Qwen3-8B (proven but older)
- **Decision:** huihui-ai pre-abliterated (`huihui_ai/qwen3.5-abliterated:9b`)
- **Reasoning:** Saves abliteration step, available on Ollama (6.6GB Q4), safetensors on HuggingFace for training. Same architecture as existing qwen3.5:9b used in StratOS.
- **Risk:** Unverified abliteration quality. Mitigated by Phase 0 baseline testing.

## D-002: Synthetic data
- **Date:** 2026-03-15
- **Decision:** Zero synthetic data for V1
- **Reasoning:** LimaRP + PIPPA are human-written and uncensored. Only generate synthetic if Phase 0 identifies specific gaps that human data can't fill.

## D-003: Style conflict (LimaRP vs PIPPA)
- **Date:** 2026-03-15
- **Decision:** Option 3 — accept the blend, steer at inference via register detection
- **Reasoning:** Simplest approach for V1. Register detection heuristic will inject style instructions at runtime. If style inconsistency shows up in holdout scores, revisit with style tagging for V2.
