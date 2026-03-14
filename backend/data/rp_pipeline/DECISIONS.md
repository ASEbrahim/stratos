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
- **Decision:** Option 3 -- accept the blend, steer at inference via register detection
- **Reasoning:** Simplest approach for V1. Register detection heuristic will inject style instructions at runtime. If style inconsistency shows up in holdout scores, revisit with style tagging for V2.

## D-004: Proceed to training despite strong baseline (4.2/5)
- **Date:** 2026-03-15
- **Decision:** Proceed to Phase 1 (data preparation) -- do NOT ship base model as-is
- **Reasoning:** Baseline meets the 4.0 threshold, but B-10 (multi-NPC) scored 3/5 and register shifting is only 4/5. These are core RP capabilities that training can meaningfully improve. If training doesn't improve average by +1.0, fall back to base model with prompt engineering only.
- **Risk:** Training could make things worse (overfitting, style collapse). Mitigated by the 3-run comparison design (A/B/C + baseline D).

## D-005: Synthetic data for gaps
- **Date:** 2026-03-15
- **Decision:** Generate synthetic multi-NPC and register-shifting examples for training
- **Reasoning:** Phase 0 identified multi-NPC voicing (B-10: 3/5) and register shifting (B-03: 4/5) as specific gaps. LimaRP is mostly single-NPC. PIPPA may have some multi-NPC but quality varies. Targeted synthetic generation for these two gap areas is justified.
