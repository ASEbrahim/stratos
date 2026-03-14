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
- **Decision:** Generate synthetic examples for identified gaps (expanded scope after 35-test run)
- **Reasoning:** 35-test baseline identified more gaps than initial 10 tests. Training data should include: multi-NPC dialogue, register-shifting, combat sequences, moral dilemma scenes, brevity-matching examples, and explicit internal monologue.

## D-006: Think mode is the #1 deployment blocker
- **Date:** 2026-03-15
- **Decision:** Fix think mode leakage at Modelfile/inference layer BEFORE training
- **Reasoning:** 4/35 tests (11.4%) produced completely empty output because entire response was consumed by `<think>` blocks. `PARAMETER stop "<think>"` is unreliable. Options: (1) add `/no_think` instruction to system prompt, (2) retry with empty-output detection at inference time, (3) test `PARAMETER think false` if Ollama supports it. This is a serving issue — training won't fix it.

## D-007: Revised baseline assessment (4.2 -> 3.63)
- **Date:** 2026-03-15
- **Decision:** Expanded testing from 10 to 35 tests confirms training IS needed
- **Reasoning:** 10-test average (4.2) was misleadingly high. 35-test average (3.63) is below the 4.0 threshold. Think leakage (11.4%), combat avoidance (2/5), moral code failure (2/5), and emotional escalation plateau (3/5) are all training targets. The base model is strong on genre voice, atmosphere, and mixed format but weak on character code enforcement, pacing, and reliable output generation.
