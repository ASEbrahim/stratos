# Qwen3.5 Model Evaluation Report
Date: 2026-03-07

## Qwen3.5-9B Base (Unfine-tuned) — Holdout Eval

**Full 1,500-sample holdout evaluation against Opus ground truth.**

| Metric | Qwen3.5-9B (base) | V2 Fine-tuned Qwen3-8B |
|--------|-------------------|----------------------|
| MAE | **1.297** | 1.544 |
| Median AE | 1.000 | — |
| Mean bias | +0.478 | — |
| Direction accuracy | 87.3% | — |
| Within 1.0 | 62.1% | — |
| Within 2.0 | 78.3% | — |
| Parse failures | 0 (0.0%) | — |
| Avg time/call | 1.86s | — |

### Per-Band MAE
| Band | MAE | Bias | N |
|------|-----|------|---|
| 0-2 (noise) | 1.132 | +0.542 | 1222 |
| 2-4 (tangential) | 2.034 | +0.585 | 176 |
| 4-6 (moderate) | 1.827 | -0.125 | 51 |
| 6-8 (high) | 2.243 | -0.638 | 42 |
| 8-10 (critical) | 1.889 | -1.778 | 9 |

### Key Findings
- **16% lower MAE than V2 fine-tuned model WITHOUT any fine-tuning**
- Zero parse failures (perfect format compliance)
- Positive bias (+0.478): slightly over-scores, especially in noise band
- High-relevance articles slightly under-scored (critical band bias -1.778)
- Fine-tuning Qwen3.5-9B should push MAE well below 1.0

## Qwen3.5-0.8B — Pre-filter Test (50 samples)

| Metric | Value |
|--------|-------|
| MAE | 3.630 |
| Direction accuracy | 30% |
| Mean bias | +3.49 |

**Verdict: NOT viable as pre-filter.** Massive positive bias, terrible accuracy.

## Qwen3.5-35B-A3B — Wizard/Briefing Test

Compared against current wizard model (qwen3:14b):

### Wizard (Profile Generation)
| Metric | 35B-A3B | 14B |
|--------|---------|-----|
| Time | 235.1s | 27.4s |
| Tokens | 2048 (hit limit) | 883 (complete) |
| Quality | Better keywords (e.g., "IRA tax credits", "FERC") | Good, complete output |

### Briefing Generation
| Metric | 35B-A3B | 14B |
|--------|---------|-----|
| Time | 48.4s | 21.7s |
| Tokens | 347 (truncated) | 592 (complete) |
| Quality | More polished prose | Well-structured, concise |

**Verdict: Keep qwen3:14b as wizard model.** The 35B-A3B produces marginally better prose
but is 2-8x slower and frequently hits token limits before completing. The speed difference
is unacceptable for interactive wizard/briefing use cases.

## VRAM Coexistence Test

Scorer (8.7GB) + qwen3:14b (9.3GB) = ~18GB models + KV cache > 24GB.
**FAILED** — cannot coexist in 24GB VRAM. Keep OLLAMA_MAX_LOADED_MODELS=1.

## OLLAMA_NUM_PARALLEL Benchmark

NUM_PARALLEL=4 gives 0.94x throughput (6% SLOWER than sequential).
GPU serializes inference. **Keep NUM_PARALLEL=1.**

## Recommendations

1. **Upgrade base model to Qwen3.5-9B for V2.2 training** — architecture improvement alone gives 16% MAE reduction
2. **Keep qwen3:14b as wizard/briefing model** — best speed/quality tradeoff
3. **Do NOT use qwen3.5:0.8b** — not viable for any scoring task
4. **Expected V2.2 MAE after fine-tuning: < 1.0** (base 1.297 + fine-tuning gains)
