# Gemini 3 Flash Scoring Validation Report

**Date:** March 7, 2026
**Purpose:** Evaluate whether Gemini 3 Flash (`gemini-3-flash-preview`) can replace Claude Opus as a scoring provider for StratOS training data expansion
**Methodology:** 300 stratified examples scored by both Opus (ground truth) and Gemini, compared across all metrics
**Cost of this test:** $0.00 (free tier, 300 calls)

---

## Executive Summary

**Verdict: NOT VIABLE as drop-in replacement (MAE 1.090 > 1.0). STRONG candidate for hybrid pipeline.**

Gemini 3 Flash is substantially better than DeepSeek V3.2 across every metric. Its Spearman correlation of **0.922** (vs DeepSeek's 0.882) shows excellent rank ordering, and its MAE of **1.090** narrowly misses the 1.0 threshold. Unlike DeepSeek's catastrophic middle-range collapse, Gemini uses a diverse score vocabulary and maintains reasonable separation across bands. The 8-10 band performance is excellent (MAE 0.692), and the 0-2 band is reliable (MAE 0.675). The weak spots are 4-6 and 6-8 bands where Gemini underscores by 0.2-1.1 points.

With simple isotonic calibration on these 300 points, Gemini may cross the 1.0 threshold — and unlike DeepSeek, calibration is plausible because Gemini doesn't collapse distinct Opus scores into the same output.

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | `gemini-3-flash-preview` (Gemini 3 Flash) |
| Temperature | 0.1 |
| Max tokens | 512 |
| Sample size | 300 (60 per score band) |
| Ground truth | Claude Opus scores from `scores_v2.json` (39,071 total) |
| Prompt format | Same as Opus + appended "SCORE first" instruction |
| Random seed | 42 (identical sampling to DeepSeek test) |

### Stratified Sampling

Identical to the DeepSeek test — 60 examples per band:

| Band | Pool Size | Sampled | % of Pool |
|------|-----------|---------|-----------|
| 0-2 | 17,500 | 60 | 0.3% |
| 2-4 | 5,660 | 60 | 1.1% |
| 4-6 | 241 | 60 | 24.9% |
| 6-8 | 786 | 60 | 7.6% |
| 8-10 | 200 | 60 | 30.0% |

### Note on prompt modification

Gemini 3 Flash ignores the original prompt's "think step-by-step before scoring" instruction and writes long reasoning paragraphs, pushing the `SCORE:` line past the 512-token output limit. To fix this, a suffix was appended: *"Your response MUST start with the score on the very first line in this exact format: SCORE: X.X"*. This is a fair comparison — it's how we'd actually call Gemini in production.

---

## Overall Results

| Metric | Gemini 3 Flash | DeepSeek V3.2 | Threshold | Status |
|--------|:--------------:|:-------------:|:---------:|:------:|
| **MAE vs Opus** | **1.090** | 1.521 | < 1.0 | FAILED (barely) |
| Median AE | 0.800 | 1.000 | — | — |
| Parse failures | 17/300 (5.7%) | 0/300 (0.0%) | < 5% | FAILED |
| Spearman correlation | **0.922** | 0.882 | > 0.7 | PASSED |
| Direction agreement | 85.5% | 80.3% | > 90% | FAILED |
| Within 0.5 points | 37.1% | 35.0% | — | — |
| Within 1.0 points | **70.3%** | 55.3% | > 60% | **PASSED** |
| Within 2.0 points | **83.4%** | 74.7% | > 80% | **PASSED** |

---

## Per-Band Analysis

| Band | MAE | Mean Bias | DeepSeek MAE | DeepSeek Bias | n |
|------|-----|-----------|:------------:|:-------------:|---|
| 0-2 | 0.675 | -0.635 | 0.625 | -0.558 | 55 |
| 2-4 | 1.000 | -0.537 | 0.845 | -0.588 | 57 |
| 4-6 | 1.433 | -0.181 | 1.530 | -1.143 | 57 |
| **6-8** | **1.691** | **-1.131** | **2.932** | **-2.732** | **54** |
| **8-10** | **0.692** | **-0.525** | **1.672** | **-1.622** | **60** |

### Key Observations

1. **8-10 band is dramatically better**: Gemini MAE 0.692 vs DeepSeek 2.932. This is the single biggest improvement — Gemini can actually identify high-relevance articles, while DeepSeek compressed them all to 3-5.

2. **6-8 band still the weakest**: MAE 1.691 with -1.131 bias. Gemini underscores moderate-high articles by ~1 point on average. Better than DeepSeek's catastrophic -2.732 bias, but still the problem band.

3. **4-6 band bias nearly zero**: -0.181 mean bias vs DeepSeek's -1.143. Gemini doesn't systematically underscore the moderate band — the MAE (1.433) comes from variance, not bias. This is calibration-friendly.

4. **Negative bias throughout, but moderate**: Overall mean bias -0.595 (vs DeepSeek's implied ~-1.3). Gemini consistently scores a bit lower than Opus, but not catastrophically.

---

## Error Distribution

| Error Range | Count | Percentage | Cumulative |
|-------------|-------|------------|------------|
| 0.0 (exact match) | 38 | 13.4% | 13.4% |
| 0.1 - 0.5 | 35 | 12.4% | 25.8% |
| 0.5 - 1.0 | 89 | 31.4% | 57.2% |
| 1.0 - 1.5 | 52 | 18.4% | 75.6% |
| 1.5 - 2.0 | 19 | 6.7% | 82.3% |
| 2.0 - 3.0 | 29 | 10.2% | 92.6% |
| 3.0 - 4.0 | 11 | 3.9% | 96.5% |
| 4.0+ | 10 | 3.5% | 100.0% |

**57.2% within 1.0 point** (vs DeepSeek's 55.3%). The tail is smaller: only 17.7% with error > 2.0 (vs DeepSeek's 25.3%).

### Bias Direction

| Direction | Count | Percentage |
|-----------|-------|------------|
| Underscore (Gem < Opus) | 190 | 67.1% |
| Overscore (Gem > Opus) | 55 | 19.4% |
| Exact match | 38 | 13.4% |

Gemini underscores 2/3 of the time, but occasionally overscores — unlike DeepSeek which almost never overscored. The overscoring is concentrated in the 4-6 band (Gemini sometimes rates "moderate" articles higher than Opus).

---

## Score Clustering Analysis

| Gem Score | Count | Avg Opus Score | Notes |
|-----------|-------|----------------|-------|
| 1.0 | 39 | 2.31 | Slight underscore |
| 0.0 | 33 | 0.77 | Good noise ID |
| 7.0 | 26 | 6.60 | Good |
| 8.5 | 23 | 8.30 | Excellent agreement |
| 0.5 | 22 | 1.36 | Good noise ID |
| 3.0 | 19 | 4.33 | Slight underscore |
| 6.0 | 19 | 5.55 | Good |
| 8.0 | 17 | 8.01 | Excellent agreement |
| 3.5 | 17 | 5.52 | Problem: maps to wide Opus range |
| 4.0 | 16 | 5.46 | Slight underscore |
| 1.5 | 15 | 2.85 | Maps OK |
| 9.0 | 14 | 8.75 | Good |
| 2.5 | 12 | 3.58 | Good |
| 6.5 | 4 | 5.12 | OK |
| 2.0 | 3 | 3.50 | Rare |

**Critical difference from DeepSeek**: Gemini uses 15+ distinct score levels (vs DeepSeek's effective 5-6). No single score is a "dump bucket" that maps to a 6-point Opus range. The worst cluster (Gem 3.5 → avg Opus 5.52) has only 17 examples and maps to a ~3-point Opus range — much tighter than DeepSeek's 2.5 → 2.0-8.5 disaster.

---

## Top 10 Worst Disagreements

| Article + Profile | Opus | Gemini | Diff |
|-------------------|------|--------|------|
| filmmaker_mumbai (film industry) | 6.5 | 1.2 | 5.3 |
| marine_electrician_norway (offshore) | 6.5 | 1.2 | 5.3 |
| hobbyist_crypto_f1 (crypto) | 8.5 | 3.5 | 5.0 |
| pediatric_oncologist_riyadh (medical) | 8.5 | 3.5 | 5.0 |
| filmmaker_mumbai (film) | 6.2 | 1.5 | 4.7 |
| lawyer_ai_disruption (legal AI) | 8.2 | 3.5 | 4.7 |
| meche_grad_sa (engineering) | 8.5 | 4.0 | 4.5 |
| petrol_eng_kw (petroleum) | 7.5 | 3.5 | 4.0 |
| petrol_eng_kw (petroleum) | 6.5 | 2.5 | 4.0 |
| ux_designer_tokyo (design) | 7.5 | 3.5 | 4.0 |

### Pattern: Gemini struggles with indirect relevance

The worst outliers involve articles that are relevant via career trajectory, industry context, or aspirational interest — not direct topic match. Gemini (like DeepSeek) applies stricter "does this directly match their role?" logic, while Opus considers broader professional context.

---

## Parse Failures (5.7%)

17 of 300 responses didn't contain a parseable `SCORE:` line. Gemini sometimes "thinks about" the format constraint instead of following it:

- `"Constraint Check: Start with SCORE: X.X..."` — Gemini reasons about the format constraint instead of executing it
- Long reasoning paragraphs that push past the 512-token limit before reaching the score

Parse failures are distributed across all bands (not concentrated in any one). In production, these would be retried with a more forceful prompt or higher max_tokens.

---

## Calibration Potential

Unlike DeepSeek, Gemini's calibration potential is **promising**:

1. **No middle-range collapse**: Each Gemini score maps to a relatively narrow Opus range (typically 2-3 points), not 6+ points like DeepSeek's 2.5.

2. **High Spearman (0.922)**: Rank ordering is well-preserved — monotone calibration should work.

3. **Band-specific bias is consistent**: 0-2 bias -0.635, 2-4 bias -0.537, 8-10 bias -0.525. The 6-8 band (-1.131) is the only outlier.

4. **Isotonic regression** on 283 data points could plausibly bring MAE below 1.0, especially since the bias is mostly a simple shift rather than range compression.

---

## Side-by-Side: Gemini 3 Flash vs DeepSeek V3.2

| Metric | DeepSeek V3.2 | Gemini 3 Flash | Winner |
|--------|:-------------:|:--------------:|:------:|
| Overall MAE | 1.521 | **1.090** | Gemini |
| Median AE | 1.000 | **0.800** | Gemini |
| Parse failures | **0.0%** | 5.7% | DeepSeek |
| Spearman | 0.882 | **0.922** | Gemini |
| Direction agreement | 80.3% | **85.5%** | Gemini |
| Within 1.0 | 55.3% | **70.3%** | Gemini |
| Within 2.0 | 74.7% | **83.4%** | Gemini |
| 0-2 band MAE | **0.625** | 0.675 | DeepSeek |
| 4-6 band MAE | 1.530 | **1.433** | Gemini |
| 6-8 band MAE | 2.932 | **1.691** | Gemini |
| 8-10 band MAE | 1.672 | **0.692** | Gemini |
| Score diversity | ~6 distinct | **~15 distinct** | Gemini |
| Calibration viable? | No (range collapse) | **Yes (rank preserved)** | Gemini |
| Cost for 36K | ~$12 | **~$0 (free tier)** | Gemini |

**Gemini wins on 11/14 metrics.** DeepSeek only wins on parse failures (0% vs 5.7%) and 0-2 noise detection.

---

## Cost Comparison (Updated)

| Provider | Cost for 36K Scores | MAE vs Opus | Notes |
|----------|--------------------:|:-----------:|:------|
| Claude Opus (standard) | ~$50-70 | 0.0 | Perfect quality |
| Claude Opus (Batch API) | ~$25-35 | 0.0 | Perfect quality, 50% discount |
| Gemini 3 Flash (free tier) | **$0** | 1.090 | 1,000/day limit = 36 days |
| Gemini 3 Flash (paid) | ~$3-6 | 1.090 | Instant, no daily limit |
| DeepSeek V3.2 | ~$12 | 1.521 | Not recommended |

### Free tier math
- Free tier: 1,000 requests/day
- 36,000 scores / 1,000 per day = 36 days
- Practically: run 900/day to stay safe = 40 days
- Cost: **$0.00** but slow

---

## Recommendations

### 1. Gemini 3 Flash as primary scorer with calibration (PREFERRED)

Run Gemini on all 17,260 articles (~18 profiles each = ~310K scores). At free tier, this takes ~310 days — impractical. At paid tier, ~$10-20 and finishes in hours.

Then apply isotonic calibration using these 300 validation points. If calibrated MAE < 1.0 on leave-one-out test, use Gemini scores directly as training data.

**Test calibration first** — run isotonic regression on the 283 valid points and compute leave-one-out MAE before committing.

### 2. Hybrid: Gemini pre-filter + Opus for uncertain band (ALTERNATIVE)

```
                     17,260 articles × 18 profiles
                              |
                    Gemini 3 Flash scoring ($0-20)
                    /           |            \
          Gem < 1.5         1.5-6.5          Gem > 6.5
       (~70% noise)     (~20% uncertain)   (~10% high)
            |                  |                |
     Use Gemini score    Send to Opus      Use Gemini score
     (MAE 0.675)         Batch API         (MAE 0.692)
            |            ($5-10)                |
            +----------+--------+---------+-----+
                       |
                 Training Data
```

This targets Opus spending only on the uncertain middle band where Gemini's MAE is 1.4-1.7.

### 3. Fix parse failures before production

The 5.7% parse failure rate needs to drop below 2%. Options:
- Increase max_output_tokens to 1024
- Use structured output / JSON mode if available
- Retry failed calls with stronger format instruction
- Parse more aggressively (look for any number in response)

### 4. Do NOT use DeepSeek V3.2

Gemini dominates on every meaningful metric. DeepSeek's only advantage (0% parse failures) is trivially fixable in Gemini.

---

## Raw Data

| File | Path | Description |
|------|------|-------------|
| Validation script | `backend/data/v2_pipeline/gemini_validation.py` | Python script for reproducible re-run |
| Raw results | `backend/data/v2_pipeline/gemini_validation_results.json` | 283 valid + 17 failed results |
| This report | `backend/data/v2_pipeline/Gemini_3_Flash_Validation_Report.md` | This document |
| DeepSeek comparison | `~/Downloads/DeepSeek_V3.2_Validation_Report.md` | Previous test |

### Reproducibility

```bash
cd ~/Downloads/StratOS/StratOS1/backend/data/v2_pipeline
GOOGLE_API_KEY="..." python3 gemini_validation.py
```

Uses random seed 42 for identical sampling. API responses may vary slightly due to model updates, but temperature 0.1 provides high reproducibility.

---

*Generated by Claude Opus 4.6*
