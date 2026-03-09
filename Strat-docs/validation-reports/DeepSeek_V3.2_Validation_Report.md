# DeepSeek V3.2 Scoring Validation Report

**Date:** March 7, 2026
**Purpose:** Evaluate whether DeepSeek V3.2 (`deepseek-chat`) can replace Claude Opus as a scoring provider for StratOS training data expansion
**Methodology:** 300 stratified examples scored by both Opus (ground truth) and DeepSeek, compared across all metrics
**Cost of this test:** ~$0.10 (300 API calls)

---

## Executive Summary

**Verdict: NOT VIABLE as a drop-in replacement. POTENTIALLY VIABLE with calibration.**

DeepSeek V3.2 produces scores with a systematic negative bias (underscoring by 1-3 points), resulting in an overall MAE of **1.521** against the <1.0 threshold. However, the strong Spearman correlation of **0.882** indicates that DeepSeek preserves rank ordering — it consistently underscores rather than randomly disagreeing. This opens the door to calibration-based approaches that could make DeepSeek usable at ~$12 for 36,000 examples vs ~$50-70 for Opus.

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| DeepSeek model | `deepseek-chat` (V3.2) |
| Temperature | 0.1 |
| Max tokens | 512 |
| Sample size | 300 (60 per score band) |
| Ground truth | Claude Opus scores from `scores_v2.json` (39,071 total) |
| Prompt format | Identical to Opus — single user message with profile + article + scoring rubric |
| Random seed | 42 |

### Stratified Sampling

The 39,071 Opus-scored examples are heavily skewed toward noise (79.6% in 0-2 band). To ensure meaningful coverage of all score bands, 60 examples were sampled from each band:

| Band | Pool Size | Sampled | % of Pool |
|------|-----------|---------|-----------|
| 0-2 | 17,500 | 60 | 0.3% |
| 2-4 | 5,660 | 60 | 1.1% |
| 4-6 | 241 | 60 | 24.9% |
| 6-8 | 786 | 60 | 7.6% |
| 8-10 | 200 | 60 | 30.0% |

---

## Overall Results

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| **MAE vs Opus** | **1.521** | < 1.0 | FAILED |
| Median Absolute Error | 1.000 | — | — |
| Parse failures | 0/300 (0.0%) | < 5% | PASSED |
| Spearman correlation | 0.882 (p=1.34e-99) | > 0.7 | PASSED |
| Direction agreement (>5 vs <5) | 80.3% | > 90% | FAILED |
| Within 0.5 points | 35.0% | — | — |
| Within 1.0 points | 55.3% | > 60% | FAILED |
| Within 2.0 points | 74.7% | > 80% | FAILED |

---

## Per-Band Analysis

| Band | MAE | Mean Bias | Min Bias | Max Bias | n |
|------|-----|-----------|----------|----------|---|
| 0-2 | 0.625 | -0.56 | -1.5 | +0.5 | 60 |
| 2-4 | 0.845 | -0.59 | -2.0 | +2.7 | 60 |
| 4-6 | 1.530 | -1.14 | -2.5 | +2.8 | 60 |
| **6-8** | **2.932** | **-2.73** | **-6.0** | **+2.0** | **60** |
| 8-10 | 1.672 | -1.62 | -6.5 | +0.5 | 60 |

### Key Observations

1. **Monotonically increasing bias**: The negative bias grows with Opus score level. DeepSeek treats noise similarly to Opus but drastically underscores high-relevance articles.

2. **6-8 band is catastrophic**: DeepSeek scores articles that Opus rates 6-8 as 3-5 on average. This is the most damaging band because it contains articles with genuine but moderate relevance — exactly the nuanced cases where scoring quality matters most for training.

3. **0-2 band is good enough**: MAE 0.625 with only -0.56 bias means DeepSeek could reliably identify noise articles for pre-filtering, even without calibration.

4. **High-score compression**: DeepSeek rarely scores above 8.5 (only 24/300 samples scored 8.5+), while Opus uses the full 0-10 range. DeepSeek compresses everything into a narrower effective range.

---

## Error Distribution

| Error Range | Count | Percentage | Cumulative |
|-------------|-------|------------|------------|
| 0.0 (exact match) | 32 | 10.7% | 10.7% |
| 0.1 - 0.5 | 73 | 24.3% | 35.0% |
| 0.6 - 1.0 | 61 | 20.3% | 55.3% |
| 1.1 - 1.5 | 48 | 16.0% | 71.3% |
| 1.6 - 2.0 | 10 | 3.3% | 74.7% |
| 2.1 - 3.0 | 37 | 12.3% | 87.0% |
| 3.1 - 4.0 | 19 | 6.3% | 93.3% |
| 4.0+ | 20 | 6.7% | 100.0% |

**55.3% of scores are within 1.0 point** — meaning nearly half disagree by more than 1 point. The long tail (25.3% with error > 2.0) is concerning for training data quality.

---

## DeepSeek Score Clustering

DeepSeek V3.2 shows strong clustering around specific scores:

| DS Score | Count | Avg Opus Score | Notes |
|----------|-------|----------------|-------|
| 0.0 | 35 | 0.80 | Good noise identification |
| 0.5 | 23 | 1.41 | Good noise identification |
| 1.0 | 9 | 1.50 | Good |
| **1.5** | **59** | **3.11** | **Over-used, maps to wide Opus range** |
| 2.0 | 1 | 2.00 | Rare |
| **2.5** | **56** | **4.63** | **Over-used, maps to wide Opus range** |
| **3.5** | **36** | **5.58** | **Maps to moderate-high Opus range** |
| 4.0 | 4 | 7.38 | Rare, severely underscored |
| 4.5 | 4 | 7.25 | Rare, severely underscored |
| 6.2 | 22 | 6.88 | Reasonable |
| 6.8 | 5 | 6.84 | Good agreement |
| 7.2 | 5 | 8.04 | Slight underscore |
| 7.5 | 16 | 8.17 | Consistent 1-point underscore |
| 8.5 | 20 | 8.53 | Good agreement |
| 9.0 | 4 | 8.68 | Good |

**Problem**: DeepSeek uses 1.5 and 2.5 as "dump" scores for anything it considers low-to-moderate relevance, even when Opus scored 4-6. This creates a bimodal failure mode — low-relevance articles are scored fine, high-relevance articles are severely underscored, and the middle range is mapped to just two scores.

---

## Top 10 Worst Disagreements

| Article + Profile | Opus | DeepSeek | Diff |
|-------------------|------|----------|------|
| cpeg_student_kw (embedded systems) | 9.0 | 2.5 | 6.5 |
| meche_grad_sa (engineering) | 8.5 | 2.5 | 6.0 |
| cpeg_student_kw (tech article) | 8.5 | 2.5 | 6.0 |
| cpeg_student_kw (industry news) | 8.5 | 2.5 | 6.0 |
| cpeg_student_kw (tech industry) | 7.5 | 1.5 | 6.0 |
| chef_mexico_city (food industry) | 7.2 | 1.5 | 5.7 |
| ux_designer_tokyo (design) | 8.5 | 3.5 | 5.0 |
| meche_grad_sa (engineering) | 6.5 | 1.5 | 5.0 |
| undeclared_student_chicago | 6.5 | 1.5 | 5.0 |
| filmmaker_mumbai (film industry) | 7.5 | 2.5 | 5.0 |

### Profile-level pattern in outliers

| Profile | Appearances in Top 20 |
|---------|----------------------|
| cpeg_student_kw | 5 |
| filmmaker_mumbai | 3 |
| meche_grad_sa | 2 |
| undeclared_student_chicago | 2 |
| architect_qatar | 2 |
| finance_student_kw | 2 |

DeepSeek particularly struggles with student/junior profiles where relevance depends on career trajectory and aspirational interest rather than current role match. Opus's reasoning considers "this could affect their career path" while DeepSeek applies stricter "does this directly relate to their current situation" logic.

---

## Calibration Analysis

### Linear Calibration

```
Opus_predicted = 0.892 * DeepSeek_score + 1.672
Calibrated MAE: 1.182 (22% improvement, still above 1.0)
```

Linear calibration is insufficient because the bias is not uniform across the range.

### Bin-Based Calibration (Lookup Table)

Using the 300 data points as a calibration table (DS score -> average Opus score):

```
Calibrated MAE: 1.054 (in-sample, overfitted)
Leave-one-out MAE: 1.436 (more realistic)
```

The leave-one-out estimate (1.436) shows that simple bin calibration doesn't generalize well, primarily because DeepSeek clusters into too few distinct scores — a 2.5 from DeepSeek maps to Opus scores ranging from 2.0 to 8.5.

### Calibration Verdict

Calibration is **unlikely to bring MAE below 1.0** because the problem isn't just a shifted scale — DeepSeek collapses the middle range. When one DS score (e.g., 2.5) maps to Opus scores spanning 2.0-8.5, no monotone calibration function can distinguish them. The information is simply lost.

---

## Cost Comparison

| Provider | Cost for 36K Scores | MAE vs Opus | Calibration Needed |
|----------|--------------------:|:-----------:|:------------------:|
| Claude Opus (standard) | ~$50-70 | 0.0 (baseline) | No |
| Claude Opus (Batch API) | ~$25-35 | 0.0 (baseline) | No |
| DeepSeek V3.2 (raw) | ~$12 | 1.521 | Yes (insufficient) |
| DeepSeek V3.2 (calibrated) | ~$12 | ~1.2-1.4 | Yes (still above 1.0) |
| Gemini 2.5 Flash | ~$6 | Unknown | TBD |

---

## Recommendations

### 1. Do NOT use DeepSeek V3.2 for ground truth training data (raw or calibrated)

With MAE ~1.5 and severe underscoring of high-relevance articles, DeepSeek scores would teach the fine-tuned model to systematically underscore — the exact opposite of what we want. The middle-range collapse means calibration cannot recover the lost information.

### 2. DeepSeek IS viable for noise pre-filtering

The 0-2 band performance (MAE 0.625) means DeepSeek can reliably identify noise articles. A two-stage pipeline could work:
- **Stage 1**: DeepSeek scores all 17,260 articles (~$5.66). Articles scoring < 1.0 are classified as noise (high confidence).
- **Stage 2**: Send only the non-noise articles (~3,500 based on current distribution) to Opus for ground truth (~$10-15 with Batch API).
- **Total cost**: ~$16-21 instead of $50-70. Saves 50-65% by filtering noise cheaply.

### 3. Test Gemini 2.5 Flash next

May have different bias characteristics. Worth a $0.10 validation test with the same 300 samples before committing to Opus.

### 4. Consider a hybrid approach for maximum cost efficiency

```
                     17,260 articles
                          |
                    DeepSeek pre-filter ($5.66)
                    /                    \
            DS < 1.0                   DS >= 1.0
        (~13,800 noise)           (~3,460 non-noise)
              |                          |
      Label as score 0.5            Send to Opus Batch API
      (free, MAE 0.625)            ($10-15 for ground truth)
              |                          |
              +----------+------+--------+
                         |
                   Training Data
                    (~17,260 examples)
```

This gives Opus-quality ground truth where it matters (moderate-to-high relevance articles) while using DeepSeek's reliable noise detection to avoid paying Opus $35-50 to confirm that obviously irrelevant articles are indeed irrelevant.

---

## Raw Data

### Files

| File | Path | Description |
|------|------|-------------|
| Validation script | `backend/data/v2_pipeline/deepseek_validation.py` | Python script for reproducible re-run |
| Raw results | `backend/data/v2_pipeline/deepseek_validation_results.json` | 300 result objects with scores and errors |
| This report | `~/Downloads/DeepSeek_V3.2_Validation_Report.md` | This document |

### Reproducibility

```bash
cd ~/Downloads/StratOS/StratOS1/backend/data/v2_pipeline
python3 deepseek_validation.py
```

Uses random seed 42 for identical sampling. API responses may vary slightly due to model updates, but temperature 0.1 provides high reproducibility.

---

## Appendix: Methodology Notes

1. **Prompt format**: The exact same user-message prompt sent to Opus (from `batch_input.jsonl`) was sent to DeepSeek. No system message — everything is in a single user message containing the scoring rubric, user profile, and article.

2. **Score parsing**: Regex extraction of `SCORE: X.X` pattern, with Chinese format fallback (`评分: X.X`). DeepSeek achieved 0% parse failures — it consistently follows the requested output format.

3. **Stratified sampling**: Equal band representation (60 per band) oversamples rare high-score articles. This is intentional — uniform sampling would yield ~240 noise articles and ~6 high-score articles, providing no statistical power for the bands that matter most.

4. **Spearman vs Pearson**: Spearman rank correlation (0.882) is more appropriate than Pearson for ordinal scoring data. It measures monotonic relationship strength regardless of scale compression.

5. **API details**: DeepSeek V3.2 pricing is $0.27/M input tokens + $1.10/M output tokens. Average ~400 input + ~200 output tokens per call = $0.000328/call. Total test cost: ~$0.10.

---

*Generated by Claude Opus 4.6 | Commit: c177c95*
