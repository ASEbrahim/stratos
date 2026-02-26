# StratOS V2 Scorer — Evaluation Report

Training loss: 0.0000, Training time: 0.0h

## Metrics Summary

| Metric | V1 Baseline | V2 Result |
|--------|-------------|-----------|
| Direction accuracy | 90.7% | 98.1% |
| PSR (Profile Sensitivity Ratio) | 39.7% | 24.4% |
| MAE | 1.553 | 0.393 |
| Spearman rho (aggregate) | -- | 0.750 |
| Think block emptiness rate | ~85% | 0.0% |
| Parse failures | -- | 0/2048 |
| PSR articles evaluated | -- | 561 |

## Per-Profile Spearman rho

| Profile | N | rho | Flag |
|---------|---|-----|------|
| Agricultural commodities trader in São Paulo, Brazil | 69 | 0.589 |  |
| Architect working on smart city infrastructure projects in D | 68 | 0.833 |  |
| Biotech researcher at KAUST studying gene therapy in Jeddah, | 68 | 0.621 |  |
| CTO of a mobile payments startup in Lagos, Nigeria | 68 | 0.723 |  |
| Computer Engineering student at American University of Kuwai | 67 | 0.879 |  |
| Corporate lawyer at a major law firm in London, UK | 67 | 0.704 |  |
| Cybersecurity analyst at a Kuwaiti bank in Kuwait | 67 | 0.790 |  |
| Data Scientist at a Dubai fintech startup in Dubai, UAE | 68 | 0.822 |  |
| Digital marketing manager at a K-pop entertainment agency in | 69 | 0.524 |  |
| Emergency department nurse practitioner in Toronto, Canada | 69 | 0.717 |  |
| Executive chef and restaurant owner in Mexico City, Mexico | 68 | 0.650 |  |
| Finance & Accounting student at GUST Kuwait in Kuwait | 68 | 0.759 |  |
| HVAC technician and small business owner in Houston, Texas,  | 69 | 0.751 |  |
| Hospital pharmacist specializing in oncology in Paris, Franc | 70 | 0.675 |  |
| Independent documentary filmmaker in Mumbai, India | 67 | 0.754 |  |
| Indie game developer and studio founder in Berlin, Germany | 68 | 0.615 |  |
| Marine electrician on offshore wind vessels in Stavanger, No | 69 | 0.689 |  |
| Mechanical Engineering fresh graduate seeking NEOM/Aramco ro | 68 | 0.738 |  |
| Mining engineer at a copper mine in Santiago, Chile | 69 | 0.758 |  |
| Pediatric oncologist at King Faisal Specialist Hospital in R | 68 | 0.674 |  |
| Petroleum Engineering student at Kuwait University in Kuwait | 68 | 0.732 |  |
| Pipeline welder and CWB-certified inspector in Edmonton, Alb | 70 | 0.746 |  |
| Professional bonsai artist and competition judge in Kyoto, J | 69 | 0.416 |  |
| Retired IT consultant in Lisbon, Portugal | 67 | 0.702 |  |
| Retired UN diplomat and policy consultant in Vienna, Austria | 68 | 0.716 |  |
| Senior geophysicist at KOC in Kuwait | 69 | 0.803 |  |
| Sports physiotherapist at a professional rugby club in Sydne | 69 | 0.669 |  |
| Supply chain analyst at a logistics company in Manama, Bahra | 67 | 0.696 |  |
| UX Designer at a consumer electronics company in Tokyo, Japa | 69 | 0.736 |  |
| Undeclared sophomore at University of Chicago in Chicago, Il | 68 | 0.813 |  |

## Profile-Awareness Sanity Check

### "NCT JNJM Debuts 'BOTH SIDES' with Dual Charms"

GT spread: 8.5, V1 spread: 2.0, V2 spread: 8.5

| Profile | GT | V1 | V2 |
|---------|----|----|-----|
| Digital marketing manager at a K-pop entertainment | 8.5 | 2.0 | 8.5 |
| Computer Engineering student at American Universit | 0.5 | 0.0 | 1.0 |
| Corporate lawyer at a major law firm in London, UK | 0.0 | 0.0 | 1.0 |
| Data Scientist at a Dubai fintech startup in Dubai | 0.0 | 0.0 | 1.0 |
| Marine electrician on offshore wind vessels in Sta | 0.0 | 0.0 | 0.0 |

### "Home of Indies auf der Gamescom 2026: Das neue Standkonzept"

GT spread: 8.5, V1 spread: 0.2, V2 spread: 8.5

| Profile | GT | V1 | V2 |
|---------|----|----|-----|
| Indie game developer and studio founder in Berlin, | 8.5 | N/A | 8.5 |
| Corporate lawyer at a major law firm in London, UK | 1.0 | 0.0 | 1.0 |
| Independent documentary filmmaker in Mumbai, India | 1.0 | 0.2 | 1.0 |
| Marine electrician on offshore wind vessels in Sta | 0.5 | 0.0 | 0.0 |
| Pipeline welder and CWB-certified inspector in Edm | 0.0 | 0.0 | 0.0 |

### "SC appoints Abdulkader inaugural fellow"

GT spread: 8.0, V1 spread: 3.0, V2 spread: 7.5

| Profile | GT | V1 | V2 |
|---------|----|----|-----|
| Finance & Accounting student at GUST Kuwait in Kuw | 8.5 | 3.0 | 8.5 |
| Undeclared sophomore at University of Chicago in C | 2.5 | 0.5 | 3.5 |
| Architect working on smart city infrastructure pro | 1.0 | 0.2 | 1.0 |
| Emergency department nurse practitioner in Toronto | 1.0 | 0.0 | 1.0 |
| Professional bonsai artist and competition judge i | 0.5 | 0.0 | 1.0 |

### "Diversity In Clinical Trials: Current Gaps And How To Fix Them"

GT spread: 8.0, V1 spread: 0.0, V2 spread: 7.5

| Profile | GT | V1 | V2 |
|---------|----|----|-----|
| Pediatric oncologist at King Faisal Specialist Hos | 8.5 | N/A | 8.5 |
| Independent documentary filmmaker in Mumbai, India | 4.0 | 0.2 | 3.5 |
| Retired UN diplomat and policy consultant in Vienn | 2.0 | 0.2 | 1.0 |
| CTO of a mobile payments startup in Lagos, Nigeria | 1.0 | 0.2 | 1.0 |
| Pipeline welder and CWB-certified inspector in Edm | 0.5 | 0.2 | 1.0 |

### "FAAN bans cash transactions for revenue payments effective Feb. 28"

GT spread: 8.0, V1 spread: 0.0, V2 spread: 7.5

| Profile | GT | V1 | V2 |
|---------|----|----|-----|
| CTO of a mobile payments startup in Lagos, Nigeria | 8.5 | N/A | 8.5 |
| Hospital pharmacist specializing in oncology in Pa | 1.0 | 0.0 | 1.0 |
| Mining engineer at a copper mine in Santiago, Chil | 1.0 | 0.0 | 1.0 |
| Digital marketing manager at a K-pop entertainment | 1.0 | 0.0 | 1.0 |
| Pipeline welder and CWB-certified inspector in Edm | 0.5 | 0.0 | 1.0 |

**Verdict:** Avg V1 spread: 1.04, Avg V2 spread: 7.90 — PASS

## Artifacts

- GGUF: `/home/ahmad/Downloads/StratOS/StratOS1/backend/data/v2_pipeline/training_output/v2_scorer.gguf`
- Final checkpoint: `/home/ahmad/Downloads/StratOS/StratOS1/backend/data/v2_pipeline/training_output/final_checkpoint`
- Sampler verification: `/home/ahmad/Downloads/StratOS/StratOS1/backend/data/v2_pipeline/training_output/sampler_verification.json`
- Training log: `/home/ahmad/Downloads/StratOS/StratOS1/backend/data/v2_pipeline/training_output/training_log.txt`

**NOT deployed. NOT registered in Ollama. Review this report before deploying.**