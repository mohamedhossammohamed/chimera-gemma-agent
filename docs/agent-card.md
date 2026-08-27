# CHIMERA Gemma Agent — Agent Card

**Version:** 0.1.0 | **Model:** `google/gemma-3-27b-it:free` via OpenRouter | **License:** MIT

## Overview

Standalone agent for prostate cancer triage, designed for resource-constrained environments. Processes 423 cases across three tasks: biopsy decision, treatment decision, and time-to-recurrence.

## Architecture

```
Raw inputs (clinical notes, PSA trends, embeddings 1024/960)
  → OpenMed parser (CPU)
    → Structured JSON (95% token reduction)
  → Gated fusion (6 modalities, site normalization)
    → Fused vector (16-d)
  → Gemma 3 27B (OpenRouter, 18 RPM) + PyCox discrete hazard
    → Final JSON (decision, confidence, weights, rationale)
```

**OpenMed parser:** Recomputes PSA kinetics, normalizes PI-RADS, family history, and surgical pathology with negation handling.

**Gated fusion:** Handles missing modalities via learned gates; per-hospital normalization for MRI embeddings.

**Gemma client:** Reads `OPENROUTER_API_KEY` from environment, token-bucket rate limiting, exponential backoff, JSON mode, offline mock fallback.

**PyCox head:** Discrete-time survival (10 bins, Kaplan-Meier S(t)=Π(1-d_i/n_i), Greenwood variance) for Task 3.

## Intended use

Research use for prostate cancer risk stratification. Not for direct clinical decision-making without validation.

## Evaluation

- **Task1 (91 labeled):** accuracy 0.714, F1 0.809
- **Task2 (72 labeled):** accuracy 0.500
- **Task3 (75, 19 events/56 censored):** C-index 0.750, mean predicted S12 0.847

## Limitations

- Performance measured in mock mode; live API may vary
- Training cohort is enriched (83% PI-RADS 4-5), not screening population
- Embeddings show site effects; normalization applied
- Requires API key for live inference; offline mock available

## Reproducibility

```bash
cp .env.example .env
pip install -e .
python scripts/run_batch.py --data ~/Downloads/train_release --out iterations/results/outputs
pytest -q
```

## Security

API keys via environment only; `.env` is gitignored.
