# CHIMERA Gemma Agent

Standalone pipeline for prostate cancer triage, built for resource-constrained environments.

**Pipeline:** `Raw clinical traces → OpenMed parser (CPU) → Gated fusion → Gemma 3 27B via OpenRouter + PyCox → Structured JSON`

This project adapts the [CHIMERA dashboard](https://mohamedhossammohamed.github.io/chimera-dashboard/) computations for batch processing of the 423-case `train_release` cohort.

## Overview

- **OpenMed parser (CPU):** Extracts structured fields (Gleason, margin, PSA kinetics) from free-text reports with 95% token reduction.
- **Gated fusion:** Combines MRI (1024-d), biopsy (960-d), and prostatectomy embeddings across 6 modalities, handling missing data.
- **Gemma 3 27B (OpenRouter):** Generates clinical decisions with structured JSON, rate-limited for the free tier (18 RPM) and with offline mock fallback.
- **PyCox head (Task 3):** Discrete-time survival model for recurrence prediction with right-censored data (75 cases, 19 events).

## Quickstart

```bash
cp .env.example .env  # add OPENROUTER_API_KEY
pip install -e .
python -m src.agent.pipeline --case /path/to/train_release/task1/PT-pseudo_*
python scripts/run_batch.py --data /path/to/train_release --out iterations/results/outputs
```

Without an API key the agent runs in offline mock mode, suitable for testing.

## Project structure

```
src/
  parsers/openmed_cpu.py
  llm/gemma_client.py
  survival/pycox_head.py
  fusion/gated.py
  agent/pipeline.py
iterations/
docs/  # GitHub Pages
```

## Results

Evaluated on `train_release` (423 cases: Task1 195, Task2 153, Task3 75):

- **Task1 biopsy decision (91 labeled):** accuracy 0.714, F1 0.809
- **Task2 treatment decision (72 labeled):** accuracy 0.500
- **Task3 survival (75, 19 events):** C-index 0.750, CAPRA-S ≥6 vs <6 risk ratio 2.86, mean predicted S12 0.847

See `docs/` for the full results gallery and `docs/agent-card.md` for model details.

## Security

`.env` is gitignored. Store `OPENROUTER_API_KEY` in the environment, never in version control.

## License

MIT
