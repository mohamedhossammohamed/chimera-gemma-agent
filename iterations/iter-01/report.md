# Iteration iter-01 — 2026-08-27T10:21:14.221003

- **Model:** google/gemma-3-27b-it:free (mock)
- **n:** 423 tasks {'1': 195, '2': 153, '3': 75}
- **Pipeline:** OpenMed CPU (V-PSA fix, list negation, site-norm) → Gated Fusion (6ch) → Gemma + PyCox
- **Mean risk:** 0.847
- **Defects fixed this iter:** 15 from Wave01 (4 missing emb, PC1 collapse site-norm, V-PSA flag, pmhx, etc.)
- **Tests:** 4/4 passed (parsers, fusion, gemma mock, survival C-index)
- **Provenance:** every field badge CALCULATED/UPLOADED, no secret committed

Sample output: `{"decision": "yes", "confidence": "clear", "variable_weights": {"pirads": "decisive", "psa": "important", "psad": "important", "bx": "important", "age": "noted", "dre": "noted", "fh": "noted", "comorb...`

Next: Iter-02 will tune fusion weights + add 5-ARI×2.
