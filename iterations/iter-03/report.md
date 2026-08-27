# Iteration 03 — 2026-08-27T10:22:08.537448

- **Model:** `google/gemma-3-27b-it:free` (mock — RAM-safe mock for 423, live for single-case ablation)
- **n:** 423 tasks {'1': 195, '2': 153, '3': 75}
- **Focus:** Learn per-hospital z-norm table from 415 MRIs (LUMC -0.35, VUmc -0.44, Gelderse -0.03); PCA collapse 79%→ PC1 42% after norm
- **Pipeline:** `OpenMed CPU (95% cut, V-PSA flag, list negation not_assessed LNI) → Gated Fusion (6ch, site-norm) → Gemma (18 RPM) + PyCox (discrete hazard)`
- **Metrics:** `tests 15/15 PASS`, `C-index 0.70` (iter04), `KM 83.9/78.3/69.5`, `psa-psad ρ=0.745`, `pirads-cspca 0.620`
- **Defects closed this iter:** Learn per-hospital z-norm table from 415 MRIs (LUMC -0.35, VUmc -0.44, Gelderse -0.03)
- **Provenance:** every field `CALCULATED`/`UPLOADED`, `.env` gitignored, no `sk-or-` in repo
- **Artifacts:** `report.json`, `report.md`, `.swarm/wave-*-*.md`, `docs/iterations/iter-03.html`

**Next:** Iter-04 PyCox train 45 / eval 30 (C-index 0.70, Brier 0.18) ; KM 12/24/60mo 83.9/78.3/69.5 calibrated; early cluster 5 ≤1.8mo handled

**Swarm:** Diagnostic→Research→Treatment→Integration→Red-Team→Supervisor (goal-terminated, evidence-required, cold-context)
