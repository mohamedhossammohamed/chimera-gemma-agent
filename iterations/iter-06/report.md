# Iteration 06 — 2026-08-27T10:22:08.538775

- **Model:** `google/gemma-4-26b-a4b-it:free` (mock — RAM-safe mock for 423, live for single-case ablation)
- **n:** 423 tasks {'1': 195, '2': 153, '3': 75}
- **Focus:** Add cribriform pattern feature to variable_weights (present in 47 T2 pathology reports, ρ with bx_isup 0.41)
- **Pipeline:** `OpenMed CPU (95% cut, V-PSA flag, list negation not_assessed LNI) → Gated Fusion (6ch, site-norm) → Gemma (18 RPM) + PyCox (discrete hazard)`
- **Metrics:** `tests 15/15 PASS`, `C-index 0.70` (iter04), `KM 83.9/78.3/69.5`, `psa-psad ρ=0.745`, `pirads-cspca 0.620`
- **Defects closed this iter:** Add cribriform pattern feature to variable_weights (present in 47 T2 pathology reports, ρ with bx_isup 0.41)
- **Provenance:** every field `CALCULATED`/`UPLOADED`, `.env` gitignored, no `sk-or-` in repo
- **Artifacts:** `report.json`, `report.md`, `.swarm/wave-*-*.md`, `docs/iterations/iter-06.html`

**Next:** Iter-07 Fix cT variants cT2/cTx/cT3 → cT2a/None/cT3a; EAU unclassified 64→12 cases

**Swarm:** Diagnostic→Research→Treatment→Integration→Red-Team→Supervisor (goal-terminated, evidence-required, cold-context)
