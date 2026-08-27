# Iteration 07 — 2026-08-27T10:22:08.539051

- **Model:** `google/gemma-3-27b-it:free` (mock — RAM-safe mock for 423, live for single-case ablation)
- **n:** 423 tasks {'1': 195, '2': 153, '3': 75}
- **Focus:** Fix cT variants cT2/cTx/cT3 → cT2a/None/cT3a; EAU unclassified 64→12 cases
- **Pipeline:** `OpenMed CPU (95% cut, V-PSA flag, list negation not_assessed LNI) → Gated Fusion (6ch, site-norm) → Gemma (18 RPM) + PyCox (discrete hazard)`
- **Metrics:** `tests 15/15 PASS`, `C-index 0.70` (iter04), `KM 83.9/78.3/69.5`, `psa-psad ρ=0.745`, `pirads-cspca 0.620`
- **Defects closed this iter:** Fix cT variants cT2/cTx/cT3 → cT2a/None/cT3a
- **Provenance:** every field `CALCULATED`/`UPLOADED`, `.env` gitignored, no `sk-or-` in repo
- **Artifacts:** `report.json`, `report.md`, `.swarm/wave-*-*.md`, `docs/iterations/iter-07.html`

**Next:** Iter-08 Normalize family_history 7 variants → 3 (yes/no/unknown); Yes 11% not 17%

**Swarm:** Diagnostic→Research→Treatment→Integration→Red-Team→Supervisor (goal-terminated, evidence-required, cold-context)
