# Iteration 08 — 2026-08-27T10:22:08.539410

- **Model:** `google/gemma-4-26b-a4b-it:free` (mock — RAM-safe mock for 423, live for single-case ablation)
- **n:** 423 tasks {'1': 195, '2': 153, '3': 75}
- **Focus:** Normalize family_history 7 variants → 3 (yes/no/unknown); Yes 11% not 17%
- **Pipeline:** `OpenMed CPU (95% cut, V-PSA flag, list negation not_assessed LNI) → Gated Fusion (6ch, site-norm) → Gemma (18 RPM) + PyCox (discrete hazard)`
- **Metrics:** `tests 15/15 PASS`, `C-index 0.70` (iter04), `KM 83.9/78.3/69.5`, `psa-psad ρ=0.745`, `pirads-cspca 0.620`
- **Defects closed this iter:** Normalize family_history 7 variants → 3 (yes/no/unknown)
- **Provenance:** every field `CALCULATED`/`UPLOADED`, `.env` gitignored, no `sk-or-` in repo
- **Artifacts:** `report.json`, `report.md`, `.swarm/wave-*-*.md`, `docs/iterations/iter-08.html`

**Next:** Iter-09 Polish GitHub Pages iteration gallery + provenance badges CALCULATED/UPLOADED + docs timeline

**Swarm:** Diagnostic→Research→Treatment→Integration→Red-Team→Supervisor (goal-terminated, evidence-required, cold-context)
