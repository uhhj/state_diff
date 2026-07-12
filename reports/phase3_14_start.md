# Phase3.14 Start Audit

- Timestamp: `2026-07-12T02:11:16.709903+00:00`
- Main: `c2f38b8373ac17a47e53a78086ec66d0d3d5f2e0` (`Experiment1`)
- Submodule: `633a88752445cf5d6776ed374fdbbdb35f93050c` (`ccda-cable`)
- Phase3.13 audit: `PASS`
- Phase3.14 provenance gate: `FAIL`
- Root cause: `phase314_data_provenance_failed`

The formal manifest source hash for `ccda_phase3/schema_v2.py` does not match the current final source. Phase3.14 training is blocked; no deterministic, DDPM, IDM, candidate-support, query-local, Phase4, or CPS run was started.
