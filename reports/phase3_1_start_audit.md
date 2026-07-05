# Phase3.1 Medium Evidence Audit Start

- Timestamp: `2026-07-05T21:03:43.536318+08:00`
- Main branch: `Experiment1`
- Main HEAD: `13d9c6c0373a6d110451cb7bb9fb020dc05cc1e9`
- origin/Experiment1: `13d9c6c0373a6d110451cb7bb9fb020dc05cc1e9`
- local_medium_commit_not_pushed: `false`

## Main Status

```text
M reports/phase3_code_reading_notes.md
 M scripts/phase3_pre_medium_audit.py
?? reports/phase3_1_start_audit.md
?? scripts/phase3_1_code_hazard_audit.py
?? scripts/phase3_1_medium_evidence_audit.py
?? scripts/phase3_1_run_audit.sh
```

## Submodule

```text
388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)
```

## Purpose

Review Phase3 medium WARN items before rollout / Phase4:
- branch-bias evidence on primary pair free vs hidden_breakaway_pin
- low future-error contrast warning
- paper_state vs state_action near-identical warning
- action / IDM OOD warning
- hidden metadata leakage safeguards
- stale / hard-coded code hazards
