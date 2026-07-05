# Phase2.5b Recoverability Audit Fix Start

- Timestamp: `2026-07-05T11:00:55+08:00`
- Main branch: `Experiment1`
- Main HEAD: `ded17b1b667d52e5157bc93e746153a134e2c1e1`
- origin/Experiment1: `ded17b1b667d52e5157bc93e746153a134e2c1e1`

## Main status

```text
?? reports/phase2_5b_code_audit.md
?? reports/phase2_5b_start_audit.md
```

## Submodule

```text
90c0b842f86ef4a9de0101e5e3d5a14a9db8cde7 external/deformable-ravens (heads/ccda-cable)
```

## Objective

Fix recoverability audit validity:
1. Search sanity must pass on free.
2. Add breakaway-specific oracle.
3. Tune hidden_breakaway_pin / hidden_partial_pin into a hard-but-recoverable branch.
4. Do not run Phase3 medium until a recoverable branch is selected.
