# Phase3.4 Rollout Smoke Start

- Timestamp: `2026-07-05T22:26:54+08:00`
- Main branch: `Experiment1`
- Main HEAD: `0f825485c173cedd42a1c9810603f85966622f76`
- origin/Experiment1: `0f825485c173cedd42a1c9810603f85966622f76`

## Main status

```text
M scripts/phase3_policy_rollout.py
?? scripts/phase3_4_rollout_preflight.py
?? scripts/phase3_4_rollout_result_audit.py
?? scripts/phase3_4_run_rollout_smoke.sh
```

## Submodule

```text
388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)
```

## Objective

Run a small learned rollout smoke:
- TensorFlow-free Ravens runtime
- primary pair: free_vs_hidden_breakaway_pin
- diagnostic pair: free_vs_hidden_pin
- no Phase4
- no CPS
- no paper-level claims
