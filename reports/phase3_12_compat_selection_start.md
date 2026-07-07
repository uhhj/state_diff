# Phase3.12 Future Target Locality / Compatibility-Aware Branch Selection Start

- Timestamp: 2026-07-07T14:51:25+08:00
- Main branch: Experiment1
- Main HEAD: 04ff1504eb7fc84e4d710caf967ba17bafc0d3c2
- origin/Experiment1: 04ff1504eb7fc84e4d710caf967ba17bafc0d3c2

## Main status

```text
?? checkpoints/
?? scripts/phase3_12_analyze_compat_selection.py
?? scripts/phase3_12_compat_selection_preflight.py
?? scripts/phase3_12_compat_selection_rollout.py
?? scripts/phase3_12_run_compat_selection.sh
```

## Submodule

```text
 388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)
```

## Objective

Run a compatibility-aware future target selection diagnostic:
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- compare ddpm_mean, input_nearest, condition_nearest, compat_global_topk, compat_condition_topk, compat_condition_action_geom
- condition-aware selectors are diagnostic upper bounds only
