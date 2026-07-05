# Phase3.3 Rollout Runtime Fix Start

- Timestamp: `2026-07-05T21:51:34+08:00`
- Main branch: `Experiment1`
- Main HEAD: `a83b43c150cd841450386b93c1d6431c044723db`
- origin/Experiment1: `a83b43c150cd841450386b93c1d6431c044723db`

## Main status

```text
?? scripts/phase3_3_probe_ravens_imports.py
?? scripts/phase3_3_runtime_verdict.py
```

## Submodule

```text
388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)
```

## Objective

Fix rollout runtime readiness in `coord_bimanual` without running rollout:
- torch must import
- ravens must import
- pybullet must import
- numpy must import
- phase3 windows must load
- phase3 checkpoints must exist
- action codec must be dim 14
- no camera_config action path
