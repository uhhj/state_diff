# Phase3.9 Geometry-Aware IDM Repair Start

- Timestamp: `2026-07-06T19:33:28+08:00`
- Main branch: `Experiment1`
- Main HEAD: `4be9aaf5ffeab648e110e8f847347b598d43d447`
- origin/Experiment1: `4be9aaf5ffeab648e110e8f847347b598d43d447`

## Main status

```text

```

## Submodule

```text
388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)
```

## Objective

Train only a geometry-aware inverse dynamics model and run a controlled matched-prefix retry:
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- checkpoints are local-only and must not be committed
- GT-blended diagnostics remain diagnostic upper bounds only
