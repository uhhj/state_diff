# Phase3.12b Condition Proxy / Score Ablation Diagnostic Start

- Timestamp: `2026-07-10T09:56:05+08:00`
- Main branch: `Experiment1`
- Main HEAD: `55fa480797f693a7477ba607f9fa5ddc0ddc1a89`
- origin/Experiment1: `55fa480797f693a7477ba607f9fa5ddc0ddc1a89`

## Main status

```text
?? checkpoints/
?? reports/phase3_12_workers/
```

## Submodule

```text
388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)
```

## Objective

Run condition-proxy and score-ablation diagnostics:
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- test whether observable state/action-history proxies can replace condition_nearest upper bound
- test whether action-geometry score actually helps top-K future selection
