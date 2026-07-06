# Phase3.10 Controlled Learned Rollout Retry Start

- Timestamp: 2026-07-06T21:55:31+08:00
- Main branch: Experiment1
- Main HEAD: ae76cb48ce4460459a6e097df1982ad98650902f
- origin/Experiment1: ae76cb48ce4460459a6e097df1982ad98650902f

## Main status

?? checkpoints/
?? reports/phase3_10_learned_rollout_start_audit.md

## Submodule

 388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)

## Objective

Run a controlled learned rollout retry:
- DDPM predicted future
- repaired inverse dynamics
- decoded executable pick_place action
- PyBullet closed-loop rollout
- no training
- no future DDPM training
- no Phase4
- no CPS
- checkpoints local-only
