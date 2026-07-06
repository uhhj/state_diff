# Phase3.10b Learned Future Quality + Rollout Error Audit Start

- Timestamp: 2026-07-06T23:25:07+08:00
- Main branch: Experiment1
- Main HEAD: 87a2a01e2963c2832e5300cfb2e8c62a78b7447a
- origin/Experiment1: 87a2a01e2963c2832e5300cfb2e8c62a78b7447a

## Main status

?? checkpoints/
?? reports/phase3_10b_future_rollout_audit_start.md

## Submodule

 388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)

## Objective

Run a small per-step trace audit after Phase3.10 failed to support repaired learned rollout:
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- diagnose predicted future quality, closed-loop history shift, and action geometry under predicted future
