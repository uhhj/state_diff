# Phase3.8c Pose0-XY Repair Confirmation Start

- Timestamp: `2026-07-06T19:09:51+08:00`
- Main branch: `Experiment1`
- Main HEAD: `e7bcdc0e15512509cee47699ed9435f9e07ebc7e`
- origin/Experiment1: `e7bcdc0e15512509cee47699ed9435f9e07ebc7e`

## Main status

```text

```

## Submodule

```text
388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)
```

## Objective

Run a bounded no-early-stop confirmation probe:
- verify whether Phase3.8b pose0 XY repair signal holds across free, hidden_breakaway_pin, hidden_high_friction
- use state_action + idm_gt_future only
- run 2 windows per condition and 5 variants
- no Phase4
- no CPS
- no retraining
- GT-blended variants remain diagnostic upper bounds only
