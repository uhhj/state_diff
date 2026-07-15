# Phase3.14b-r2.5.6 Stage D.1 Asymmetric Gate Audit

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r256_staged1_lower_tail_ground_truth_collapse_drives_directional_gate_coupling`
- Required next path: `AUDIT_CABLE_XY_SEGMENT_COLLAPSE_PROVENANCE_BEFORE_GATE_OR_BRANCH_REPAIR`

## Frozen replay

- Isolated workers exact: `true`
- Model SHA256: `4a9ebcdb9d1995602a501c8820e07c0efcb757f1e3efff4ac0f2c5a4a23421d2`
- Reverse candidate SHA256: `1ab9f8d0026f215bf9e3b8aafeb4f8799bc1f8d329d8773e415413699a6c6d2d`

## Directional threshold attribution

- Stage-D joint threshold: `2543.16681`
- Stage-D lower threshold: `2543.16681`
- Stage-D upper threshold: `3.93794639`
- Joint-to-upper ratio: `645.81042`
- Scale-floor saturation rate: `1`
- Lowest lower-tail driver ratio: `0.00124600759`

## Candidate directional gate

- Bonferroni lower threshold: `6678.81611`
- Bonferroni upper threshold: `4.74744661`
- Bonferroni calibration group acceptance: `0.976190476`
- Bonferroni probe row acceptance: `0.944444444`
- Bonferroni reverse combined row-any: `0`

## Frozen physical-valid branch

- Stage-D joint K=8 support: `0.717741935483871`
- Naive asymmetric K=8 support: `None`
- Bonferroni asymmetric K=8 support: `None`

## Boundary

- The Stage-B model, scheduler, split, normalization, seed, objective and K were unchanged.
- The Stage-D gate file and all historical evidence were not modified.
- No candidate gate was selected and no threshold was changed.
- Probe targets and model candidates were not used to fit the candidate thresholds.
- DeformableRavens was not modified or executed.
- No formal pilot, formal diffusion/reverse, IDM, action-diverse collection, candidate execution, Phase4 or CPS was run.
- No checkpoint, tensor, NPZ, image or video was persisted.
- `train_only_recommendation=None` and `selected_configuration=None`.
