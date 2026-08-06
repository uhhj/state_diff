# Phase 0B: Hidden Local Friction Soft BlockPush

Phase 0B evaluates one tightly paired PyBullet counterfactual: a 72-node soft block is pushed by the same XArm joint-target sequence over either a uniform low-friction floor or a visually hidden local high-friction patch. The run is an audit dataset experiment, not training and not a paper-level claim.

## Frozen setup

- Pair: `sbp_074001`, seed `74001`.
- Grid: `6 x 4 x 3`; the 24 top-layer nodes are the formal visible keypoints.
- Floor: five coplanar, collision-only tiles. Only the patch lateral-friction coefficient changes between branches.
- Pairing: one settled low-friction state is saved and restored; probe and test IK are solved once before branching, then identical joint-target arrays are replayed.
- Formal observations: motor torque, joint reaction wrench, end-effector tracking error, and end-effector contact wrench.
- Oracle diagnostics: patch contact count, normal/tangential load, slip speed, and stick ratio. Oracle channels are excluded from formal observations and policy inputs.

The complete initial configuration and decision thresholds are in [`configs/experiment3/soft_blockpush_phase0b.json`](../../configs/experiment3/soft_blockpush_phase0b.json). The research-plan amendment is [`CCDA_RESEARCH_PLAN_V4_BLOCKPUSH_AMENDMENT.md`](CCDA_RESEARCH_PLAN_V4_BLOCKPUSH_AMENDMENT.md).

## Execution

On the configured server environment:

```bash
cd /data/Experiment3/state_diff
/miniforge3/envs/coord_bimanual/bin/python -m pytest -q \
  tests/test_block_pushing.py \
  tests/experiment3/test_soft_block_lattice.py \
  tests/experiment3/test_friction_tiles.py \
  tests/experiment3/test_soft_block_pushing.py \
  tests/experiment3/test_soft_block_metrics.py \
  tests/experiment3/test_soft_block_pairing.py

bash scripts/experiment3/phase0_soft_blockpush/run_phase0b.sh
```

The orchestration script always runs and analyzes probe-only first. It executes the full Pair only when the probe verdict is `PHASE0B_PROBE_MECHANISM_COMPLETE`.

## Scientific boundary

A full result may establish only that one strict single-Pair hidden-local-friction soft-body branch mechanism exists. It does not establish multi-seed robustness, held-out generalization, training readiness, or a publication conclusion. Absolute keypoint displacement alone is insufficient for a strong deformation claim; rigid-aligned residual and structural strain remain required.
