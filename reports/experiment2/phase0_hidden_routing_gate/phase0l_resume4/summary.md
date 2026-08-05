# Experiment2 Phase 0L Resume4 — Frozen Public Routing Layout

- Public layout mode: `frozen_at_branch_arm`
- Seed 71001 preferred probe index: `10`
- Seed 71001 selected probe index: `7`
- Seed 71001 offline all-bead roof clearance: `0.002049999999999998 m`
- Seed 71001 runtime initial fixture clearance: passed; exact value was not persisted
- Engineering blocker: `stage-1 and final pulls must be collinear`
- Blocker location: seed 71001 free branch, after preload and immediately before main routing motion
- Root cause: the current no-action-end endpoint pose drifted tangentially from the frozen branch-arm target line, so the frozen stage-1/final targets failed the existing primitive's strict direction-equality precondition relative to the current pose0
- Action/free-task/hidden-task public-layout match: `N/A`; the pair-level exact gate was not reached
- Completed accepted pairs: `0`
- Seeds 71002 and 71003: not run
- Raw artifacts retained on server: `2` observation PNGs
- Verdict: `HIDDEN_ROUTING_GATE_RESUME4_SMOKE_BLOCKED`
- Scientific status: `UNTESTED`
- Training performed: `False`
- Geometry/action/outcome search performed: `False`
- Visualization performed: `False`
- No retry or post-result mode, target, threshold, action, geometry, outcome, or seed change was made.
