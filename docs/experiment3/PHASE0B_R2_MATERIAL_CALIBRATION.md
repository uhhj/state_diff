# Phase 0B-R2: True Microstep Material Calibration

R1 confirmed that C0 can deform and partly recover, but it was too compliant and exceeded edge/face gates. C2 suffered numerical runaway because Bullet internal substeps reused one stale spring force across the complete 240 Hz outer interval. R2 is restricted to material mechanics and does not run a Free/High Pair.

R2 replaces stale-force substeps with a manual loop. For each outer step and each selected microstep it rereads node state, vectorizes all structural/shear/bending edge evaluations, aggregates equal-and-opposite node forces, reapplies the current distributed load, and advances Bullet once at `outer_dt / M` with `numSubSteps=1`. The outer trace remains 240 Hz.

Material profiles retain the 8:3:0.8 stiffness ratio and vary only a multiplier A4/A6/A8. Damping is recomputed from a fixed damping ratio `zeta=0.25` and reduced node mass, and every explicit profile value is checked to `1e-12`.

Calibration is gated in this order:

1. A8 worst-case two-node and 2x2x2 mechanics validation over `M={4,8,16,32}` selects the smallest converged microstep count.
2. A6 axial coupon is run first; only its verdict may authorize A4 or A8.
3. The same permitted profile must pass shear.
4. That profile must pass a fully dynamic gravity/table settle without recentering or velocity reset.
5. Only then are profile and microstep files frozen.

This phase may not modify patch, pusher, goal, Pair actions, R1 Pair code, or old evidence. It may not create probe/full Pair outputs, run development seeds, or train State Diff, IDM, or CFPM.
