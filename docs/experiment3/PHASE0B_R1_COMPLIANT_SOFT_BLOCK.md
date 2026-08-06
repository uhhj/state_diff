# Phase 0B-R1: Compliant Soft Block

Phase 0B established a strict hidden-friction Pair, early physics-rate sensor response, and a global future-state branch. Its midpoint P2P lattice behaved as an over-constrained quasi-rigid truss: absolute state and COM diverged, while rigid-aligned deformation remained at micrometre scale and task-progress did not branch.

R1 preserves that implementation as `p2p_legacy` and introduces an explicit Kelvin–Voigt mass-spring material. Structural, shear, and bending edges apply equal-and-opposite axial spring/damper forces before every Bullet step; they create no internal P2P constraints. Spring energy, maximum force, cap activity, and evaluation counts are recorded.

The explicit network uses two deterministic Bullet internal integration substeps per 240 Hz outer physics step. Spring forces are evaluated once before each outer step; policy/trace timing remains 240 Hz and solver iterations remain frozen at 80. This numerical-stability setting is recorded in coupon and Pair metadata.

## Gated workflow

1. An XArm-free material coupon fixes one face, loads the opposite face, and checks millimetre-scale rigid-aligned deformation, face-relative displacement, structural stability, force-cap activity, and elastic recovery.
2. The first permitted material profile passing coupon gates is frozen. Pair outcomes cannot change its stiffness or damping.
3. Probe-only checks hidden-friction load, patch-node slip reduction, inside/outside displacement gradient, and a formal 10 Hz onset at least one complete policy sample before macro-visible divergence.
4. Full Pair evaluates visible, full-node, rigid-aligned and target-progress branches at `test_end`, plus the median of the final three test policy samples. Post-test state is reported as recovery evidence, not substituted for test-time deformation.

Formal channels remain motor torque, joint reaction wrench, end-effector tracking error, and end-effector contact wrench. Patch contact, slip and local-membership channels remain oracle-only. The public video never displays the patch.

R1 remains a single-Pair audit. It does not authorize State Diff, IDM or CFPM training, multi-seed claims, or paper conclusions. A complete single Pair only permits a separate proposal for 5–10 development seeds.
