# PB4-C0 Feasibility-Normalized Signed Qpos-Target Family

Verdict: `PB4C0_FEASIBILITY_NORMALIZED_FAMILY_FROZEN`

**Classification:** `CPU_ONLY_ACTION_FAMILY_DERIVATION`

**Formal Condition-5 confirmed:** `false`

**GPU executed:** `false`

## Unit-scale diagnostic

- Unit {-1,0,+1} grid violating commands: 98
- Unit grid generated commands: 360

## Derived global scale

- alpha_max_hard_limits: 0.26808133907545595
- alpha_reference_before_safety_factor: 0.26808133907545595
- feasibility_safety_factor: 0.94999999999999996
- alpha_formal: 0.25467727212168312

## Limiting constraint

- time: 20
- relative step: 20
- arm: 1
- joint: panda_joint2
- limiting bound: lower
- limiting unit-mask sign: -1

## Formal grid validation

- out-of-limit commands: 0
- minimum revolute limit margin: 0.0043752254295348347 rad
- minimum finger limit margin: 0 m

## Boundary

PB4-C0 derives and freezes an engineering-feasible signed qpos-target family only. It does not test control relevance and does not authorize Condition-5 claims.

STOP after committing this result. PB4-C1, if pursued, must be a separate preregistered exploratory GPU phase using the exact frozen alpha_formal.
