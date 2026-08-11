# PB0-T Wiring-post Simulator-Native Contact Confirmation

Verdict: `PB0T_NATIVE_POST_CONTACT_MISMATCH_NOT_CONFIRMED`

## Scope

- Targeted simulation only: Yes
- Large-scale recollection: No
- Model training: No
- Formal PB0 verdict changed: No

## Native signal

- ROD-rigid: `vertices_collision.collided / penetration / geom_idx`
- ROD-ROD hidden post: `rr_constraints.penetration`

## Replay

- Unique targeted rollouts: 12
- Targeted pairs: 20
- Global replay alignment valid: True
- Replay-invalid rollouts: 0
- Confirmed shortlist blocked by alignment failure: False
- Alignment tolerance (engineering only): 5e-05 m

## Native confirmation

- Confirmed at target sample: 0
- Also stable across core t=17..19: 0
- Confirmed shortlist size: 0

## Next action

Do not send REV4 contact-proxy-only pairs to Gate 4. The targeted published-task replay did not find a simulator-native post-contact mismatch between the two current states. Stop treating the 3 mm proxy as ground truth. From a project-efficiency standpoint, move to another published task such as DLO-Lab Wrapping rather than adding more geometric proxy engineering.
