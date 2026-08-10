# PB0-S REV4 Wiring-post Hidden Descriptor Robustness

Verdict: `PB0S_REV4_HIDDEN_DESCRIPTOR_ROBUSTNESS_AUDITED`

Formal PB0 verdict remains: `PB0_WIRING_POST_NO_NATURAL_CCDA_PAIRS`

## Pair decomposition

- Original hidden pairs: 27569
- contact-proxy-only: 27569
- original-wrap-only: 0
- contact + original-wrap: 0
- Original-wrap-supported total: 0

## Contact-proxy threshold robustness

- Symmetric boundary margin median: 0.00021561681303834766
- Symmetric boundary margin p95: 0.0008058430251655516
- Symmetric boundary margin max: 0.0012871349241386004
- Threshold-separation width median: 0.0010602985230354217
- Threshold-separation width max: 0.002635574384868146

The full deadband sensitivity curve is in EVIDENCE.json. No deadband value is promoted to a new gate.

## Shortlist

- Source: `top_by_contact_symmetric_boundary_margin`
- Future divergence used for selection: No
- Robot force/sensor used for selection: No
- Candidate count: 20

## Findings

- `PB0S_REV4_ORIGINAL_HIDDEN_DESCRIPTOR_DECOMPOSED`
- `PB0S_REV4_CONTACT_PROXY_CONTRIBUTES_TO_HIDDEN_PAIR_SET`
- `PB0S_REV4_NO_ORIGINAL_WRAP_SUPPORTED_PAIRS_DETECTED`

## Next action

Do not send contact-proxy-only pairs directly to formal Gate 4. Use the margin-ranked shortlist for a small targeted replay that first checks whether a genuine simulator-native DLO-post contact/coupling signal can be obtained from the pinned DLO-Lab implementation at the candidate timestep. Only physically confirmed hidden-state differences should proceed to snapshot same-action branching. If a stable physical contact readout cannot be obtained, do not promote the 3 mm proxy to ground truth; move to a published task with a cleaner privileged interaction state.
