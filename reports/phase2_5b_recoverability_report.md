# Phase2.5b Recoverability Grid Selection

## Verdict

- Verdict: `PASS`
- Selected recoverable condition: `hidden_breakaway_pin`
- Selected recoverable config: `breakaway_force_2p6_disp_0p045_pull_0p36`
- Selected env vars: `{"CCDA_BREAKAWAY_BEAD_RATIO": "0.45", "CCDA_BREAKAWAY_DISP": "0.045", "CCDA_BREAKAWAY_FORCE": "2.6", "CCDA_ORACLE_BREAKAWAY_PULL_DIST": "0.36"}`

## Search Sanity

| Metric | Value |
|---|---:|
| free nominal success | 1.000 |
| free guided_search success | 1.000 |
| search sanity pass | True |

## Audited Conditions

| Config | Condition | Nominal | Oracle | Search | Gap | Class |
|---|---|---:|---:|---:|---:|---|
| breakaway_force_2p6_disp_0p045_pull_0p36 | free | 1.000 | 0.750 | 1.000 | -0.250 | ambiguous_needs_tuning |
| breakaway_force_2p6_disp_0p045_pull_0p36 | hidden_breakaway_pin | 0.000 | 0.500 | 0.250 | 0.500 | recoverable_cps_candidate |
| breakaway_force_2p6_disp_0p045_pull_0p36 | hidden_high_friction | 0.000 | 0.500 | 0.750 | 0.500 | recoverable_cps_candidate |
| breakaway_force_2p6_disp_0p045_pull_0p36 | hidden_pin | 0.000 | 0.000 | 0.000 | 0.000 | impossible_diagnostic |

## Candidate Table

| Config | Condition | Nominal | Oracle | Search | Gap | Class |
|---|---|---:|---:|---:|---:|---|
| breakaway_force_2p6_disp_0p045_pull_0p36 | hidden_breakaway_pin | 0.000 | 0.500 | 0.250 | 0.500 | recoverable_cps_candidate |

## Interpretation

- PASS means Phase3 medium can be configured with the selected recoverable branch.
- WARN means a near candidate exists but is not strong enough for paper-level CPS success improvement.
- FAIL means do not run Phase3 medium.
- `hidden_pin` remains hard diagnostic, not the CPS success-improvement target.

## Selected Environment Variables

- `CCDA_BREAKAWAY_BEAD_RATIO=0.45`
- `CCDA_BREAKAWAY_DISP=0.045`
- `CCDA_BREAKAWAY_FORCE=2.6`
- `CCDA_ORACLE_BREAKAWAY_PULL_DIST=0.36`
