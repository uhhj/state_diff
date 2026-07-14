# Phase3.14b-r2.5.4 Resume1 Shared-Prior Determinism Audit

## Verdict

- Verdict: `PASS` (train-only diagnostic chain completed)
- Root cause: `phase314b_r254_resume1_cross_device_bitwise_sha_contract_overstrict`
- Functional prior contract supported: `true`
- Next: `phase3_14b_r254_resume2_robot_proxy_attribution_functional_prior_contract`
- Train-only recommendation: `None`
- Selected configuration: `None`

## Historical versus current prior

- Historical GPU: `NVIDIA GeForce RTX 4080 SUPER`
- Current GPU: `NVIDIA GeForce RTX 4090`
- Expected historical state SHA256: `083278d6b0710ddc3863d822978973f29842bc757edf73ff03a93bb154f5665d`
- Observed state SHA256 values: `['8610337d4e869764c7315280056f6af9151ccf2d9872fc47d3afc9dca066e904', '8610337d4e869764c7315280056f6af9151ccf2d9872fc47d3afc9dca066e904', '8610337d4e869764c7315280056f6af9151ccf2d9872fc47d3afc9dca066e904']`
- Exact historical SHA matches: `0`
- Same-device state hashes all exact: `true`
- Same-device prediction hashes all exact: `true`
- Same-device functional equivalence: `true`
- Historical functional fingerprint: `true`

## Repeated fits

| Run | State SHA256 | Prediction SHA256 | Prior z MSE |
|---|---|---|---:|
| `repeat_a` | `8610337d4e869764c7315280056f6af9151ccf2d9872fc47d3afc9dca066e904` | `70c3a6bde584ee6973190823e37548b69121ddf7fbc4c0cd5fe069a4d69ae7b9` | 0.175076887 |
| `repeat_b` | `8610337d4e869764c7315280056f6af9151ccf2d9872fc47d3afc9dca066e904` | `70c3a6bde584ee6973190823e37548b69121ddf7fbc4c0cd5fe069a4d69ae7b9` | 0.175076887 |
| `repeat_c` | `8610337d4e869764c7315280056f6af9151ccf2d9872fc47d3afc9dca066e904` | `70c3a6bde584ee6973190823e37548b69121ddf7fbc4c0cd5fe069a4d69ae7b9` | 0.175076887 |

## Same-device pairwise differences

| Pair | State exact | Param max/RMSE | Prediction max/RMSE | z-MSE relative | PASS |
|---|---:|---|---|---:|---:|
| `repeat_a__vs__repeat_b` | true | 0/0 | 0/0 | 0 | true |
| `repeat_a__vs__repeat_c` | true | 0/0 | 0/0 | 0 | true |
| `repeat_b__vs__repeat_c` | true | 0/0 | 0/0 | 0 | true |

## Historical functional fingerprint

- Historical prior z MSE: `0.174935177`
- Current median prior z MSE: `0.175076887`
- Prior z-MSE relative difference: `0.000809416`
- Loss-curve relative error p95: `0.00273202`
- Final-loss relative error: `0.000928887`

## Contract interpretation

- Same-run loaded snapshot integrity remains exact-SHA protected.
- Same-device reruns are judged by exact SHA plus strict parameter/prediction bounds.
- Historical cross-device reproduction is judged by the committed prior functional fingerprint, not parameter bytes alone.
- Robot-proxy attribution was not run or interpreted in this stage.

## Boundaries

- Reverse sampling rerun: `False`
- Validation targets used: `False`
- Formal test read: `False`
- Formal training: `False`
- Checkpoint/model weights saved: `False`
- Formal IDM / candidate execution: `False`
- Phase4 / CPS: `False`

A PASS verdict means only that the shared-prior diagnostic chain completed.
