# Phase3.14b-r2 Preflight

- Verdict: `PASS`
- Root cause: `phase314b_r2_preflight_supported`
- Device: `cuda`
- Train rows: `1000`
- Validation rows: `102`

| Config | Objective | Schedule | Terminal alpha-bar | Amplification |
|---|---|---|---:|---:|
| `epsilon_cosine_cap_0p5` | `epsilon` | `cosine_capped` | 2.73037489e-04 | 60.5186 |
| `sample_cosine` | `sample` | `cosine_original` | 2.42854100e-07 | 2029.2113 |
| `v_prediction_cosine` | `v_prediction` | `cosine_original` | 2.42854100e-07 | 2029.2113 |

- Default 100-step linear schedule is excluded because its terminal signal coefficient is `0.602962`.
- No training, test evaluation, IDM, Phase4, or CPS was run.
