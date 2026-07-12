# Phase3.14b-r2 Targeted Training

- Verdict: `PASS` (training completed)
- Root cause: `phase314b_r2_training_completed`
- Device: `cuda`
- Runs: `9`

| Config | Seed | Epoch | Stable | K1 | Best-8 | Physical | t99 |
|---|---:|---:|---|---:|---:|---:|---:|
| `epsilon_cosine_cap_0p5` | 31431 | 750 | `False` | 1.02068499 | 0.39794651 | 0.000000 | 0.76945877 |
| `epsilon_cosine_cap_0p5` | 31432 | 725 | `False` | 0.94501619 | 0.38724312 | 0.000000 | 0.89826117 |
| `epsilon_cosine_cap_0p5` | 31433 | 700 | `False` | 0.99445205 | 0.39547606 | 0.000000 | 0.82249145 |
| `sample_cosine` | 31431 | 25 | `False` | 0.12388412 | 0.12386202 | 0.000000 | 0.16842485 |
| `sample_cosine` | 31432 | 25 | `False` | 0.12337314 | 0.12334792 | 0.000000 | 0.16841674 |
| `sample_cosine` | 31433 | 25 | `False` | 0.12337516 | 0.12335131 | 0.000000 | 0.16767075 |
| `v_prediction_cosine` | 31431 | 25 | `False` | 0.10176105 | 0.07982986 | 0.000000 | 0.16325151 |
| `v_prediction_cosine` | 31432 | 25 | `False` | 0.10141402 | 0.08134168 | 0.000000 | 0.16543911 |
| `v_prediction_cosine` | 31433 | 75 | `False` | 0.10000765 | 0.07881213 | 0.000000 | 0.16417107 |

- Test was not used.
- IDM, action execution, Phase4, and CPS were not run.
