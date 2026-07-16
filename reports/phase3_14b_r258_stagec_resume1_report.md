# Phase3.14b-r2.5.8 Stage C Resume1

## Commit-outcome recovery

- Original implementation commit: `0692a5ba794e1b7477d56874f7d003ec887eafc1`
- Original `git commit` process status: `141`
- Git transaction completed: `true`
- First test-gate SHA256: `52edc86bcfe6e70d018a2e9d395ed8ecaf3e8c147c95448eefad2f3d1ff9de43`
- First blocked SHA256: `504eaca096f0d22ff09f1c95871eb7a4a5d105bb836a2f4fca08cb8203b586fe`
- First calibration started: `false`
- Original reports modified: `false`
- Recovery namespace: `phase3_14b_r258_stagec_resume1`

## Scientific calibration result
# Phase3.14b-r2.5.8 Stage C

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r258_stagec_valid_endpoint_reachable_but_balanced_local_descent_misaligned`
- Required next path: `CALIBRATE_OBJECTIVE_TRAIN_DISTRIBUTION_DIRECTION_SURROGATE_WITHOUT_HOLDOUT_LEAKAGE`

## Immutable Stage-B binding

- Base evidence commit: `174d4428f835bb7fa76d9bdd497c9ff63452122f`
- Base worker SHA256: `f61f2cb9226904d360a1e9199b6d9434f35b499d4d15cec7048d62e6d4b45d96`
- Base contract SHA256: `2322d276e57deae59135e0170920d6c9b2df3aeab040dbe58daf73db2adde684`
- Base selection SHA256: `ed716f7fca55c2b222f6363908f50f89a981a821751f2bddbb7f24ea9df6a443`

## Portable environment and control

- Compatibility pass: `true`
- Compatibility SHA256: `03fef5cfcfac7933bca0dfc72e67156bb260d50ae7f2f1f6a6375a34c90ad7ec`
- Observation SHA256: `2cbee59380ec632c9372ab2e84309f9f1b971f9c4ab7594e3139084d197df7c3`
- GPU: `NVIDIA GeForce RTX 4080 SUPER`
- Compute capability: `8.9`
- Required-operation dry run: `true`
- Cold CUDA context before control replay: `true`
- Stage-C reference equivalence: `true`

## Objective-train balanced reference

- Fit rows / groups: `638` / `126`
- Fit target SHA256: `db5e699a29a342e09075b824b7532aff234bd24d60909079d871b9ab8f309afa`
- Fit group SHA256: `ba071d0d3dfae273a9b74e64bb4623a6abf8b897e8cbe6ccd85dee29ce2506cf`
- Frozen upper element coverage: `0.9996081504702194`
- Lower q01/q05/q10 element coverage: `0.9890282131661442` / `0.9498262232520104` / `0.8996183726318658`
- IQR anchor element coverage: `0.49853482349734224`
- Selection-holdout target used for fitting: `false`

## Candidate summary

| Candidate | Lower q | Lower weight | Anchor weight | Direction all | Scientific witnesses all | Eligible | Witness radii |
|---|---:|---:|---:|---:|---:|---:|---|
| upper_only_reference | 0.05 | 0 | 0 | historical | historical | false | n/a |
| band_q01_w1 | 0.01 | 1 | 0 | false | false | false | {"10": null, "25": null, "50": null} |
| band_q05_w1 | 0.05 | 1 | 0 | false | false | false | {"10": null, "25": null, "50": null} |
| band_q05_w2 | 0.05 | 2 | 0 | false | false | false | {"10": null, "25": null, "50": null} |
| band_q10_w1 | 0.10 | 1 | 0 | false | false | false | {"10": null, "25": null, "50": null} |
| band_q05_w1_a0p25 | 0.05 | 1 | 0.25 | false | false | false | {"10": null, "25": null, "50": null} |
| band_q05_w1_a1p00 | 0.05 | 1 | 1 | false | false | false | {"10": null, "25": null, "50": null} |

## Selectable candidate details

### `band_q01_w1`

| t | Initial descent cosine | Nonnegative rate | Direction pass | Scientific witness |
|---:|---:|---:|---:|---|
| 10 | 0.0441615 | 0.788136 | false | none |
| 25 | 0.0315101 | 0.779661 | false | none |
| 50 | 0.0292213 | 0.805085 | false | none |

### `band_q05_w1`

| t | Initial descent cosine | Nonnegative rate | Direction pass | Scientific witness |
|---:|---:|---:|---:|---|
| 10 | 0.0377873 | 0.754237 | false | none |
| 25 | 0.0272765 | 0.766949 | false | none |
| 50 | 0.0252259 | 0.800847 | false | none |

### `band_q05_w2`

| t | Initial descent cosine | Nonnegative rate | Direction pass | Scientific witness |
|---:|---:|---:|---:|---|
| 10 | 0.0285535 | 0.720339 | false | none |
| 25 | 0.0207994 | 0.716102 | false | none |
| 50 | 0.0194695 | 0.75 | false | none |

### `band_q10_w1`

| t | Initial descent cosine | Nonnegative rate | Direction pass | Scientific witness |
|---:|---:|---:|---:|---|
| 10 | 0.0367685 | 0.75 | false | none |
| 25 | 0.0265905 | 0.75 | false | none |
| 50 | 0.0246114 | 0.79661 | false | none |

### `band_q05_w1_a0p25`

| t | Initial descent cosine | Nonnegative rate | Direction pass | Scientific witness |
|---:|---:|---:|---:|---|
| 10 | 0.0359381 | 0.737288 | false | none |
| 25 | 0.0259607 | 0.758475 | false | none |
| 50 | 0.0240876 | 0.788136 | false | none |

### `band_q05_w1_a1p00`

| t | Initial descent cosine | Nonnegative rate | Direction pass | Scientific witness |
|---:|---:|---:|---:|---|
| 10 | 0.0329726 | 0.733051 | false | none |
| 25 | 0.0238695 | 0.737288 | false | none |
| 50 | 0.0222766 | 0.788136 | false | none |

## Selection and classification

- Selected configuration: `none`
- Eligible candidates: `none`
- Primary failure locus: `local_descent_misalignment`
- Contract SHA256: `355e3a12e13533b4a6f283344cfda1be20c4d6a4bef2fdd28f2ba03b3397b696`
- Selection SHA256: `613dd19b4611fddd24e5a5a71c29054b8a668a868d40198caccb42c82f4b3f50`

## Boundary

- Lower and anchor boundaries were fit only on the 638-row objective-training population.
- The 236-row holdout target was used only to evaluate objective reachability and direction.
- The upper-only negative control was loaded from immutable Stage-B evidence and was not rerun or selectable.
- No model candidate was trained and the model architecture was unchanged.
- Direct-x0 oracle optimization modified only temporary output tensors.
- The frozen probe was not accessed.
- No full repaired model, reverse sampling, formal training, IDM, candidate execution, DeformableRavens, Phase4, or CPS was run.
- No checkpoint, weights, prediction tensor, oracle tensor, candidate tensor, NPZ, cache, image, or video was persisted.
