# Phase3.14b-r2.5.7 Stage C Top-K Quadratic Calibration

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r257_stagec_topk_quadratic_fidelity_tradeoff`
- Required next path: `CALIBRATE_TIMESTEP_GATED_TOPK_QUADRATIC_OBJECTIVE`

## Immutable Stage-B binding

- Stage-B worker identity SHA256: `17bfea56e69b46946af40ffc590ea83b42fd360b598d99649aec9339b9ab274c`
- Stage-B mechanism contract SHA256: `829082531e2b52298bf2bd2a5b5ccb32509c5d4b8bd430c837aa610a95a4ec22`
- Stage-B mechanism contract file SHA256: `fb5eddb8fbdae36fee095ee204d46036039b2c8188849f9521a1cec835188ae2`

## Gradient-balanced contract

- Contract SHA256: `ccf8db5daa183b871521f1e3237192400ad04e7cb1058c5e5cac1d5b5d180deb`
- Initial model SHA256: `4f1c102d9f39ba59544f5565b4c01fe03e073fca596aefdc822b707cd2f2c7fd`
- Lambda calibration SHA256: `fb91d29c84cedccf8f72458f08c2f213bbbc542663593094c0ab651701a052f0`
- Top-k variants: `8, 16`
- Target gradient ratios: `0.25, 0.5, 1`
- Frozen probe accessed: `false`
- Reverse sampling run: `false`

## Train-only candidates

| Candidate | K | Target ratio | Lambda | t10 excess reduction | t10 upper | Train-control ratio | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | — | 0 | 0 | 0 | 0 | 1 | false |
| topk8_r0p25 | 8 | 0.25 | 0.0201291935217 | 0.0176507653 | 0 | 2.10796967 | false |
| topk8_r0p50 | 8 | 0.5 | 0.0402583870433 | 0.0476292032 | 0 | 3.05929208 | false |
| topk8_r1p00 | 8 | 1 | 0.0805167740866 | 0.132150722 | 0 | 3.95087077 | false |
| topk16_r0p25 | 16 | 0.25 | 0.0229781516729 | 0.120638009 | 0 | 1.81463653 | false |
| topk16_r0p50 | 16 | 0.5 | 0.0459563033458 | 0.189942971 | 0 | 2.587402 | false |
| topk16_r1p00 | 16 | 1 | 0.0919126066917 | 0.234729841 | 0 | 3.68905551 | false |

## Selection

- Selection SHA256: `858d8ca83e1bba9740829546878b7d5b2e91344a2ecefe2aea8100aecc307464`
- Selected configuration: `none`
- Best continuous-response candidate: `topk16_r1p00`
- Best t10 mean-excess reduction: `0.234729841`
- Best t10 binary upper row-any: `0`

## Classification

- Primary failure locus: `fidelity_tradeoff`

## Boundary

- The Stage-B evidence and mechanism contract were immutable.
- One exact diffusion-only control and six gradient-balanced top-k quadratic candidates were trained on the Stage-A objective-training split.
- Candidate lambda values were derived from the common initial-model gradient norms; Stage-A Huber lambda values were not reused.
- Selection used only the 236-row grouped train-only holdout.
- The frozen 126-row probe was not accessed.
- No full-874-row repaired model, reverse sampling, formal pilot, IDM, data collection, candidate execution, DeformableRavens, Phase4, or CPS was run.
- No checkpoint, weights, tensor, NPZ, cache, image, or video was persisted.
