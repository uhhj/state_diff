# Phase3.14b-r2.5.8 Stage A Portable Resume1 — Hardware-Portable Conflict Projection

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r258_stagea_conflict_projected_k16_no_geometry_or_fidelity_solution`
- Required next path: `AUDIT_DIRECT_X0_MODEL_PARAMETERIZATION_AND_UPPER_GEOMETRY_TARGET_REACHABILITY`

## Immutable Resume4 binding

- Base evidence commit: `f0a5bec1f89f625e74150e55e2da884413d72eb9`
- Resume4 worker SHA256: `0e87be2ae637ad1dd3680535e40ab8b29f7d7e54bf8c6555817d4da4617cac63`
- Resume4 environment SHA256: `7b51113fb013a7f24716049fe4e11464261af62b0b3c4950cdf454fc21aad260`
- Resume4 selection SHA256: `136d4575add0452759106f4f141837dc78ff72f042eb0d08ea37cb0c96334fb6`
- Resume4 contract SHA256: `cc2cb6f87947cceb070104a7f50a04e1f40d9074612007f4790688bb77111c50`

## Portable compute environment

- Compatibility pass: `true`
- Compatibility SHA256: `03fef5cfcfac7933bca0dfc72e67156bb260d50ae7f2f1f6a6375a34c90ad7ec`
- Hardware observation SHA256: `a20b4f68cbbe7ab1dd6f04d35887e99971706658b6113fd8dfd97d1e9849707e`
- GPU: `NVIDIA GeForce RTX 3090`
- Compute capability: `8.6`
- Total memory bytes: `25296044032`
- Driver observation: `580.119.02`
- Python/NumPy/PyTorch/Torch-CUDA: `3.9.15` / `1.23.3` / `1.12.1.post200` / `11.2`
- Required-operation dry run: `true`
- Dry-run peak reserved bytes: `243269632`
- Cold CUDA context before control replay: `true`

## Frozen Stage-C reference replay

- Reference equivalence pass: `true`
- Byte-exact to reference: `true`
- Reference calibration SHA256: `fb91d29c84cedccf8f72458f08c2f213bbbc542663593094c0ab651701a052f0`
- Observed calibration SHA256: `fb91d29c84cedccf8f72458f08c2f213bbbc542663593094c0ab651701a052f0`
- Reference control SHA256: `dee61b8e31455eeb0e7e0eceae5c1500e3ce57ca8bd42989378d9f76fd779ae7`
- Observed control SHA256: `dee61b8e31455eeb0e7e0eceae5c1500e3ce57ca8bd42989378d9f76fd779ae7`
- Calibration numeric differences: `0`
- Control numeric differences: `0`
- Stage-C nonzero candidates trained: `0`

## Conflict-projection calibration

- Initial model SHA256: `4f1c102d9f39ba59544f5565b4c01fe03e073fca596aefdc822b707cd2f2c7fd`
- Calibration SHA256: `2460d4bbb4faf379c3021b6ed87b517575f1b7e1c33773440d7348f817de27db`
- Block order: `noisy_projection, condition_projection, time_projection, residual_block_0, residual_block_1, residual_block_2, residual_block_3, output_norm, output_head`

| Candidate | Mode | Cutoff | Target ratio | Lambda | Initial pre cosine | Initial post cosine | Initial removed norm |
|---|---|---:|---:|---:|---:|---:|---:|
| global_t50_r0p25 | global | 50 | 0.25 | 0.0185079099326 | 0.502665901 | 0.502665901 | 0 |
| block_t25_r0p25 | blockwise | 25 | 0.25 | 0.0144962388337 | 0.419659574 | 0.419659574 | 0 |
| block_t50_r0p10 | blockwise | 50 | 0.1 | 0.00740316397304 | 0.502665901 | 0.502665901 | 0 |
| block_t50_r0p25 | blockwise | 50 | 0.25 | 0.0185079099326 | 0.502665901 | 0.502665901 | 0 |
| block_t50_r0p50 | blockwise | 50 | 0.5 | 0.0370158198652 | 0.502665901 | 0.502665901 | 0 |

## Train-only candidates

| Candidate | t10 reduction | t25 reduction | t50 reduction | Train ratio | Projection trigger | Removed norm | t10 upper | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global_t50_r0p25 | 0.0281783019 | -0.0836371224 | -0.248819209 | 2.29464081 | 0 | 0 | 0 | false |
| block_t25_r0p25 | -0.272791284 | -0.453733913 | -0.596067091 | 2.43016268 | 0.139375 | 0.000334445392 | 0 | false |
| block_t50_r0p10 | -0.0461205039 | -0.133283667 | -0.21968814 | 1.41295573 | 0.131625 | 0.000517176728 | 0 | false |
| block_t50_r0p25 | -0.0144244901 | -0.123620028 | -0.288856201 | 2.34860129 | 0.396625 | 0.00248021102 | 0 | false |
| block_t50_r0p50 | 0.0376065024 | -0.0879616786 | -0.253919488 | 3.18879741 | 0.434875 | 0.00343179325 | 0 | false |

## Selection and mechanism attribution

- Selection SHA256: `62e2800808620dc48858dec5105f97fca71ac25a5ca987242ae5dc2a9cc71a4f`
- Selected configuration: `none`
- Global negative control: `global_t50_r0p25`
- Best blockwise candidate: `block_t50_r0p50`
- Blockwise t10 advantage over global: `0.009428200496948258`
- Primary failure locus: `joint_tradeoff`

## Boundary

- Resume1 corrected only the preflight output lifecycle: a temporary directory was created and environment.json did not exist before atomic_write_once.
- The original seven portable files and first blocked summary were committed unchanged as failure provenance.
- The model architecture and upper-geometry target were unchanged.
- Only the optimizer gradient-combination rule changed.
- Global projection was a negative control and could not be selected.
- Diffusion gradients were immutable under projection.
- GPU model, UUID, PCI bus, compute capability, driver, and memory were audit observations rather than Resume4 equality gates.
- Frozen Stage-C reference admission used exact discrete semantics and pre-registered numerical tolerances; candidate thresholds were not relaxed.
- Two formal workers on the current rented instance were byte-exact.
- The frozen probe was not accessed.
- No full-874-row repaired model, reverse sampling, formal training, IDM, candidate execution, DeformableRavens, Phase4, or CPS was run.
- No checkpoint, weights, prediction tensor, candidate tensor, NPZ, cache, image, or video was persisted.
