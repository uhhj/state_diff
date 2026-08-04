# Experiment2 Bootstrap Summary

- Implementation status: **PASS** for the local repository contract and exact simulator pin.
- Audit status: **PASS** for local integrity, content hashes, submodule cleanliness, conservative secret scanning, and GitHub publication.
- Server bootstrap status: **PASS**. Independently verified host keys, dedicated key authentication, and the clean `/data/state_diff_experiment2` clone were validated.
- Scientific status: **NOT_STARTED**. No task implementation, data generation, training, or experiment was run.

## Provenance

- Branch point: `6a3c817f70547d3011e61a179880d0430a974da7`
- Implementation commit: `14ec3f00bd44326d4a4a9f865521bfbed327e45d`
- Remote operations commit: `b172fc76b46ca344710b8cac3808335804860e3b`
- Audit normalization commit: `6f44bfe9b15cb14422c9d3083b80c6f3e62e5148`
- Evidence commit: the commit containing this report (`SELF`)
- Submodule commit: `633a88752445cf5d6776ed374fdbbdb35f93050c`

## Environment inventory and unresolved risk

The existing `defravens37` and `coord_bimanual` environments activate successfully from `/miniforge3/envs`. Their package cache and reproducibility manifests are stored under `/data`. DeformableRavens imports and `ccda-slack-cable-v2` registration pass. The remaining operational risk is that `nvidia-smi` cannot communicate with the NVIDIA driver and both TensorFlow and PyTorch report no usable GPU.

## Next allowed action

Repair or attach the GPU driver/device before GPU-dependent work. After that, the roadmap permits freezing the Hidden-Friction Cable task definition and beginning Phase 0 / Stage M0 physical and data audit. Experiment1 remains untouched.
