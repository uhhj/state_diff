# Experiment2 Bootstrap

This branch prepares the repository and simulator baseline for the CCDA Experiment2 research line. It does not implement or run a scientific task.

## Provenance

- Source repository: `https://github.com/uhhj/state_diff.git`
- Branch: `Experiment2`
- Branch point: `origin/main` at `6a3c817f70547d3011e61a179880d0430a974da7`
- Creation method: a new branch created directly from the exact branch-point commit, with no merge or cherry-pick from prior experimental branches

## Research contract

- Roadmap: `docs/CCDA_RESEARCH_ROADMAP.md`
- Roadmap SHA-256: `bed5e4f6b05d6f6dae79d1d67ef6bfef5ebd0ceb327d3c5059a35f7f68881dcd`
- Future roadmap changes require a version increment, a dated changelog entry, affected-gate and artifact analysis, and an explicit statement about invalidated evidence.

## Simulator baseline

- Path: `external/deformable-ravens`
- URL: `https://github.com/uhhj/deformable-ravens.git`
- Reviewed ref: `ccda-cable`
- Pinned commit: `633a88752445cf5d6776ed374fdbbdb35f93050c`
- The superproject gitlink is authoritative; the submodule is intentionally checked out detached at the exact commit.

## GPU server

- SSH alias: `ccda-gpu`
- Endpoint: `jq1.9gpu.com:15560`
- Worktree: `/data/state_diff_experiment2`
- Host-key status: `VERIFIED` against fingerprints obtained independently from the cloud-provider console
- Authentication: dedicated Ed25519 key with strict host-key checking
- Clone status: the published `Experiment2` branch and recursive submodule were cloned cleanly and SHA-aligned
- Environment inventory: the server retains the prior `/data/state_diff2` Experiment1 tree, but the default shell currently has no Conda/virtual environment or common simulation packages, and `nvidia-smi` cannot communicate with the driver

No Hidden-Friction Cable task implementation, dataset generation, training, or scientific experiment was performed during this bootstrap.
