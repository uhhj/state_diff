# DeformableRavens Submodule Setup

## Purpose

DeformableRavens is used as the external simulation environment for CCDA cable experiments. It is tracked as a Git submodule instead of being copied directly into the state_diff repository.

## StateDiff

- Repository path: `/data/state_diff2`
- Branch: `Experiment1`
- Commit before this setup: `b3fc2f1433bf4acd5b41010da803add365d1e108`

## Submodule

- Path: `external/deformable-ravens`
- Remote: `https://github.com/uhhj/deformable-ravens.git`
- Branch: `ccda-cable`
- Commit: `73982748bdfe756e04553e28698948facccecb41`

## Phase0 Status

The original DeformableRavens `cable-line-notarget` task has already passed the Phase0 smoke test locally:

- Data demos: 10
- Goal episodes: 20
- Hidden contact: not included in Phase0

Generated data and goals are intentionally not committed.

## Usage

After cloning state_diff, initialize the submodule with:

```bash
git submodule update --init --recursive
```

Or clone with:

```bash
git clone --recurse-submodules https://github.com/uhhj/state_diff.git
```

## Development Rules

- Modify DeformableRavens task code inside `external/deformable-ravens`.
- Commit DeformableRavens changes to `uhhj/deformable-ravens`, branch `ccda-cable`.
- Then return to `state_diff` and commit the updated submodule pointer.
- Do not commit generated data, goals, videos, checkpoints, or cache files.
