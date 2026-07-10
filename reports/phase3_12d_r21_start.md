# Phase3.12d-r2.1 Start

- Main branch: `Experiment1`
- Main HEAD: `faf8df5b7cdf5105c8b01c546e0eb7d6c60f11de`
- Submodule: `e5525384af4af5bd0a02c3d1fd33ae84323d44e2`
- Conda env: `coord_bimanual`

## Objective

Correct the Phase3.12d-r2 post-arm environment comparator:

- historical comparison: hidden(t=N) versus free(t=0)
- corrected comparison: hidden(t=N) versus free(t=N)
- add free replicate and hidden-unarmed controls
- keep absolute no-action drift as diagnostic only
- do not modify physical task semantics
- run query-local candidate oracle only after corrected audit passes
- no training, Phase4, or CPS

## Preserved Work

Pre-existing untracked runtime artifacts and the two older `ccda_phase3` edits
are preserved and excluded from this r2.1 change.
