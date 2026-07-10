# Phase3.12c Matched-Reset Integrity + Paired Selector Re-evaluation Start

- Timestamp: `2026-07-10T12:02:10+08:00`
- Main branch: `Experiment1`
- Main HEAD: `3401fd7164669804511d1df66ac6bd9e8dd00c5c`
- origin/Experiment1: `3401fd7164669804511d1df66ac6bd9e8dd00c5c`
- Submodule: `388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)`

## Main status

```text
?? checkpoints/
?? reports/phase3_12_workers/
?? reports/phase3_12b_workers/
```

## Objective

1. Audit deterministic matched reset across selectors.
2. Block selector comparison unless reset integrity passes.
3. Compare selectors with paired delta_fraction under identical condition + visible_seed.
4. Re-evaluate the two seed blocks used by Phase3.12 and Phase3.12b.
5. No model training.
6. No future DDPM training.
7. No Phase4.
8. No CPS.
