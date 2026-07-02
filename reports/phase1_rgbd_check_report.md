# Phase1 RGB-D Observation Check Report

## Purpose

This report checks whether hidden-contact conditions leak into saved RGB-D observations.
For each visible seed, the first observation under each hidden condition is compared against `free`.

## Aggregate Metrics

| Condition | N | Mean RGB Abs Diff | Max RGB Abs Diff | Mean Depth Abs Diff | Max Depth Abs Diff |
|---|---:|---:|---:|---:|---:|
| free | 5 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| hidden_pin | 5 | 0.000002 | 0.221176 | 0.000000 | 0.009458 |
| hidden_high_friction | 5 | 0.000011 | 0.387451 | 0.000001 | 0.019320 |
| hidden_side_jam | 5 | 0.001262 | 0.770196 | 0.000106 | 1.317826 |

## Interpretation

- Low RGB/depth difference between `free` and hidden conditions supports the claim that hidden contact is not visually leaked.
- Nonzero difference can still occur because the cable may settle slightly differently after hidden contact is applied.
- This report is a Phase1 visual sanity check, not the final CCDA threshold audit.
