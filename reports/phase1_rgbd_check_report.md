# Phase1 RGB-D Observation Check Report

## Purpose

This report checks whether hidden-contact conditions leak into saved RGB-D observations.
For each visible seed, the first observation under each hidden condition is compared against `free`.

## Aggregate Metrics

| Condition | N | Mean RGB Abs Diff | Max RGB Abs Diff | Mean Depth Abs Diff | Max Depth Abs Diff |
|---|---:|---:|---:|---:|---:|
| free | 5 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| hidden_pin | 5 | 0.000002 | 0.202353 | 0.000000 | 0.005531 |
| hidden_high_friction | 5 | 0.000010 | 0.381961 | 0.000001 | 0.020867 |

## Interpretation

- Low RGB/depth difference between `free` and hidden conditions supports the claim that hidden contact is not visually leaked.
- Nonzero difference can still occur because the cable may settle slightly differently after hidden contact is applied.
- This report is a Phase1 visual sanity check, not the final CCDA threshold audit.
