# Phase3 Canonicalization Report

Canonicalization is an audit control for the matched-input CCDA premise. It changes only model inputs within paired conditions and never changes future-state targets, action targets, success labels, or condition labels.

## Summary

- Enabled: `True`
- Rule: `group by split_name, visible_seed, window_t; copy free input to paired hidden branches; targets remain condition-specific`
- Canonicalized windows: `50`
- Raw pair count: `50`
- Post pair count: `50`

## Input Differences

| Metric | Raw | Post |
|---|---:|---:|
| mean pair paper_x max abs diff | `0.12923611821956002` | `0.0` |
| max pair paper_x max abs diff | `1.905490756034851` | `0.0` |
| mean pair state_action_x max abs diff | `0.12923611821956002` | `0.0` |
| max pair state_action_x max abs diff | `1.905490756034851` | `0.0` |
