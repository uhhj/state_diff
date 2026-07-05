# Phase3 Canonicalization Report

Canonicalization is an audit control for the matched-input CCDA premise. It changes only model inputs within paired conditions and never changes future-state targets, action targets, success labels, condition labels, hidden-contact metadata, or recoverability parameters.

## Summary

- Enabled: `True`
- Rule: `group by split_name, visible_seed, window_t; copy free input to paired hidden branches; targets remain condition-specific`
- Conditions: `free, hidden_pin, hidden_high_friction, hidden_breakaway_pin`
- Primary pair: `free_vs_hidden_breakaway_pin`
- Diagnostic pair: `free_vs_hidden_pin`
- Canonicalized windows: `717`
- Raw pair count: `717`
- Post pair count: `717`

## Input Differences By Hidden Condition

| Hidden condition | Raw paper max | Raw state_action max | Post paper max | Post state_action max |
|---|---:|---:|---:|---:|
| `hidden_pin` | `1.942394733428955` | `1.942394733428955` | `0.0` | `0.0` |
| `hidden_high_friction` | `1.9335333108901978` | `1.9335333108901978` | `0.0` | `0.0` |
| `hidden_breakaway_pin` | `1.9388586282730103` | `1.9388586282730103` | `0.0` | `0.0` |
