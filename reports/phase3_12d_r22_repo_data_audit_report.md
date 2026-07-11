# Phase3.12d-r2.2 Repository and Data Integrity Audit

- Verdict: `WARN`
- Root cause: `phase312d_r22_repo_data_integrity_passed_with_legacy_warning`
- Collection allowed: `True`
- Windows: `956` rows
- Legacy checkpoint bridge only: `True`

## Dimensions

```json
{
  "N": 956,
  "action_dim": 14,
  "expected_paper_x_dim": 405,
  "expected_state_action_x_dim": 447,
  "n_beads": 24,
  "paper_x_shape": [
    956,
    405
  ],
  "state_action_x_shape": [
    956,
    447
  ],
  "state_dim": 135,
  "tf": 4,
  "th": 3,
  "y_action_shape": [
    956,
    14
  ],
  "y_state_shape": [
    956,
    4,
    135
  ]
}
```

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `source_file_basename_reused_across_split_roots` | {'count': 44, 'interpretation': 'source_file stores an episode basename, not a globally unique path. The train/heldout roots and visible seeds are disjoint; basename reuse is reported but is not itself source leakage.'} |
| `WARN` | `legacy_environment_semantics` | windows lack deferred-arming semantics manifest |
| `WARN` | `paired_input_canonicalization_verified` | {'rule': 'group by split_name, visible_seed, window_t; copy free input to paired hidden branches; targets remain condition-specific', 'sample_modes': {'canonicalized_free_input_exact_targets': 96, 'raw_exact': 32}, 'targets_reconstructed_exactly': True} |
