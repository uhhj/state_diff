# Phase3 Pre-Medium Audit Report

## Verdict

- Verdict: `WARN`
- Train visible seeds: `100`
- Heldout visible seeds: `30`
- Train/Heldout seed overlap: `[]`
- Complete condition groups: `True`
- Missing branch refs: `0`
- Input leakage pass: `True`
- Action-history target exact match rate: `0.0`
- Prediction rows: `2256`

## Feature Schema

- Feature schema present: `True`
- forbidden_not_in_x: `['breakaway_max_disp_seen', 'breakaway_release_step', 'breakaway_released', 'breakaway_step', 'breakaway_threshold', 'ccda_pair_group', 'ccda_visible_seed', 'condition_id', 'condition_id_onehot', 'condition_label', 'condition_name', 'final_fraction', 'hidden_condition', 'hidden_contact_meta', 'pair_group', 'recoverability_class_candidate', 'recoverability_params', 'source_file', 'success', 'visible_seed']`

## Prediction Metadata Counts

### training_backend

| Value | Count |
|---|---:|
| `torch` | 2256 |

### future_model_type

| Value | Count |
|---|---:|
| `torch_conditional_ddpm_future_state` | 2256 |

### ddpm_used

| Value | Count |
|---|---:|
| `true` | 2256 |

### branch_reference_mode

| Value | Count |
|---|---:|
| `split_visible_seed_window_t` | 2256 |

### scheduler_type

| Value | Count |
|---|---:|
| `diffusers.DDPMScheduler` | 2256 |

### beta_schedule

| Value | Count |
|---|---:|
| `squaredcos_cap_v2` | 2256 |

### prediction_type

| Value | Count |
|---|---:|
| `epsilon` | 2256 |

### variance_type

| Value | Count |
|---|---:|
| `fixed_small` | 2256 |

### denoiser_arch

| Value | Count |
|---|---:|
| `mlp` | 2256 |

### conditional_unet1d_used

| Value | Count |
|---|---:|
| `false` | 2256 |

### paper_alignment_level

| Value | Count |
|---|---:|
| `ddpm_scheduler_aligned_mlp_denoiser` | 2256 |

### eval_sample_seed_mode

| Value | Count |
|---|---:|
| `paired_shared_visible_seed` | 2256 |

## Canonicalization

- raw_max_pair_paper_x_max_abs_diff: `1.942394733428955`
- post_max_pair_paper_x_max_abs_diff: `0.0`
- raw_max_pair_state_action_x_max_abs_diff: `1.942394733428955`
- post_max_pair_state_action_x_max_abs_diff: `0.0`

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `variance_fixed_small_not_learned_range` | fixed_small is acceptable before medium, but not exact learned_range alignment. |
| `WARN` | `mlp_denoiser_not_conditional_unet1d` | Scheduler is aligned, but denoiser is still MLP, not original ConditionalUnet1D. |
| `WARN` | `canonicalization_strong_intervention` | raw_max_pair_paper_x_max_abs_diff=1.942394733428955; acceptable as matched-input audit control. |

## Interpretation

- FAIL blocks MODE=medium.
- WARN is allowed before medium only if explicitly discussed.
- `fixed_small` variance and `mlp` denoiser are acceptable for scheduler-aligned medium, but not exact ConditionalUnet1D / learned_range reproduction.
