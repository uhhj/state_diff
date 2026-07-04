# Phase3 Pre-Medium Audit Report

## Verdict

- Verdict: `WARN`
- Train visible seeds: `10`
- Heldout visible seeds: `6`
- Train/Heldout seed overlap: `[]`
- Complete condition groups: `True`
- Missing branch refs: `0`
- Input leakage pass: `True`
- Action-history target exact match rate: `0.0`
- Prediction rows: `84`

## Feature Schema

- Feature schema present: `True`
- forbidden_not_in_x: `['ccda_pair_group', 'condition_id', 'condition_name', 'final_fraction', 'hidden_condition', 'hidden_contact_meta', 'source_file', 'success']`

## Prediction Metadata Counts

### training_backend

| Value | Count |
|---|---:|
| `torch` | 84 |

### future_model_type

| Value | Count |
|---|---:|
| `torch_conditional_ddpm_future_state` | 84 |

### ddpm_used

| Value | Count |
|---|---:|
| `true` | 84 |

### branch_reference_mode

| Value | Count |
|---|---:|
| `split_visible_seed_window_t` | 84 |

### scheduler_type

| Value | Count |
|---|---:|
| `diffusers.DDPMScheduler` | 84 |

### beta_schedule

| Value | Count |
|---|---:|
| `squaredcos_cap_v2` | 84 |

### prediction_type

| Value | Count |
|---|---:|
| `epsilon` | 84 |

### variance_type

| Value | Count |
|---|---:|
| `fixed_small` | 84 |

### denoiser_arch

| Value | Count |
|---|---:|
| `mlp` | 84 |

### conditional_unet1d_used

| Value | Count |
|---|---:|
| `false` | 84 |

### paper_alignment_level

| Value | Count |
|---|---:|
| `ddpm_scheduler_aligned_mlp_denoiser` | 84 |

### eval_sample_seed_mode

| Value | Count |
|---|---:|
| `paired_shared_visible_seed` | 84 |

## Canonicalization

- raw_max_pair_paper_x_max_abs_diff: `1.905490756034851`
- post_max_pair_paper_x_max_abs_diff: `0.0`
- raw_max_pair_state_action_x_max_abs_diff: `1.905490756034851`
- post_max_pair_state_action_x_max_abs_diff: `0.0`

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `variance_fixed_small_not_learned_range` | fixed_small is acceptable before medium, but not exact learned_range alignment. |
| `WARN` | `mlp_denoiser_not_conditional_unet1d` | Scheduler is aligned, but denoiser is still MLP, not original ConditionalUnet1D. |
| `WARN` | `canonicalization_strong_intervention` | raw_max_pair_paper_x_max_abs_diff=1.905490756034851; acceptable as matched-input audit control. |

## Interpretation

- FAIL blocks MODE=medium.
- WARN is allowed before medium only if explicitly discussed.
- `fixed_small` variance and `mlp` denoiser are acceptable for scheduler-aligned medium, but not exact ConditionalUnet1D / learned_range reproduction.
