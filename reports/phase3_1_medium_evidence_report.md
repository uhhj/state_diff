# Phase3.1 Medium Evidence Audit

## Verdict

- Verdict: `WARN`
- Recommendation: No FAIL. Rollout smoke may be prepared, but only after user approval.
- Conditions: `free hidden_pin hidden_high_friction hidden_breakaway_pin`
- Primary pair: `free_vs_hidden_breakaway_pin`
- Diagnostic pair: `free_vs_hidden_pin`

## Gate Checks

| Check | Result |
|---|---|
| `conditions_match_required` | `True` |
| `all_torch` | `True` |
| `prediction_primary_recorded` | `True` |
| `primary_pair_correct` | `True` |
| `leakage_pass` | `True` |
| `integration_pass` | `True` |
| `canonical_post_zero` | `True` |
| `forbidden_not_in_explicit_feature_schema` | `True` |
| `y_action_dim_14` | `True` |
| `action_history_no_exact_target_leak` | `True` |
| `action_debug_not_fail` | `True` |
| `sanity_not_fail` | `True` |

## Primary Pair Statistics

| Baseline | Condition | Count | Wrong-Branch | Branch-Accuracy | p_free | p_primary | Future Error | Action MSE | Action OOD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_state` | `free` | 282 | 0.667941 | 0.332059 | 0.332059 | 0.667941 | 0.0823528 | 0.0045433 | 0.549511 |
| `paper_state` | `hidden_breakaway_pin` | 282 | 0.332059 | 0.667941 | 0.332059 | 0.667941 | 0.081501 | 0.00481076 | 0.549511 |
| `state_action` | `free` | 282 | 0.661181 | 0.338819 | 0.338819 | 0.661181 | 0.0818042 | 0.00447397 | 0.549975 |
| `state_action` | `hidden_breakaway_pin` | 282 | 0.338819 | 0.661181 | 0.338819 | 0.661181 | 0.0810252 | 0.00474882 | 0.549975 |

## Paired Deltas

| Baseline | Count | Delta wrong hidden-free | 95% CI | Delta future hidden-free | 95% CI | Delta action MSE hidden-free |
|---|---:|---:|---|---:|---|---:|
| `paper_state` | 282 | -0.335882 | [-0.422883, -0.249997] | -0.000851795 | [-0.00123123, -0.000489883] | 0.00026746 |
| `state_action` | 282 | -0.322363 | [-0.409259, -0.237979] | -0.000778982 | [-0.00116141, -0.000387998] | 0.000274849 |

## Diagnostic Hidden Pin

| Baseline | Free future error | Hidden pin future error | Hidden pin - free |
|---|---:|---:|---:|
| `paper_state` | 0.0823528 | 0.206809 | 0.124457 |
| `state_action` | 0.0818042 | 0.205365 | 0.12356 |

## Warnings And Failures

| Level | Name | Detail |
|---|---|---|
| `WARN` | `canonicalization_strong_intervention` | raw max pair paper_x diff=1.942394733428955 |
| `WARN` | `low_primary_future_error_contrast` | paper_state: hidden-free future error delta=-0.000852 |
| `WARN` | `low_primary_future_error_contrast` | state_action: hidden-free future error delta=-0.000779 |
| `WARN` | `paper_state_state_action_nearly_identical` | free: abs future error diff=0.000548608 |
| `WARN` | `paper_state_state_action_nearly_identical` | hidden_high_friction: abs future error diff=0.000490491 |
| `WARN` | `paper_state_state_action_nearly_identical` | hidden_breakaway_pin: abs future error diff=0.000475794 |
| `WARN` | `many_near_zero_action_dims` | 8/14 dims std < 1e-8 |
| `WARN` | `idm_ood_max_high` | 3.5273444652557373 |
| `WARN` | `action_debug_warn` | See phase3_action_idm_debug_report.md |
| `WARN` | `phase3_sanity_warn` | See phase3_sanity_check_report.md |

## Interpretation

- No rollout, Phase4, or CPS was run by this audit.
- `WARN` items are documented evidence-review items, not automatic FAILs.
- Rollout smoke may be prepared only after user approval and explicit `PHASE3_ALLOW_ROLLOUT=1`.
