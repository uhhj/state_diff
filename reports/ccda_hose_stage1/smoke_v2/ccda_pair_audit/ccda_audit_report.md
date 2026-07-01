# Stage 1.5 CCDA Pair Audit

## Dataset

`data/ccda_hose_stage1/smoke_v2.npz`

## Purpose

This audit checks whether the current MuJoCo hose insertion environment produces the intended CCDA structure:

- small visible-history distance;
- small proprioception distance;
- small action-history distance;
- large hidden-contact distance;
- large future hose-state distance;
- different success or branch outcomes.

## Thresholds

```json
{
  "tau_vis": 0.02,
  "tau_prop": 0.01,
  "tau_act": 0.003,
  "tau_contact": 0.2,
  "tau_future": 0.025
}
```

## Pair summary

| Pair type | N pairs | mean d_vis | mean d_prop | mean d_act | mean d_contact | mean d_future | success diff | branch diff | CCDA ratio | Impact-CCDA ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_vs_A` | 6 | 0.000042 | 0.000000 | 0.000000 | 0.000001 | 0.000136 | 0.000 | 0.000 | 0.000 | 0.000 |
| `B_vs_B` | 6 | 0.000017 | 0.000000 | 0.000000 | 0.000000 | 0.000059 | 0.000 | 0.000 | 0.000 | 0.000 |
| `A_vs_B` | 16 | 0.000031 | 0.000000 | 0.000000 | 0.353553 | 0.000861 | 0.000 | 0.000 | 0.000 | 0.000 |

## Interpretation

The desired pattern is:

| Pair type | Expected pattern |
| --- | --- |
| `A_vs_A` | low d_vis, low d_contact, low d_future, low success_diff |
| `B_vs_B` | low d_vis, low/medium d_contact, medium d_future, low/medium success_diff |
| `A_vs_B` | low d_vis, high d_contact, high d_future, high success_diff |

If `A_vs_B` has high d_contact but low d_future, the environment only proves hidden force difference, not future state branch divergence.
If `A_vs_B` has high d_vis, the audit time leaks the hidden condition visually and the setting is not a clean CCDA test.
