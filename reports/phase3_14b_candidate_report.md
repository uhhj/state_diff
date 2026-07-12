# Phase3.14b DDPM Candidate Support

- Verdict: `FAIL`
- Root cause: `phase314b_candidate_physical_validity_failed`
- Selected model: `mlp_ddpm`
- Input: `paper_state`
- Seed: `31421`
- Test pair keys: `103`
- Branch-eligible queries: `200`

| Gate | Value |
|---|---:|
| `best16_vs_deterministic_ci_low` | `-10.70471022296516` |
| `best16_vs_k1_ci_low` | `14.793536238589333` |
| `both_branch_support_rate_k16` | `0.0` |
| `both_branch_support_rate_k32` | `0.0` |
| `physical_sample_validity_rate_k16` | `0.0` |
| `query_has_valid_candidate_rate_k16` | `0.0` |
| `finite_samples` | `True` |

- IDM and query-local execution were not run.
- Phase4 and CPS remain blocked.
