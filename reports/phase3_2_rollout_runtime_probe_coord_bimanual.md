# Phase3.2 Rollout Runtime Probe

## Verdict

- Verdict: `FAIL`
- Learned rollout ready: `False`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`

## Checks

| Check | Result |
|---|---:|
| `imports_torch` | `True` |
| `imports_ravens` | `False` |
| `imports_pybullet` | `True` |
| `imports_numpy` | `True` |
| `windows_loadable` | `True` |
| `checkpoint_files_exist` | `True` |
| `conditions_match` | `True` |
| `primary_matches` | `True` |
| `diagnostic_matches` | `True` |
| `y_action_dim_14` | `True` |
| `codec_dim_14` | `True` |
| `codec_no_camera_config` | `True` |

## Imports

| Package | OK | Version | Error |
|---|---:|---|---|
| `torch` | `True` | `1.12.1.post200` | `None` |
| `ravens` | `False` | `None` | `ModuleNotFoundError("No module named 'meshcat'")` |
| `pybullet` | `True` | `unknown` | `None` |
| `numpy` | `True` | `1.23.3` | `None` |

## Checkpoints

| Baseline | Complete checkpoints | First complete |
|---|---:|---|
| `paper_state` | 6 | `/data/state_diff2/checkpoints/phase3/paper_state/fold_0_seed_0` |
| `state_action` | 6 | `/data/state_diff2/checkpoints/phase3/state_action/fold_0_seed_0` |

## Interpretation

- PASS means this Python environment can attempt learned rollout smoke.
- FAIL means do not run learned rollout in this environment.
- Do not use NumPy fallback to bypass missing torch.
