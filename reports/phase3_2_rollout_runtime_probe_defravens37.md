# Phase3.2 Rollout Runtime Probe

## Verdict

- Verdict: `FAIL`
- Learned rollout ready: `False`
- Python: `/root/miniforge3/envs/defravens37/bin/python`
- Conda env: `defravens37`

## Checks

| Check | Result |
|---|---:|
| `imports_torch` | `False` |
| `imports_ravens` | `True` |
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
| `torch` | `False` | `None` | `ModuleNotFoundError("No module named 'torch'")` |
| `ravens` | `True` | `unknown` | `None` |
| `pybullet` | `True` | `unknown` | `None` |
| `numpy` | `True` | `1.19.5` | `None` |

## Checkpoints

| Baseline | Complete checkpoints | First complete |
|---|---:|---|
| `paper_state` | 6 | `/data/state_diff2/checkpoints/phase3/paper_state/fold_0_seed_0` |
| `state_action` | 6 | `/data/state_diff2/checkpoints/phase3/state_action/fold_0_seed_0` |

## Interpretation

- PASS means this Python environment can attempt learned rollout smoke.
- FAIL means do not run learned rollout in this environment.
- Do not use NumPy fallback to bypass missing torch.
