# Phase3.3 Ravens Import Probe

- Python: `/miniforge3/envs/coord_bimanual/bin/python3`
- Conda env: `coord_bimanual`
- DeformableRavens path injected: `True`
- all_required_ok: `False`

## Import Results

| Module | OK | Version | Error |
|---|---:|---|---|
| `numpy` | `True` | `1.23.3` | `` |
| `torch` | `True` | `1.12.1.post200` | `` |
| `pybullet` | `True` | `unknown` | `` |
| `meshcat` | `True` | `unknown` | `` |
| `ravens` | `False` | `None` | `ModuleNotFoundError("No module named 'tensorflow'")` |
| `ravens.tasks` | `True` | `unknown` | `` |

## Interpretation

- This probe does not run rollout.
- If `ravens` fails because of missing `meshcat`, install meshcat in `coord_bimanual` only.
- Do not modify torch or numpy.
