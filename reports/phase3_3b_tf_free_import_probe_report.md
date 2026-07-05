# Phase3.3b TensorFlow-Free Import Probe

## Verdict

- Verdict: `PASS`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Environment module: `ravens.environment`

## Checks

| Check | Result |
|---|---:|
| `python_env_coord_bimanual` | `True` |
| `safe_numpy` | `True` |
| `safe_torch` | `True` |
| `safe_pybullet` | `True` |
| `safe_ravens_tasks` | `True` |
| `safe_environment` | `True` |
| `task_registry_ok` | `True` |
| `hidden_contact_task_registered` | `True` |
| `tensorflow_not_loaded` | `True` |
| `ravens_agents_not_loaded` | `True` |
| `ravens_models_not_loaded` | `True` |
| `ravens_datasets_not_loaded` | `True` |

## Safe Import Results

| Module | OK | Version | Error |
|---|---:|---|---|
| `numpy` | `True` | `1.23.3` | `None` |
| `torch` | `True` | `1.12.1.post200` | `None` |
| `pybullet` | `True` | `unknown` | `None` |
| `meshcat` | `True` | `unknown` | `None` |
| `ravens.tasks` | `True` | `unknown` | `None` |
| `ravens.environment` | `True` | `unknown` | `None` |

## Forbidden Loaded Modules

| Prefix | Loaded modules |
|---|---|
| `tensorflow` | `[]` |
| `ravens.agents` | `[]` |
| `ravens.models` | `[]` |
| `ravens.datasets` | `[]` |

## Interpretation

- This probe intentionally avoids top-level `import ravens`.
- PASS means rollout runtime may not need TensorFlow.
- FAIL blocks rollout smoke.
