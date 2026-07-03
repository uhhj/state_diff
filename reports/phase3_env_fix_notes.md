# Phase3 Runtime Environment Fix Notes

## Repository State

- Main repository: `/data/state_diff2`
- Main branch: `Experiment1`
- Main commit before this fix: `7bd00e4dfbfeb51a60a3111009ded26672ee5ac7`
- DeformableRavens submodule commit: `b95180593d3366b754a4d8a4488d723bc9d61be5`
- Submodule status during this fix: clean

## Environment Probe

### defravens37

```text
python: /root/miniforge3/envs/defravens37/bin/python
torch_missing: ModuleNotFoundError("No module named 'torch'")
ravens: OK
```

Conclusion:

- `defravens37` has DeformableRavens / `ravens`: yes.
- `defravens37` is missing PyTorch: yes.
- `defravens37` remains the data generation environment and optional rollout environment only.

### coord_bimanual

```text
python: /miniforge3/envs/coord_bimanual/bin/python
torch: 1.12.1.post200
cuda: True
ravens_missing_expected: ModuleNotFoundError("No module named 'ravens'")
```

Conclusion:

- `coord_bimanual` has PyTorch: yes.
- `coord_bimanual` CUDA availability is reported by torch as true.
- `coord_bimanual` does not need `ravens` for Phase3 offline train/eval.

## Fix Goal

This fix prevents Phase3 train/eval from silently using NumPy ridge fallback when the user intended PyTorch StateDiff-style baselines. NumPy fallback is now disabled by default and can only be enabled with `--allow_numpy_fallback` or `ALLOW_NUMPY_FALLBACK=1` for explicit pipeline smoke.

The expected runtime split is:

- Generate data: `defravens37`.
- Prepare windows, input leakage check, train, eval, aggregate: `coord_bimanual` with torch.
- Rollout: `defravens37` plus torch, or explicitly marked fallback/runtime-limited smoke.
