# Phase3.14b-r2.5.1 Blocked Report

- Verdict: `BLOCKED`
- Root cause: `phase314b_r251_execution_failed`
- Exit code: `1`
- Stage: `pilot`

## Command

```text
python scripts/phase3_14b_r251_run_pilot.py --root /data/state_diff2 --preflight-report reports/phase3_14b_r251_preflight_summary.json --prior-steps 5000 --residual-steps 8000 --batch-size 64 --prior-learning-rate 1e-3 --residual-learning-rate 1e-3 --evaluation-noises 8 --reverse-samples 16 --calibration-batches 8
```

## Exact exception tail

```text
/miniforge3/envs/coord_bimanual/lib/python3.9/site-packages/wandb/apis/public.py:2997: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import parse_version
Traceback (most recent call last):
  File "/data/state_diff2/scripts/phase3_14b_r251_run_pilot.py", line 479, in <module>
    main()
  File "/data/state_diff2/scripts/phase3_14b_r251_run_pilot.py", line 319, in main
    reverse_metrics_raw = paired_reverse_pool_metrics_with_inversion(
  File "/data/state_diff2/ccda_phase3/phase314b_r251_gradient_calibration.py", line 1088, in paired_reverse_pool_metrics_with_inversion
    metric = nearest_index_metrics(
  File "/data/state_diff2/ccda_phase3/phase314b_r22_geometry.py", line 52, in nearest_index_metrics
    p=ordered_xy(_future(pred))[:,-1]; t=ordered_xy(_future(target))[:,-1]; d=((p[:,:,None]-t[:,None])**2).sum(-1); idx=d.argmin(-1); inv=np.mean(np.diff(idx,axis=1)<0,axis=1); return {'nearest_inversion':inv,'nearest_unique_fraction':np.asarray([len(np.unique(x))/24 for x in idx])}
  File "/data/state_diff2/ccda_phase3/phase314b_r22_geometry.py", line 21, in _future
    if a.ndim!=3 or a.shape[1:]!=(4,87) or not np.isfinite(a).all(): raise ValueError(f'{name} must be finite [N,4,87]')
ValueError: future must be finite [N,4,87]

```

Formal test, formal training, IDM, candidate execution, Phase4 and CPS were not authorized.
