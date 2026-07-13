# Phase3.14b-r2.4.2 Blocked Report

- Verdict: `BLOCKED`
- Root cause: `phase314b_r242_execution_failed`
- Exit code: `1`
- Line: `95`

```text
python scripts/phase3_14b_r242_run_pilot.py --root "${ROOT}" --preflight-report "${PREFLIGHT_JSON}" --prior-steps 5000 --residual-steps 8000 --paired-residual-steps 8000 --batch-size 64 --prior-learning-rate 1e-3 --residual-learning-rate 1e-3 --decoupled-prior-learning-rate 1e-4 --evaluation-noises 8
```

## Exact Exception

```text
Traceback (most recent call last):
  File "/data/state_diff2/scripts/phase3_14b_r242_run_pilot.py", line 446, in <module>
    main()
  File "/data/state_diff2/scripts/phase3_14b_r242_run_pilot.py", line 261, in main
    direct_width_stability[str(width)] = summarize_seed_stability(runs)
  File "/data/state_diff2/ccda_phase3/phase314b_r242_frozen_prior.py", line 910, in summarize_seed_stability
    normalized = [_prior_seed_run_summary(run) for run in runs]
  File "/data/state_diff2/ccda_phase3/phase314b_r242_frozen_prior.py", line 870, in _prior_seed_run_summary
    raise ValueError(f"prior seed run metrics missing {key}")
ValueError: prior seed run metrics missing ordered_rmse_p95
```

The producer stores this statistic at `metrics.ordered_rmse.p95`, while the
seed-stability consumer expects the nonexistent flat key
`metrics.ordered_rmse_p95`. The three width-512 runs were not serialized into
a pilot summary, so no pass count is claimed. Width 1024, factorized residual,
paired residual, and finalization did not run.

Formal test, formal training, IDM, candidate execution, Phase4 and CPS were not authorized.
