# Phase3.14b-r2.4.2 Blocked Report

- Verdict: `BLOCKED`
- Root cause: `phase314b_r242_execution_failed`
- Exit code: `1`
- Line: `108`

```text
python scripts/phase3_14b_r242_run_pilot.py --root "${ROOT}" --preflight-report "${PREFLIGHT_JSON}" --prior-steps 5000 --residual-steps 8000 --paired-residual-steps 8000 --batch-size 64 --prior-learning-rate 1e-3 --residual-learning-rate 1e-3 --decoupled-prior-learning-rate 1e-4 --evaluation-noises 8
```

## Exact Exception

```text
Traceback (most recent call last):
  File "/data/state_diff2/scripts/phase3_14b_r242_run_pilot.py", line 464, in <module>
    main()
  File "/data/state_diff2/scripts/phase3_14b_r242_run_pilot.py", line 289, in main
    unique_results_runtime[variant.name] = train_factorized_variant(
  File "/data/state_diff2/ccda_phase3/phase314b_r242_frozen_prior.py", line 1321, in train_factorized_variant
    prior_after_warmup["aggregate"]["metrics"]["z_mse"]
KeyError: 'aggregate'
```

The direct-prior width-512 and width-1024 three-seed calculations completed in
memory, but no pilot summary was serialized, so no pass counts are claimed.
The first unique-free factorized variant then failed while computing prior
drift: `_prior_metrics()` emits `prior_after_warmup.metrics.z_mse`, while this
consumer expects the obsolete nested path
`prior_after_warmup.aggregate.metrics.z_mse`. Remaining unique-free variants,
paired low/mid variants, high-noise diagnostics, and finalization did not run.

Formal test, formal training, IDM, candidate execution, Phase4 and CPS were not authorized.
