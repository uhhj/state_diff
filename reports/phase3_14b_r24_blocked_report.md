# Phase3.14b-r2.4 Blocked Report

## Verdict

- Verdict: `BLOCKED`
- Root cause: `phase314b_r24_supplied_fake_scheduler_missing_get_velocity`
- Static tests: `38 passed, 1 failed`

## Failure

The supplied test fixture `FakeScheduler` implements `add_noise()` but does
not implement `get_velocity()`. The supplied
`analytic_formula_oracle()` calls the repository's existing
`training_target()`, which correctly requires
`scheduler.get_velocity()` for v-prediction.

```text
AttributeError: 'FakeScheduler' object has no attribute 'get_velocity'
```

Failing test:

```text
tests/test_phase314b_r24_noisy_skip.py::test_analytic_formula_oracle_passes
```

The six supplied files were verified against their documented SHA256 values,
but were not committed because the mandatory static test gate failed. No
threshold was changed and the failing test was not skipped.

## Boundaries

- Validation targets used: `False`
- Formal test read: `False`
- Formal training: `False`
- Candidate selected: `False`
- Checkpoint or weights saved: `False`
- IDM / candidate execution: `False`
- Phase4 / CPS: `False`
- DeformableRavens modified: `False`
