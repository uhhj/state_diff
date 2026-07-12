# Phase3.14b-r2.3 Blocked Report

## Verdict

- Verdict: `BLOCKED`
- Root cause: `phase314b_r23_checkpoint_diagnosis_insufficient_distinct_validation_seeds`
- Last successful gate: `phase314b_r23_preflight_passed`

## Failure

The checkpoint diagnosis requested 64 validation rows through a helper that
requires one distinct visible seed per selected row. The immutable validation
split contains 102 rows but only 51 distinct visible seeds, so the request
cannot be satisfied under the implemented sampling contract.

```text
RuntimeError: not enough distinct visible seeds
```

Failing command:

```bash
python scripts/phase3_14b_r23_checkpoint_diagnosis.py \
  --root /data/state_diff2 \
  --device cuda \
  --validation-rows 64 \
  --trace-rows 16 \
  --gradient-rows 32
```

## Boundary Confirmation

- Formal test read: `False`
- Formal training: `False`
- Tiny-overfit controls run: `False`
- IDM / candidate execution: `False`
- Phase4 / CPS: `False`
- r2.3 checkpoints created: `False`
- DeformableRavens submodule modified: `False`

The diagnosis stopped at the first failed step. No threshold or sample request
was silently reduced, and no later stage was skipped.
