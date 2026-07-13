# Phase3.14b-r2.5.3 Blocked Report

- Verdict: `BLOCKED`
- Root cause: `phase314b_r253_execution_failed`
- Exit code: `1`
- Line: `finalization`
- Pilot summary exists: `True`
- Pilot summary SHA256: `a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86`

## Exact Exception

```text
Traceback (most recent call last):
  File "/data/state_diff2/scripts/phase3_14b_r253_finalize.py", line 499, in <module>
    main()
  File "/data/state_diff2/scripts/phase3_14b_r253_finalize.py", line 411, in main
    raise RuntimeError(f"r2.5.2 one-step reproduction failed: {name}")
RuntimeError: r2.5.2 one-step reproduction failed: v_only_frozen_control

```

Validation/formal-test/formal-training/IDM/candidate execution/Phase4/CPS were not authorized.
