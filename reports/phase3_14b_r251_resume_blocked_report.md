# Phase3.14b-r2.5.1 Resume1 Blocked Report

- Verdict: `BLOCKED`
- Root cause: `phase314b_r251_execution_failed`
- Resume generation: `1`
- Exit code: `1`
- Stage: `finalize`

## Command

```text
python scripts/phase3_14b_r251_finalize.py --root /data/state_diff2 --preflight-report reports/phase3_14b_r251_resume_preflight_summary.json --pilot-report reports/phase3_14b_r251_pilot_summary.json
```

## Exact exception tail

```text
Traceback (most recent call last):
  File "/data/state_diff2/scripts/phase3_14b_r251_finalize.py", line 436, in <module>
    main()
  File "/data/state_diff2/scripts/phase3_14b_r251_finalize.py", line 386, in main
    raise RuntimeError("unique objective matrix/order changed")
RuntimeError: unique objective matrix/order changed

```

Formal test, formal training, IDM, candidate execution, Phase4 and CPS were not authorized.
