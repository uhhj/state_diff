# Phase3.14b-r2.5.4 BLOCKED

- Failed stage: `pilot`
- Exit code: `1`
- Failed line: `162`
- Pilot summary SHA256: `None`

```text
{"cuda_available": true, "cuda_runtime": "11.2", "gpu": "NVIDIA GeForce RTX 4090", "interpreter": "/miniforge3/envs/coord_bimanual/bin/python3.9", "numpy": "1.23.3", "python": "3.9.15", "torch": "1.12.1.post200"}
phase3_14b_r254
{"output": "/data/state_diff2/reports/phase3_14b_r254_preflight_summary.json", "verdict": "PASS"}
/miniforge3/envs/coord_bimanual/lib/python3.9/site-packages/wandb/apis/public.py:2997: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import parse_version
Traceback (most recent call last):
  File "/data/state_diff2/scripts/phase3_14b_r254_run_pilot.py", line 515, in <module>
    main()
  File "/data/state_diff2/scripts/phase3_14b_r254_run_pilot.py", line 292, in main
    raise RuntimeError("shared paired-prior SHA did not reproduce")
RuntimeError: shared paired-prior SHA did not reproduce
```
