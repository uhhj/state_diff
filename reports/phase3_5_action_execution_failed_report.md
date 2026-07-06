# Phase3.5 Action Execution Diagnostic Failed

- Timestamp: 2026-07-06T11:33:48+08:00
- Phase3.5 completed diagnostic execution, but the diagnostic verdict is FAIL.
- Blocking root cause: action_codec_or_rollout_adapter_blocker.
- Oracle actions produced progress and limited success, but decoded ground-truth y_action replay produced no progress.
- No Phase4 was run.
- No CPS was run.
- Do not use this as learned policy execution evidence.

## Existing outputs

~~~text
-rw-r--r-- 1 root root  40K  7月  6 11:14 reports/phase3_5_action_codec_decode_samples.csv
-rw-r--r-- 1 root root  290  7月  6 11:14 reports/phase3_5_action_codec_inspect_report.md
-rw-r--r-- 1 root root  416  7月  6 11:14 reports/phase3_5_action_codec_inspect_summary.json
-rw-r--r-- 1 root root 5.0K  7月  6 11:33 reports/phase3_5_action_execution_diagnostic_report.md
-rw-r--r-- 1 root root  15K  7月  6 11:33 reports/phase3_5_action_execution_diagnostic_summary.json
-rw-r--r-- 1 root root    6  7月  6 11:13 reports/phase3_5_action_execution_diag.pid
-rw-r--r-- 1 root root 1.4K  7月  6 11:14 reports/phase3_5_action_execution_preflight_report.md
-rw-r--r-- 1 root root 4.1K  7月  6 11:14 reports/phase3_5_action_execution_preflight_summary.json
-rw-r--r-- 1 root root  435  7月  6 11:33 reports/phase3_5_action_execution_raw_summary.json
-rw-r--r-- 1 root root  22K  7月  6 11:33 reports/phase3_5_action_execution_run.log
-rw-r--r-- 1 root root  620  7月  6 11:14 reports/phase3_5_action_execution_runtime_config.md
-rw-r--r-- 1 root root  741  7月  6 11:05 reports/phase3_5_action_execution_start_audit.md
-rw-r--r-- 1 root root 369K  7月  6 11:33 reports/phase3_5_action_execution_trials.csv
~~~
