# Phase3.1 Next Step Plan

- Phase3.1 verdict: see reports/phase3_1_medium_evidence_report.md
- Phase3 offline medium is retained as leakage-checked DDPM evidence.
- Rollout has not been run.
- Phase4 / CPS has not been run.

## Allowed Next Step

Prepare a small rollout smoke only after user approval.

## Rollout Gate

Before rollout:
- action debug must not be FAIL
- Phase3.1 must not be FAIL
- PHASE3_ALLOW_ROLLOUT=1 must be explicitly set
- rollout should start with smoke scale only
- no paper-level claim until rollout and CPS baselines are separately audited
