#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from ccda_phase3.phase314b_r258_stagev_locked_selection_holdout_evaluation import (
    BLOCKED_REPORT, SUCCESS_REPORT, blocked_report, run_evaluation, stable_json_bytes,
    validate_environment_variables, validate_repository, write_once,
)

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--root', default='/data/state_diff2'); p.add_argument('--python-bin', default=sys.executable); a=p.parse_args()
    root=Path(a.root).resolve(); repository=None
    if (root/SUCCESS_REPORT).exists() or (root/BLOCKED_REPORT).exists():
        print('BLOCKED: Stage-V output already exists; rerun is forbidden', file=sys.stderr); return 2
    try:
        validate_environment_variables(); repository=validate_repository(root)
        summary=run_evaluation(root=root, repository=repository, python_bin=a.python_bin)
        write_once(root/SUCCESS_REPORT, stable_json_bytes(summary))
        print(json.dumps({'execution_verdict':summary['execution_verdict'],'scientific_status':summary['scientific_status'],'root_cause':summary['root_cause'],'required_next_path':summary['required_next_path'],'failed_timesteps':summary['failed_timesteps'],'scientific_result_sha256':summary['scientific_result_sha256'],'output':str(root/SUCCESS_REPORT)}, sort_keys=True)); return 0
    except BaseException as error:
        payload=blocked_report(repository=repository,error=error)
        try: write_once(root/BLOCKED_REPORT,stable_json_bytes(payload))
        except BaseException as write_error:
            print(f'BLOCKED: {error}; blocked-evidence write failed: {write_error}',file=sys.stderr); return 2
        print(f'BLOCKED: {error}',file=sys.stderr); return 2
if __name__=='__main__': raise SystemExit(main())
