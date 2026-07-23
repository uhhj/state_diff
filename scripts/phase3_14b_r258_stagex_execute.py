#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, os, subprocess, tempfile
from pathlib import Path
from ccda_phase3.phase314b_r258_stagex_tail_robust_nested_oof import (
    BLOCKED_REPORT, SUCCESS_REPORT, StageXError, blocked_report, sha256_bytes,
    stable_json_bytes, validate_environment_variables, validate_repository,
    validate_worker_payload, write_once,
)

def child(command, root, label):
    completed = subprocess.run(command, cwd=str(root), env=dict(os.environ), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise StageXError('{} rc={} stdout={!r} stderr={!r}'.format(label, completed.returncode, completed.stdout, completed.stderr))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/state_diff2')
    parser.add_argument('--python-bin', default='/miniforge3/envs/coord_bimanual/bin/python')
    args = parser.parse_args()
    root = Path(args.root).resolve(); repository = None
    try:
        validate_environment_variables(); repository = validate_repository(root)
        with tempfile.TemporaryDirectory(prefix='phase314b_r258_stagex_') as directory:
            temp = Path(directory); probe_path = temp / 'probe.json'; worker_path = temp / 'worker.json'
            child([args.python_bin, str(root/'scripts/phase3_14b_r258_stages_resume2a_worker.py'), '--root', str(root), '--mode', 'environment-probe', '--output', str(probe_path)], root, 'environment probe')
            child([args.python_bin, str(root/'scripts/phase3_14b_r258_stagex_worker.py'), '--root', str(root), '--probe', str(probe_path), '--repository-head', repository['head'], '--stageu-contract', str(root/'reports/phase3_14b_r258_stageu_candidate_frontier_contract.json'), '--output', str(worker_path)], root, 'cold Stage-X worker')
            probe = json.loads(probe_path.read_text(encoding='utf-8')); worker = json.loads(worker_path.read_text(encoding='utf-8')); validate_worker_payload(worker)
            probe_pid = int(probe['probe']['process_id'])
            worker_pid = int(worker['process_id'])
            if probe_pid == worker_pid:
                raise StageXError('probe and worker PIDs are not distinct')
        summary = copy.deepcopy(worker)
        summary['schema'] = 'phase314b_r258_stagex_tail_robust_nested_group_oof_summary_v1'
        summary['repository'] = repository
        summary['process_topology'] = {'environment_probe_process_count': 1, 'cold_science_worker_process_count': 1, 'probe_process_id': probe_pid, 'science_worker_process_id': worker_pid, 'processes_distinct': True, 'temporary_payloads_deleted': True}
        summary.pop('scientific_result_sha256', None); summary['scientific_result_sha256'] = sha256_bytes(stable_json_bytes(summary))
        write_once(root/SUCCESS_REPORT, stable_json_bytes(summary)+b'\n')
        return 0
    except BaseException as error:
        payload = blocked_report(repository, error)
        write_once(root/BLOCKED_REPORT, stable_json_bytes(payload)+b'\n')
        return 2
if __name__ == '__main__':
    raise SystemExit(main())
