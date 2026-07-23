#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from ccda_phase3.phase314b_r258_stagev_locked_selection_holdout_evaluation import (
    StageVError, load_json, stable_json_bytes, validate_environment_variables,
    validate_stageu_contract, worker_payload, write_once,
)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--probe', required=True)
    p.add_argument('--repository-head', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); root = Path(a.root).resolve()
    validate_environment_variables()
    contract = load_json(root / 'reports/phase3_14b_r258_stageu_candidate_frontier_contract.json')
    validate_stageu_contract(contract)
    result = worker_payload(root=root, probe_payload=load_json(Path(a.probe)), repository_head=a.repository_head, stageu_contract=contract)
    write_once(Path(a.output), stable_json_bytes(result))
    return 0
if __name__ == '__main__': raise SystemExit(main())
