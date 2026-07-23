#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from ccda_phase3.phase314b_r258_stagex_tail_robust_nested_oof import run_nested_oof, stable_json_bytes

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--repository-head', required=True)
    parser.add_argument('--stageu-contract', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    probe = json.loads(Path(args.probe).read_text(encoding='utf-8'))
    contract = json.loads(Path(args.stageu_contract).read_text(encoding='utf-8'))
    payload = run_nested_oof(root=Path(args.root), probe_payload=probe, repository_head=args.repository_head, stageu_contract=contract)
    Path(args.output).write_bytes(stable_json_bytes(payload) + b'\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
