#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from ccda_phase3.phase314a_contract import strict_json_dump
from ccda_phase3.phase314b_r22_contract import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='/data/state_diff2');a=ap.parse_args();root=Path(a.root);s=json.loads((root/'reports/phase3_14b_r22_formal_training_summary.json').read_text());stable=[r for r in s['runs'] if r['stable']];ok=len(stable)>=2;cause='phase314b_r22_ordered_geometry_validation_supported' if ok else 'phase314b_r22_ordered_geometry_repair_failed';selected=min(stable,key=lambda r:r['metrics']['best8_ordered_rmse']) if stable else None;summary={'verdict':'PASS' if ok else 'FAIL','root_cause':cause,'stable_seeds':[r['training_seed'] for r in stable],'stable_seed_count':len(stable),'selected':selected,'formal_test_read':False,'formal_test_run':False};strict_json_dump(root/'reports/phase3_14b_r22_validation_summary.json',summary);(root/'reports/phase3_14b_r22_validation_report.md').write_text(f'# Formal Validation\n\n- Verdict: `{summary["verdict"]}`\n- Stable seeds: `{len(stable)}/3`\n- Formal test read: `False`\n');
 if selected: write_self_hashed_json(root/'reports/phase3_14b_r22_validation_selection.json',{'artifact_version':'phase3_14b_r22_validation_selection_v1','root_cause':cause,'selected':selected,'formal_test_read':False})
 print(json.dumps(summary,indent=2));raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
