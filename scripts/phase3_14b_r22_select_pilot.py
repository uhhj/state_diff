#!/usr/bin/env python3
import json,argparse
from pathlib import Path
from ccda_phase3.phase314a_contract import strict_json_dump
from ccda_phase3.phase314b_r22_contract import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='/data/state_diff2');a=ap.parse_args();root=Path(a.root);s=json.loads((root/'reports/phase3_14b_r22_pilot_training_summary.json').read_text());eligible=[r for r in s['runs'] if GEOMETRY_CONFIGS[r['geometry_config']].selectable and r['moderate_gate']['pass']]
 if not eligible: verdict='FAIL';cause='phase314b_r22_no_pilot_geometry_repair';selected=None
 else:
  eligible.sort(key=lambda r:(-r['metrics']['calibrated']['sample_validity_rate'],-r['metrics']['calibrated']['segment_score_validity_rate'],-r['metrics']['ordered_support_rate'],r['metrics']['best8_ordered_rmse'],r['metrics']['best8_chamfer'],-r['metrics']['pool_diversity']));selected=eligible[0];verdict='PASS';cause='phase314b_r22_pilot_geometry_repair_selected'
 summary={'verdict':verdict,'root_cause':cause,'selected_config':None if selected is None else selected['geometry_config'],'selected_run':selected,'formal_test_read':False};strict_json_dump(root/'reports/phase3_14b_r22_pilot_selection_summary.json',summary);(root/'reports/phase3_14b_r22_pilot_selection_report.md').write_text(f'# Pilot Selection\n\n- Verdict: `{verdict}`\n- Selected: `{summary["selected_config"]}`\n');
 if selected:
  frozen=load_self_hashed_json(root/'reports/phase3_14b_r22_frozen_contract.json');write_self_hashed_json(root/'reports/phase3_14b_r22_pilot_selection.json',{'artifact_version':'phase3_14b_r22_pilot_selection_v1','selected_config':selected['geometry_config'],'checkpoint_sha256':selected['checkpoint_sha256'],'contract_sha256':frozen['artifact_sha256'],'source_sha256':source_sha256(root),'validation_only':True,'formal_test_read':False})
 print(json.dumps(summary,indent=2));raise SystemExit(0 if selected else 1)
if __name__=='__main__':main()
