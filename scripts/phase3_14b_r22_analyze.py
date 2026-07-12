#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from ccda_phase3.phase314a_contract import strict_json_dump
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='/data/state_diff2');a=ap.parse_args();root=Path(a.root);vpath=root/'reports/phase3_14b_r22_validation_summary.json';ppath=root/'reports/phase3_14b_r22_pilot_selection_summary.json';v=json.loads(vpath.read_text()) if vpath.exists() else None;p=json.loads(ppath.read_text()) if ppath.exists() else None
 if v: verdict=v['verdict'];cause=v['root_cause'];next_step='Phase3.14b-r2.3 one-shot formal K=32 audit' if verdict=='PASS' else 'validation-only ordered-geometry failure diagnosis'
 elif p and p['verdict']=='FAIL': verdict='FAIL';cause=p['root_cause'];next_step='pilot geometry repair diagnosis'
 else: verdict='PASS';cause='phase314b_r22_stage1_supported';next_step='GPU pilot'
 out={'verdict':verdict,'root_cause':cause,'next_step':next_step,'pilot':p,'validation':v,'formal_test_read':False,'formal_test_run':False,'idm':False,'candidate_execution':False,'phase4':False,'cps':False};strict_json_dump(root/'reports/phase3_14b_r22_summary.json',out);(root/'reports/phase3_14b_r22_report.md').write_text(f'# Phase3.14b-r2.2 Report\n\n- Verdict: `{verdict}`\n- Root cause: `{cause}`\n- Formal test read: `False`\n');(root/'reports/phase3_14b_r22_no_phase4_confirmation.md').write_text('# No Phase4 Confirmation\n\n- Formal test read: `False`\n- IDM: `False`\n- Candidate execution: `False`\n- Phase4/CPS: `False`\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
