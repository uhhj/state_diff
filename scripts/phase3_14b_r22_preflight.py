#!/usr/bin/env python3
import argparse,json,subprocess,torch,diffusers
from pathlib import Path
from ccda_phase3.phase314a_contract import sha256_file,strict_json_dump
from ccda_phase3.phase314b_contract import CACHE_SHA256
from ccda_phase3.phase314b_r22_contract import *
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',default='/data/state_diff2'); a=ap.parse_args(); root=Path(a.root).resolve(); base=require_base_state(root)
 if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip(): raise RuntimeError('tracked worktree dirty')
 if subprocess.check_output(['git','-C','external/deformable-ravens','status','--porcelain'],cwd=root,text=True).strip(): raise RuntimeError('submodule dirty')
 arrays,manifest,x,xs,tr,fit,cal,val=load_train_validation_rows(root); assert_no_test_access(arrays,tr,val)
 if not bool(np.all(np.asarray(arrays['future_active'])[:,:48])): raise RuntimeError('bead XY inactive')
 payload={'verdict':'PASS','root_cause':'phase314b_r22_preflight_supported','main_commit':base['main_commit'],'submodule_commit':base['submodule_commit'],'cache_sha256':CACHE_SHA256,'cache_manifest_sha256':sha256_file(root/'data/phase3_14_cache/phase3_14a_training_cache_manifest.json'),'torch_version':torch.__version__,'diffusers_version':diffusers.__version__,'cuda_available':torch.cuda.is_available(),'train_rows':len(tr),'fit_rows':len(fit),'calibration_rows':len(cal),'validation_rows':len(val),'validation_pair_keys':len(np.unique(arrays['pair_key'][val].astype(str))),'bead_xy_all_active':True,'source_sha256':source_sha256(root),'formal_test_read':False,'training':False,'idm':False,'candidate_execution':False,'phase4':False,'cps':False}; strict_json_dump(root/'reports/phase3_14b_r22_preflight_summary.json',payload); (root/'reports/phase3_14b_r22_preflight_report.md').write_text('# Phase3.14b-r2.2 Preflight\n\n- Verdict: `PASS`\n- Formal test read: `False`\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
