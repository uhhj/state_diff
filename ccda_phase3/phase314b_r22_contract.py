"""Phase3.14b-r2.2 frozen-contract and ordered-geometry repair contract."""
from __future__ import annotations
import hashlib, json, math, os, subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from ccda_phase3.phase314a_contract import sha256_file, strict_json_load
from ccda_phase3.phase314b_contract import CACHE_SHA256, future_standardizer
from ccda_phase3.phase314b_r2_contract import load_r2_inputs, train_indices, validation_indices
from ccda_phase3.phase314b_r21_contract import train_visible_seed_partition

PHASE='phase3_14b_r22'; BASE_MAIN_COMMIT='35edaf841106808a7f34ac6630fdc24bb0db3cef'
SUBMODULE_COMMIT='633a88752445cf5d6776ed374fdbbdb35f93050c'
MODEL_FAMILY='mlp_ddpm'; INPUT_VARIANT='paper_state'; REPAIR_CONFIG_NAME='v_prediction_cosine'
PILOT_SEED=31440; FORMAL_SEEDS=(31441,31442,31443)
R22_SOURCE_PATHS=(
'ccda_phase3/phase314b_r22_contract.py','ccda_phase3/phase314b_r22_geometry.py','ccda_phase3/phase314b_r22_loss.py',
'scripts/phase3_14b_r22_preflight.py','scripts/phase3_14b_r22_freeze_contract.py','scripts/phase3_14b_r22_reference_audit.py',
'scripts/phase3_14b_r22_train.py','scripts/phase3_14b_r22_select_pilot.py','scripts/phase3_14b_r22_select_validation.py',
'scripts/phase3_14b_r22_analyze.py','scripts/phase3_14b_r22_stage1_contract.sh','scripts/phase3_14b_r22_stage2_pilot.sh','scripts/phase3_14b_r22_stage3_formal.sh')

@dataclass(frozen=True)
class GeometryLossConfig:
 name:str; geometry_outer_weight:float; ordered_weight:float; edge_vector_weight:float; segment_length_weight:float; chain_length_weight:float; temporal_edge_weight:float; selectable:bool
 def validate(self):
  vals=(self.geometry_outer_weight,self.ordered_weight,self.edge_vector_weight,self.segment_length_weight,self.chain_length_weight,self.temporal_edge_weight)
  if not all(math.isfinite(v) and v>=0 for v in vals): raise ValueError('geometry weights must be finite and nonnegative')
  if self.selectable and (self.geometry_outer_weight<=0 or sum(vals[1:])<=0): raise ValueError('selectable geometry config must be active')
  if not self.selectable and self.name=='v_baseline_replay' and any(v!=0 for v in vals): raise ValueError('baseline replay weights must be zero')

GEOMETRY_CONFIGS={
'v_baseline_replay':GeometryLossConfig('v_baseline_replay',0,0,0,0,0,0,False),
'edge_length':GeometryLossConfig('edge_length',.30,0,1,1,.10,0,True),
'ordered_edge':GeometryLossConfig('ordered_edge',.30,.10,1,1,.10,0,True),
'ordered_edge_temporal':GeometryLossConfig('ordered_edge_temporal',.30,.10,1,1,.10,.10,True)}
for _c in GEOMETRY_CONFIGS.values(): _c.validate()

def _jsonable(x):
 if isinstance(x,Path): return str(x)
 if isinstance(x,np.ndarray):
  if x.dtype.kind in 'fc' and not np.isfinite(x).all(): raise ValueError('nonfinite array')
  return x.tolist()
 if isinstance(x,np.generic): return x.item()
 if isinstance(x,Mapping): return {str(k):_jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)): return [_jsonable(v) for v in x]
 if isinstance(x,float) and not math.isfinite(x): raise ValueError('nonfinite float')
 if x is None or isinstance(x,(str,int,float,bool)): return x
 raise TypeError(type(x))

def canonical_json_bytes(payload): return (json.dumps(_jsonable(payload),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def payload_sha256(payload): return hashlib.sha256(canonical_json_bytes({k:v for k,v in payload.items() if k!='artifact_sha256'})).hexdigest()
def write_self_hashed_json(path,payload):
 p=dict(payload); p['artifact_sha256']=payload_sha256(p); target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists(): raise RuntimeError(f'refusing to replace {target}')
 tmp=target.with_suffix(target.suffix+'.tmp'); tmp.write_bytes(canonical_json_bytes(p)); os.replace(tmp,target); return p

def load_self_hashed_json(path):
 p=json.loads(Path(path).read_text()); expected=p.get('artifact_sha256'); actual=payload_sha256(p)
 if expected!=actual: raise RuntimeError(f'self hash mismatch: {expected} != {actual}')
 return p

def source_sha256(root,paths=R22_SOURCE_PATHS): return {p:sha256_file(Path(root)/p) for p in paths}
def git_commit(root): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
def is_ancestor(root,base): return subprocess.run(['git','merge-base','--is-ancestor',base,'HEAD'],cwd=root).returncode==0

def require_base_state(root):
 root=Path(root); r2=strict_json_load(root/'reports/phase3_14b_r2_summary.json'); r21=strict_json_load(root/'reports/phase3_14b_r21_summary.json')
 if (r2.get('verdict'),r2.get('root_cause'))!=('FAIL','phase314b_r2_no_stable_configuration'): raise RuntimeError('r2 base mismatch')
 if (r21.get('verdict'),r21.get('root_cause'))!=('PASS','phase314b_r21_contract_miscalibration_and_ordered_geometry_failure_supported'): raise RuntimeError('r21 base mismatch')
 if not is_ancestor(root,BASE_MAIN_COMMIT): raise RuntimeError('base commit is not ancestor')
 sub=subprocess.check_output(['git','-C','external/deformable-ravens','rev-parse','HEAD'],cwd=root,text=True).strip()
 if sub!=SUBMODULE_COMMIT: raise RuntimeError('submodule mismatch')
 return {'r2':r2,'r21':r21,'main_commit':git_commit(root),'submodule_commit':sub}

def load_train_validation_rows(root):
 arrays,manifest,x_raw,x_std=load_r2_inputs(Path(root)); tr=train_indices(arrays); va=validation_indices(arrays)
 if not np.all(np.asarray(arrays['split_name'][tr]).astype(str)=='train'): raise RuntimeError('nontrain row')
 if not np.all(np.asarray(arrays['split_name'][va]).astype(str)=='val'): raise RuntimeError('nonval row')
 fit,cal=train_visible_seed_partition(arrays,tr)
 if set(np.asarray(arrays['visible_seed'][fit]).tolist()) & set(np.asarray(arrays['visible_seed'][cal]).tolist()): raise RuntimeError('seed leakage')
 if set(np.asarray(arrays['visible_seed'][tr]).tolist()) & set(np.asarray(arrays['visible_seed'][va]).tolist()): raise RuntimeError('train/val leakage')
 return arrays,manifest,x_raw,x_std,tr,fit,cal,va

def geometry_warmup(epoch): return min(max((int(epoch)+1)/100.0,0.0),1.0)
def strict_checkpoint_save(path,payload):
 import torch
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): raise RuntimeError(f'refusing to replace {p}')
 tmp=p.with_suffix(p.suffix+'.tmp'); torch.save(dict(payload),tmp); os.replace(tmp,p)
def assert_no_test_access(arrays,train_rows,val_rows):
 if np.any(np.asarray(arrays['split_name'][train_rows]).astype(str)!='train') or np.any(np.asarray(arrays['split_name'][val_rows]).astype(str)!='val'): raise RuntimeError('test/noncontract rows supplied')
 for p in R22_SOURCE_PATHS:
  text=Path(p).read_text()
  forbidden='test_indices'+'_and_pairs'
  if forbidden in text: raise RuntimeError(f'forbidden test loader token in {p}')
