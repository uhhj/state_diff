import numpy as np,pytest
from ccda_phase3.phase314b_r22_contract import *
def test_self_hash(tmp_path):
 p=tmp_path/'x.json';d=write_self_hashed_json(p,{'x':1});assert load_self_hashed_json(p)['x']==1;p.write_text(p.read_text().replace('1','2',1));
 with pytest.raises(RuntimeError):load_self_hashed_json(p)
def test_configs():
 GEOMETRY_CONFIGS['v_baseline_replay'].validate();GEOMETRY_CONFIGS['edge_length'].validate()
 with pytest.raises(ValueError):GeometryLossConfig('x',1,0,0,0,0,0,True).validate()
def test_warmup():assert geometry_warmup(-1)==0 and geometry_warmup(99)==1
