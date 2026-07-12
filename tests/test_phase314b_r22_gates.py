from ccda_phase3.phase314b_r22_contract import GEOMETRY_CONFIGS
def test_baseline_not_selectable():assert not GEOMETRY_CONFIGS['v_baseline_replay'].selectable
def test_selectable_exist():assert sum(c.selectable for c in GEOMETRY_CONFIGS.values())==3
