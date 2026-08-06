from pathlib import Path

from scripts.experiment3.phase0.occp_common import load_config


ROOT = Path(__file__).resolve().parents[3]


def test_phase0a_config_is_the_frozen_single_pair_smoke():
    config = load_config(
        ROOT / 'configs/experiment3/phase0/occp_single_pair.json')
    assert config['seed'] == 73001
    assert config['pair_id'] == 'occp_073001'
    assert config['execution']['hz'] == 480
    assert config['execution']['no_action_steps'] == 24
    assert config['motion']['speed'] == 0.001
    assert config['geometry']['jam_clearance_scale'] == 0.95
    assert config['geometry']['pin_from_active_exit_spacing'] == 5.0
    assert len(config['smoke_debug_history']) == 5
    assert config['trace']['stride'] == 2
