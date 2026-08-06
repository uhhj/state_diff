from pathlib import Path

from scripts.experiment3.phase0.occp_common import config_schema, load_config


ROOT = Path(__file__).resolve().parents[3]


def _config():
    return load_config(ROOT / 'configs/experiment3/phase0/occp_single_pair_r1.json')


def test_r1_config_uses_diameter_clearance_semantics():
    config = _config()
    assert config_schema(config) == 'r1'
    assert 'jam_initial_clearance_diameter_scale' in config['geometry']
    assert 'jam_clearance_scale' not in config['geometry']
    assert config['geometry']['pin_radius_diameter_scale'] == 1.75


def test_r1_config_visible_readout_count_matches_exit_layout():
    geometry = _config()['geometry']
    exit_index = geometry['hidden_start_index'] + geometry['hidden_edge_count']
    assert geometry['num_beads'] - exit_index - 1 == geometry['visible_readout_count']


def test_r1_config_uses_trace_stride_one():
    assert _config()['trace']['stride'] == 1


def test_r1_config_preserves_ccda_audit_role():
    assert _config()['dataset_role'] == 'ccda_audit'
