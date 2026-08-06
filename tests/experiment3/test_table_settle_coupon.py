import copy
import json
from pathlib import Path

import numpy as np

from scripts.experiment3.phase0_soft_blockpush_r2.analyze_table_settle import (
    COMPLETE, classify_table, freeze_outputs)
from scripts.experiment3.phase0_soft_blockpush_r2.common import load_config
from state_diff.env.block_pushing.table_settle_coupon import TableSettleCoupon


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment3/soft_blockpush_phase0b_r2.json"
PROFILE = ROOT / "configs/experiment3/material_profiles_r2/soft_block_kv_r2_a6_z025.json"


def test_short_table_has_floor_gravity_dynamic_nodes_and_separate_audit():
    config = copy.deepcopy(load_config(str(CONFIG)))
    config["table_settle"]["warmup_outer_steps"] = 2
    config["table_settle"]["audit_outer_steps"] = 3
    coupon = TableSettleCoupon(config, str(PROFILE), 2)
    try:
        initial = coupon.block.positions().copy()
        trace = coupon.run()
        assert len(coupon.block.body_ids) == 72
        assert coupon.client.getNumBodies() == 73
        assert all(coupon.client.getDynamicsInfo(body, -1)[0] > 0
                   for body in coupon.block.body_ids)
        assert [row["phase"] for row in trace] == [
            "warmup", "warmup", "audit", "audit", "audit"]
        assert trace[0]["node_velocities"] != np.zeros((72, 3)).tolist()
        assert np.max(np.abs(coupon.block.positions() - initial)) > 0
    finally:
        coupon.close()


def test_freeze_is_written_only_after_every_required_complete(tmp_path):
    config = copy.deepcopy(load_config(str(CONFIG)))
    config["report_root"] = str(tmp_path / "reports")
    root = Path(config["report_root"]); profile = "kv_r2_a6_z025"
    metadata = {"config": config, "material_profile": {
        "profile_name": profile, "profile_version": 2,
        "structural_stiffness_n_per_m": 48.0}}
    report_dir = root / "table_settle" / profile
    report_dir.mkdir(parents=True)
    blocked = {"verdict": "PHASE0B_R2_TABLE_SETTLE_UNSTABLE"}
    assert freeze_outputs(report_dir, metadata, blocked) == {}
    assert not (root / "frozen_material").exists()

    paths = (root / "mechanics_validation" / "selected_microsteps.json",
             root / "coupon" / profile / "axial" / "metrics.json",
             root / "coupon" / profile / "shear" / "metrics.json")
    values = ({"verdict": "PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE",
               "microsteps_per_outer": 8},
              {"verdict": "PHASE0B_R2_AXIAL_COMPLETE"},
              {"verdict": "PHASE0B_R2_SHEAR_COMPLETE"})
    for path, value in zip(paths, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    outputs = freeze_outputs(report_dir, metadata, {"verdict": COMPLETE})
    assert set(outputs) == {"material", "microsteps", "manifest"}
    frozen = json.loads(Path(outputs["microsteps"]).read_text(encoding="utf-8"))
    assert frozen["microsteps_per_outer"] == 8
