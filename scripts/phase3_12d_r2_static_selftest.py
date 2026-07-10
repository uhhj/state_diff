#!/usr/bin/env python3
from __future__ import annotations

import py_compile
import tempfile
from pathlib import Path

from phase3_12d_r2_patch_submodule import MARKER, patch_task
from phase3_12d_r2_prepare_query_local import copy_analyzer, transform


MOCK_TASK = '''#!/usr/bin/env python
import os
import time
from typing import Any, Dict, List
import numpy as np
import pybullet as p
from ravens.tasks.defs_cables import CableLineNoTarget

class HiddenContactCableLine(CableLineNoTarget):
    def __init__(self):
        super().__init__()
        self.hidden_condition = "free"
        self.hidden_contact_applied = False
        self.hidden_constraint_ids = []
        self.hidden_body_ids = []
        self.hidden_contact_meta = {}
        self.cable_bead_IDs = []
        self._ccda_env = None
        self._ccda_step_count = 0
        self._ccda_physics_step_count = 0
        self._breakaway_max_disp_seen = 0.0

    def reset(self, env, last_info=None):
        self._breakaway_max_disp_seen = 0.0

        super().reset(env, last_info=last_info)

        # Apply hidden contact after cable creation.
        self._apply_hidden_contact(env)

        # Let contacts settle briefly. Keep this short for smoke tests.
        settle_seconds = float(os.environ.get("CCDA_SETTLE_SECONDS", "0.25"))
        if settle_seconds > 0:
            env.start()
            time.sleep(settle_seconds)
            env.pause()

    def _ccda_extras(self):
        self._update_breakaway_meta_fields()
        bead_states = self._ordered_bead_states()
        return bead_states

    def _maybe_update_breakaway(self, check_source="reward") -> None:
        if self.hidden_condition != "hidden_breakaway_pin":
            return

    def _apply_hidden_breakaway_pin(self):
        damping = float(os.environ.get("CCDA_BREAKAWAY_DAMPING", "0.2"))
        return damping

    def physics_step_hook(self):
        pass

    def _apply_hidden_contact(self, env):
        self.hidden_contact_applied = True

    def _update_breakaway_meta_fields(self):
        pass

    def _ordered_bead_states(self):
        return []
'''


MOCK_QUERY = '''import phase3_12d_r1_common as common
import phase3_12c_matched_reset_common as p12c

def setup(env, task):
    return p12c.deterministic_reset(
        env,
        task,
        min_settle_steps=1,
        max_settle_steps=2,
        static_checks_required=1,
        static_check_interval=1,
    )

scope = "phase3_12d_r1_query_local_snapshot"
'''

MOCK_ANALYZER = 'ROOT = "phase3_12d_r1_candidate_oracle"\n'


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="phase312d_r2_selftest_") as tmp:
        root = Path(tmp)
        task = root / "task.py"
        task.write_text(MOCK_TASK)
        patch_task(task)
        once = task.read_text()
        if MARKER not in once:
            raise AssertionError("patch marker missing")
        patch_task(task)
        twice = task.read_text()
        if once != twice:
            raise AssertionError("patch is not idempotent")
        py_compile.compile(str(task), doraise=True)

        src_query = root / "r1_query.py"
        dst_query = root / "r2_query.py"
        src_query.write_text(MOCK_QUERY)
        transform(src_query, dst_query)
        query_text = dst_query.read_text()
        if "phase3_12d_r2_common" not in query_text:
            raise AssertionError("r2 common transform missing")
        if "common.deterministic_reset_deferred_arm(" not in query_text:
            raise AssertionError("deferred reset transform missing")
        if "phase3_12d_r1" in query_text:
            raise AssertionError("r1 token left in transformed query")
        py_compile.compile(str(dst_query), doraise=True)

        src_analyzer = root / "r1_analyze.py"
        dst_analyzer = root / "r2_analyze.py"
        src_analyzer.write_text(MOCK_ANALYZER)
        copy_analyzer(src_analyzer, dst_analyzer)
        if "phase3_12d_r1" in dst_analyzer.read_text():
            raise AssertionError("r1 token left in analyzer")
        py_compile.compile(str(dst_analyzer), doraise=True)

    print("[Phase3.12d-r2] static self-test PASS")


if __name__ == "__main__":
    main()
