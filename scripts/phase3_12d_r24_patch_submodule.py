#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


EXPECTED_SUBMODULE_HEAD = "e5525384af4af5bd0a02c3d1fd33ae84323d44e2"
CONDITION = "hidden_slack_breakaway_pin_v2"
MARKER = "PHASE3_12D_R24_SLACK_BREAKAWAY_V2"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def replace_once(
    text: str,
    old: str,
    new: str,
    *,
    name: str,
) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{name}: expected exactly one match, found {count}"
        )
    return text.replace(old, new, 1)


def insert_before_once(
    text: str,
    marker: str,
    block: str,
    *,
    name: str,
) -> str:
    if block.strip() in text:
        return text
    return replace_once(
        text,
        marker,
        block.rstrip() + "\n\n" + marker,
        name=name,
    )


def patch_environment(text: str) -> str:
    if MARKER in text:
        return text

    old = """                with self._ccda_step_lock:
                    p.stepSimulation()
                    if self.ee is not None:
                        self.ee.step()
                    task = getattr(self, 'task', None)
                    hook = getattr(task, 'physics_step_hook', None)
                    if callable(hook):
                        try:
                            hook()
                        except Exception as exc:
                            # Do not let an exception silently terminate the daemon
                            # thread. The rollout wrapper treats this as a hard error.
                            self._ccda_physics_hook_error = repr(exc)
"""
    new = f"""                with self._ccda_step_lock:
                    # {MARKER}: forces must be applied before the PyBullet
                    # step in which they are intended to act.
                    task = getattr(self, 'task', None)
                    pre_hook = getattr(task, 'physics_pre_step_hook', None)
                    if callable(pre_hook):
                        try:
                            pre_hook()
                        except Exception as exc:
                            if self._ccda_physics_hook_error is None:
                                self._ccda_physics_hook_error = (
                                    "physics_pre_step_hook: " + repr(exc)
                                )

                    p.stepSimulation()
                    if self.ee is not None:
                        self.ee.step()

                    hook = getattr(task, 'physics_step_hook', None)
                    if callable(hook):
                        try:
                            hook()
                        except Exception as exc:
                            # Do not let an exception silently terminate the daemon
                            # thread. The rollout wrapper treats this as a hard error.
                            if self._ccda_physics_hook_error is None:
                                self._ccda_physics_hook_error = (
                                    "physics_step_hook: " + repr(exc)
                                )
"""
    text = replace_once(
        text,
        old,
        new,
        name="environment step hook",
    )

    text = replace_once(
        text,
        """        self.pause()
        self.task = task
""",
        """        self.pause()
        self._ccda_physics_hook_error = None
        self.task = task
""",
        name="environment reset hook error",
    )
    return text


TASK_IMPORT = """from ravens.tasks.ccda_slack_breakaway import (
    SlackBreakawayConfig,
    UnilateralSlackBreakaway,
)
"""

RESET_METHOD = """    def _reset_slack_breakaway_v2_state(self) -> None:
        self._slack_model = None
        self._slack_bead_id = None
        self._slack_bead_local_index = None
        self._slack_last_output = None
        self._slack_last_applied_force = [0.0, 0.0, 0.0]
        self._slack_environment_semantics_version = (
            "ccda_hidden_slack_breakaway_v2"
        )
"""

PRE_STEP_METHODS = """    def physics_pre_step_hook(self):
        # Apply v2 unilateral tether force before the simulation step.
        if self.hidden_condition != "hidden_slack_breakaway_pin_v2":
            return
        if not self._hidden_contact_armed:
            self._update_slack_breakaway_v2_meta_fields()
            return
        if self._slack_model is None or self._slack_bead_id is None:
            raise RuntimeError(
                "slack-breakaway v2 is armed without a model or bead"
            )

        bead_id = int(self._slack_bead_id)
        position = p.getBasePositionAndOrientation(bead_id)[0]
        linear_velocity = p.getBaseVelocity(bead_id)[0]
        output = self._slack_model.evaluate(
            position,
            linear_velocity,
            physics_step=int(self._ccda_physics_step_count) + 1,
        )
        force = np.asarray(output.force_xyz, dtype=np.float64)
        if force.shape != (3,) or not np.all(np.isfinite(force)):
            raise RuntimeError("non-finite slack tether force")

        self._slack_last_output = output
        self._slack_last_applied_force = force.astype(float).tolist()

        if np.any(force != 0.0):
            p.applyExternalForce(
                bead_id,
                -1,
                forceObj=force.astype(float).tolist(),
                posObj=[float(value) for value in position],
                flags=p.WORLD_FRAME,
            )

        self._update_slack_breakaway_v2_meta_fields()
"""

SLACK_METHODS = """    def _slack_breakaway_v2_config(self) -> SlackBreakawayConfig:
        return SlackBreakawayConfig(
            slack_distance=float(
                os.environ.get("CCDA_SLACK_V2_DISTANCE", "0.010")
            ),
            spring_stiffness=float(
                os.environ.get("CCDA_SLACK_V2_STIFFNESS", "100.0")
            ),
            radial_damping=float(
                os.environ.get("CCDA_SLACK_V2_DAMPING", "0.20")
            ),
            max_tension=float(
                os.environ.get("CCDA_SLACK_V2_MAX_TENSION", "4.0")
            ),
            breakaway_extension=float(
                os.environ.get(
                    "CCDA_SLACK_V2_BREAKAWAY_EXTENSION",
                    "0.030",
                )
            ),
            breakaway_force=float(
                os.environ.get(
                    "CCDA_SLACK_V2_BREAKAWAY_FORCE",
                    "3.0",
                )
            ),
        )

    def _update_slack_breakaway_v2_meta_fields(self) -> None:
        if self.hidden_condition != "hidden_slack_breakaway_pin_v2":
            return

        model = self._slack_model
        snapshot = model.snapshot() if model is not None else None
        self.hidden_contact_meta.update(
            {
                "environment_semantics_version":
                    self._slack_environment_semantics_version,
                "condition": "hidden_slack_breakaway_pin_v2",
                "force_model": "unilateral_deadband_spring",
                "uses_world_constraint": False,
                "contains_no_action_force_below_deadband": True,
                "slack_bead_id": self._slack_bead_id,
                "slack_bead_local_index": self._slack_bead_local_index,
                "slack_state": (
                    snapshot["state"] if snapshot is not None else None
                ),
                "slack_anchor_xy": (
                    snapshot["anchor_xy"] if snapshot is not None else None
                ),
                "slack_engagement_physics_step": (
                    snapshot["engagement_physics_step"]
                    if snapshot is not None
                    else None
                ),
                "slack_release_physics_step": (
                    snapshot["release_physics_step"]
                    if snapshot is not None
                    else None
                ),
                "slack_release_reason": (
                    snapshot["release_reason"]
                    if snapshot is not None
                    else None
                ),
                "slack_max_radial_distance": (
                    snapshot["max_radial_distance"]
                    if snapshot is not None
                    else 0.0
                ),
                "slack_max_extension": (
                    snapshot["max_extension"]
                    if snapshot is not None
                    else 0.0
                ),
                "slack_max_tension": (
                    snapshot["max_tension"]
                    if snapshot is not None
                    else 0.0
                ),
                "slack_last_tension": (
                    snapshot["last_tension"]
                    if snapshot is not None
                    else 0.0
                ),
                "slack_last_force": list(
                    self._slack_last_applied_force
                ),
                "hidden_body_ids": [
                    int(value) for value in self.hidden_body_ids
                ],
                "hidden_constraint_ids": [
                    int(value) for value in self.hidden_constraint_ids
                ],
            }
        )

    def _apply_hidden_slack_breakaway_pin_v2(self) -> None:
        if self.hidden_body_ids or self.hidden_constraint_ids:
            raise RuntimeError(
                "slack-breakaway v2 must not create hidden bodies or "
                "PyBullet constraints"
            )

        bead_ratio = float(
            os.environ.get("CCDA_SLACK_V2_BEAD_RATIO", "0.45")
        )
        bead_ratio = min(max(bead_ratio, 0.05), 0.95)
        index = int(
            round(bead_ratio * (len(self.cable_bead_IDs) - 1))
        )
        index = max(0, min(len(self.cable_bead_IDs) - 1, index))
        bead_id = int(self.cable_bead_IDs[index])
        bead_position = self._bead_position(bead_id)

        self._slack_bead_id = bead_id
        self._slack_bead_local_index = index
        self._slack_model = UnilateralSlackBreakaway(
            config=self._slack_breakaway_v2_config(),
            anchor_position=bead_position,
        )
        self._slack_last_output = None
        self._slack_last_applied_force = [0.0, 0.0, 0.0]

        parameters = self._base_recoverability_params(
            "hidden_slack_breakaway_pin_v2"
        )
        parameters.update(self._slack_model.snapshot()["config"])

        self.hidden_contact_meta.update(
            {
                "condition": "hidden_slack_breakaway_pin_v2",
                "recoverability_class_candidate":
                    "recoverable_candidate",
                "recoverability_params": parameters,
                "environment_semantics_version":
                    self._slack_environment_semantics_version,
                "force_model": "unilateral_deadband_spring",
                "uses_world_constraint": False,
                "contains_no_action_force_below_deadband": True,
            }
        )
        self._update_slack_breakaway_v2_meta_fields()

    def ccda_snapshot_state(self) -> Dict[str, Any]:
        # Capture Python-side state not included in p.saveState().
        return {
            "snapshot_version": "ccda_task_snapshot_v2",
            "hidden_condition": str(self.hidden_condition),
            "ccda_step_count": int(self._ccda_step_count),
            "ccda_physics_step_count": int(
                self._ccda_physics_step_count
            ),
            "hidden_contact_pending": bool(
                self._hidden_contact_pending
            ),
            "hidden_contact_armed": bool(
                self._hidden_contact_armed
            ),
            "slack_bead_id": self._slack_bead_id,
            "slack_bead_local_index": self._slack_bead_local_index,
            "slack_last_applied_force": list(
                self._slack_last_applied_force
            ),
            "slack_model": (
                self._slack_model.snapshot()
                if self._slack_model is not None
                else None
            ),
        }

    def ccda_restore_state(self, snapshot: Dict[str, Any]) -> None:
        # Restore Python-side state after p.restoreState().
        if snapshot.get("snapshot_version") != "ccda_task_snapshot_v2":
            raise ValueError("unsupported CCDA task snapshot")
        if str(snapshot["hidden_condition"]) != str(
            self.hidden_condition
        ):
            raise ValueError("snapshot hidden condition mismatch")

        self._ccda_step_count = int(snapshot["ccda_step_count"])
        self._ccda_physics_step_count = int(
            snapshot["ccda_physics_step_count"]
        )
        self._hidden_contact_pending = bool(
            snapshot["hidden_contact_pending"]
        )
        self._hidden_contact_armed = bool(
            snapshot["hidden_contact_armed"]
        )
        self._slack_bead_id = snapshot.get("slack_bead_id")
        self._slack_bead_local_index = snapshot.get(
            "slack_bead_local_index"
        )
        self._slack_last_applied_force = [
            float(value)
            for value in snapshot.get(
                "slack_last_applied_force",
                [0.0, 0.0, 0.0],
            )
        ]

        model_snapshot = snapshot.get("slack_model")
        self._slack_model = (
            UnilateralSlackBreakaway.from_snapshot(model_snapshot)
            if model_snapshot is not None
            else None
        )
        self._slack_last_output = None
        self._update_hidden_arm_meta_fields()
        self._update_slack_breakaway_v2_meta_fields()
"""


def patch_task(text: str) -> str:
    if MARKER in text:
        return text

    text = replace_once(
        text,
        "from ravens.tasks.defs_cables import CableLineNoTarget\n",
        (
            "from ravens.tasks.defs_cables import CableLineNoTarget\n"
            + TASK_IMPORT
        ),
        name="task helper import",
    )

    text = replace_once(
        text,
        '        "hidden_breakaway_pin",\n',
        (
            '        "hidden_breakaway_pin",\n'
            '        "hidden_slack_breakaway_pin_v2",\n'
        ),
        name="condition tuple",
    )

    text = insert_before_once(
        text,
        "    def reset(self, env, last_info=None):",
        RESET_METHOD,
        name="slack reset helper",
    )

    old = "        self._breakaway_max_disp_seen = 0.0\n"
    count = text.count(old)
    if count < 2:
        raise RuntimeError(
            "expected at least two breakaway reset anchors, found "
            f"{count}"
        )
    text = text.replace(
        old,
        old + "        self._reset_slack_breakaway_v2_state()\n",
        2,
    )

    text = insert_before_once(
        text,
        "    def physics_step_hook(self):",
        PRE_STEP_METHODS,
        name="physics pre-step hook",
    )

    text = replace_once(
        text,
        """        self._ccda_physics_step_count += 1
        self._maybe_update_breakaway(check_source="physics")
""",
        """        self._ccda_physics_step_count += 1
        self._maybe_update_breakaway(check_source="physics")
        self._update_slack_breakaway_v2_meta_fields()
""",
        name="physics post-step metadata",
    )

    text = replace_once(
        text,
        """        self._update_hidden_arm_meta_fields()
        self._update_breakaway_meta_fields()
        bead_states = self._ordered_bead_states()
""",
        """        self._update_hidden_arm_meta_fields()
        self._update_breakaway_meta_fields()
        self._update_slack_breakaway_v2_meta_fields()
        bead_states = self._ordered_bead_states()
""",
        name="extras slack metadata",
    )

    text = replace_once(
        text,
        """        if condition == "hidden_breakaway_pin":
            return self._apply_hidden_breakaway_pin()
""",
        """        if condition == "hidden_breakaway_pin":
            return self._apply_hidden_breakaway_pin()
        if condition == "hidden_slack_breakaway_pin_v2":
            return self._apply_hidden_slack_breakaway_pin_v2()
""",
        name="v2 condition dispatch",
    )

    text = insert_before_once(
        text,
        '    def _apply_hidden_pin(self, mode: str = "hard") -> None:',
        SLACK_METHODS,
        name="slack implementation",
    )

    text = replace_once(
        text,
        "import os\n",
        f"# {MARKER}\nimport os\n",
        name="task patch marker",
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submodule",
        default="/data/state_diff2/external/deformable-ravens",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--allow-head-mismatch",
        action="store_true",
    )
    args = parser.parse_args()

    root = Path(args.submodule).resolve()
    head = git(root, "rev-parse", "HEAD")
    status = git(root, "status", "--short")
    allowed_dirty_path = "ravens/tasks/ccda_slack_breakaway.py"
    unexpected_status = []
    for line in status.splitlines():
        path = line[3:].strip() if len(line) >= 4 else ""
        if path != allowed_dirty_path:
            unexpected_status.append(line)
    if unexpected_status:
        raise SystemExit(
            "submodule has unexpected changes before patching:\n"
            + "\n".join(unexpected_status)
        )
    if (
        head != EXPECTED_SUBMODULE_HEAD
        and not args.allow_head_mismatch
    ):
        raise SystemExit(
            f"unexpected submodule HEAD: {head}; "
            f"expected {EXPECTED_SUBMODULE_HEAD}"
        )

    environment_path = root / "ravens/environment.py"
    task_path = root / "ravens/tasks/ccda_hidden_contact_cable.py"
    helper_path = root / "ravens/tasks/ccda_slack_breakaway.py"

    if not helper_path.exists():
        raise SystemExit(
            f"write the reviewed helper first: {helper_path}"
        )

    environment_old = environment_path.read_text()
    task_old = task_path.read_text()
    environment_new = patch_environment(environment_old)
    task_new = patch_task(task_old)

    if not args.apply:
        print(
            {
                "head": head,
                "environment_changed": environment_new != environment_old,
                "task_changed": task_new != task_old,
                "helper_sha256": hashlib.sha256(
                    helper_path.read_bytes()
                ).hexdigest(),
            }
        )
        return

    environment_path.write_text(environment_new)
    task_path.write_text(task_new)

    assert patch_environment(environment_new) == environment_new
    assert patch_task(task_new) == task_new

    print("patched", environment_path)
    print("patched", task_path)
    print("helper", helper_path)


if __name__ == "__main__":
    main()
