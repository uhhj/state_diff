#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "PHASE3_12D_R2_DEFERRED_ARMING"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def insert_before_once(text: str, anchor: str, block: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(anchor, block + anchor, 1)


def patch_task(path: Path) -> None:
    text = path.read_text()

    if MARKER in text:
        required = [
            "arm_hidden_contact_after_settle",
            "_hidden_contact_pending",
            "_hidden_contact_armed",
            "CCDA_DEFER_HIDDEN_CONTACT_ARMING",
            "CCDA_POST_ARM_SETTLE_SECONDS",
            'CCDA_BREAKAWAY_DAMPING", "0.0"',
        ]
        missing = [token for token in required if token not in text]
        if missing:
            raise RuntimeError(f"Existing r2 patch is incomplete: {missing}")
        print(f"[r2 patch] already applied: {path}")
        return

    if "class HiddenContactCableLine(CableLineNoTarget):" not in text:
        raise RuntimeError(f"Task class anchor missing in {path}")

    reset_pos = text.find("    def reset(self, env, last_info=None):")
    if reset_pos < 0:
        raise RuntimeError("reset() anchor missing")
    init_text = text[:reset_pos]
    rest = text[reset_pos:]
    init_needle = "        self._breakaway_max_disp_seen = 0.0\n"
    if init_text.count(init_needle) != 1:
        raise RuntimeError(
            "Expected exactly one __init__ _breakaway_max_disp_seen assignment"
        )
    init_replacement = init_needle + f'''
        # {MARKER}: hidden conditions are created only after the common
        # visible cable settling phase.
        self._hidden_contact_pending = False
        self._hidden_contact_armed = False
        self._hidden_contact_arm_mode = "uninitialized"
        self._hidden_contact_arm_step = None
        self._hidden_contact_arm_physics_step = None
        self._hidden_contact_arm_pre_xy = None
        self._hidden_contact_arm_post_xy = None
        self._hidden_contact_arm_max_abs_jump = 0.0
        self._hidden_contact_arm_mae_jump = 0.0
'''
    init_text = init_text.replace(init_needle, init_replacement, 1)
    text = init_text + rest

    reset_needle = (
        "        self._breakaway_max_disp_seen = 0.0\n\n"
        "        super().reset(env, last_info=last_info)\n"
    )
    reset_replacement = (
        "        self._breakaway_max_disp_seen = 0.0\n"
        "        self._hidden_contact_pending = False\n"
        "        self._hidden_contact_armed = False\n"
        "        self._hidden_contact_arm_mode = \"reset\"\n"
        "        self._hidden_contact_arm_step = None\n"
        "        self._hidden_contact_arm_physics_step = None\n"
        "        self._hidden_contact_arm_pre_xy = None\n"
        "        self._hidden_contact_arm_post_xy = None\n"
        "        self._hidden_contact_arm_max_abs_jump = 0.0\n"
        "        self._hidden_contact_arm_mae_jump = 0.0\n\n"
        "        super().reset(env, last_info=last_info)\n"
    )
    text = replace_once(
        text,
        reset_needle,
        reset_replacement,
        "reset arming-state insertion",
    )

    apply_block = '''
        # Apply hidden contact after cable creation.
        self._apply_hidden_contact(env)

        # Let contacts settle briefly. Keep this short for smoke tests.
        settle_seconds = float(os.environ.get("CCDA_SETTLE_SECONDS", "0.25"))
        if settle_seconds > 0:
            env.start()
            time.sleep(settle_seconds)
            env.pause()
'''
    if text.count(apply_block) != 1:
        raise RuntimeError(
            "immediate hidden-contact block: expected one exact match, "
            f"found {text.count(apply_block)}"
        )

    apply_replacement = '''
        # Phase3.12d-r2: all non-free conditions may defer hidden-contact
        # creation until a shared, condition-independent visible settling
        # phase has completed.
        defer_hidden = (
            os.environ.get("CCDA_DEFER_HIDDEN_CONTACT_ARMING", "0") == "1"
        )
        if self.hidden_condition == "free" or not defer_hidden:
            self._apply_hidden_contact(env)
            self._hidden_contact_pending = False
            self._hidden_contact_armed = True
            self._hidden_contact_arm_mode = "reset_immediate"
            self._hidden_contact_arm_step = int(self._ccda_step_count)
            self._hidden_contact_arm_physics_step = int(
                self._ccda_physics_step_count
            )
        else:
            self._hidden_contact_pending = True
            self._hidden_contact_armed = False
            self._hidden_contact_arm_mode = "deferred_after_visible_settle"
            self.hidden_contact_meta.update(
                {
                    "applied": False,
                    "hidden_contact_pending": True,
                    "hidden_contact_armed": False,
                    "hidden_contact_arm_mode": self._hidden_contact_arm_mode,
                }
            )

        # Long settling after hidden-contact creation reintroduces visible
        # condition leakage. This is legacy opt-in only and defaults to zero.
        post_arm_settle_seconds = float(
            os.environ.get("CCDA_POST_ARM_SETTLE_SECONDS", "0")
        )
        if (
            self._hidden_contact_armed
            and self.hidden_condition != "free"
            and post_arm_settle_seconds > 0
        ):
            env.start()
            time.sleep(post_arm_settle_seconds)
            env.pause()
'''
    text = text.replace(
        apply_block,
        "\n" + apply_replacement.lstrip("\n"),
        1,
    )

    methods = '''
    def _ordered_bead_xy_array(self) -> np.ndarray:
        """Return ordered bead XY for hidden-contact arming diagnostics."""
        if not self.cable_bead_IDs:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(
            [
                p.getBasePositionAndOrientation(int(bead_id))[0][:2]
                for bead_id in self.cable_bead_IDs
            ],
            dtype=np.float32,
        )

    def _update_hidden_arm_meta_fields(self) -> None:
        self.hidden_contact_meta["hidden_contact_pending"] = bool(
            self._hidden_contact_pending
        )
        self.hidden_contact_meta["hidden_contact_armed"] = bool(
            self._hidden_contact_armed
        )
        self.hidden_contact_meta["hidden_contact_arm_mode"] = str(
            self._hidden_contact_arm_mode
        )
        self.hidden_contact_meta["hidden_contact_arm_step"] = (
            self._hidden_contact_arm_step
        )
        self.hidden_contact_meta["hidden_contact_arm_physics_step"] = (
            self._hidden_contact_arm_physics_step
        )
        self.hidden_contact_meta["hidden_contact_arm_max_abs_jump"] = float(
            self._hidden_contact_arm_max_abs_jump
        )
        self.hidden_contact_meta["hidden_contact_arm_mae_jump"] = float(
            self._hidden_contact_arm_mae_jump
        )

    def arm_hidden_contact_after_settle(self, env=None) -> Dict[str, Any]:
        """Install latent contact at the current settled pose, exactly once.

        This method performs no physics step. Point constraints therefore use
        the current bead pose as a zero-offset world anchor, and the caller can
        separately audit immediate installation jump and short-horizon drift.
        """
        if env is None:
            env = self._ccda_env
        if env is None:
            raise RuntimeError("No environment available for hidden-contact arming")

        if self._hidden_contact_armed:
            self._update_hidden_arm_meta_fields()
            return {
                "already_armed": True,
                "condition": self.hidden_condition,
                "arm_mode": self._hidden_contact_arm_mode,
                "max_abs_jump": float(self._hidden_contact_arm_max_abs_jump),
                "mae_jump": float(self._hidden_contact_arm_mae_jump),
            }

        if self.hidden_condition == "free":
            if not self.hidden_contact_applied:
                self._apply_hidden_contact(env)
            self._hidden_contact_pending = False
            self._hidden_contact_armed = True
            self._hidden_contact_arm_mode = "free_noop"
            self._hidden_contact_arm_step = int(self._ccda_step_count)
            self._hidden_contact_arm_physics_step = int(
                self._ccda_physics_step_count
            )
            self._update_hidden_arm_meta_fields()
            return {
                "already_armed": False,
                "condition": self.hidden_condition,
                "arm_mode": self._hidden_contact_arm_mode,
                "max_abs_jump": 0.0,
                "mae_jump": 0.0,
            }

        before = self._ordered_bead_xy_array()
        self._apply_hidden_contact(env)
        after = self._ordered_bead_xy_array()

        if before.shape != after.shape:
            raise RuntimeError(
                "Bead shape changed while arming hidden contact: "
                f"{before.shape} -> {after.shape}"
            )

        diff = np.abs(after - before)
        self._hidden_contact_arm_pre_xy = before.copy()
        self._hidden_contact_arm_post_xy = after.copy()
        self._hidden_contact_arm_max_abs_jump = (
            float(np.max(diff)) if diff.size else 0.0
        )
        self._hidden_contact_arm_mae_jump = (
            float(np.mean(diff)) if diff.size else 0.0
        )
        self._hidden_contact_pending = False
        self._hidden_contact_armed = True
        self._hidden_contact_arm_mode = "deferred_zero_offset"
        self._hidden_contact_arm_step = int(self._ccda_step_count)
        self._hidden_contact_arm_physics_step = int(
            self._ccda_physics_step_count
        )
        self._update_hidden_arm_meta_fields()

        return {
            "already_armed": False,
            "condition": self.hidden_condition,
            "arm_mode": self._hidden_contact_arm_mode,
            "max_abs_jump": float(self._hidden_contact_arm_max_abs_jump),
            "mae_jump": float(self._hidden_contact_arm_mae_jump),
            "hidden_body_ids": [int(x) for x in self.hidden_body_ids],
            "hidden_constraint_ids": [int(x) for x in self.hidden_constraint_ids],
        }

'''
    text = insert_before_once(
        text,
        "    def physics_step_hook(self):\n",
        methods,
        "arming methods insertion",
    )

    text = replace_once(
        text,
        "        self._update_breakaway_meta_fields()\n"
        "        bead_states = self._ordered_bead_states()\n",
        "        self._update_hidden_arm_meta_fields()\n"
        "        self._update_breakaway_meta_fields()\n"
        "        bead_states = self._ordered_bead_states()\n",
        "_ccda_extras arming metadata",
    )

    maybe_needle = (
        "    def _maybe_update_breakaway(self, check_source=\"reward\") -> None:\n"
        "        if self.hidden_condition != \"hidden_breakaway_pin\":\n"
        "            return\n"
    )
    maybe_replacement = maybe_needle + (
        "        if not self._hidden_contact_armed:\n"
        "            self._update_breakaway_meta_fields()\n"
        "            return\n"
    )
    text = replace_once(
        text,
        maybe_needle,
        maybe_replacement,
        "breakaway arming guard",
    )

    text = replace_once(
        text,
        '        damping = float(os.environ.get("CCDA_BREAKAWAY_DAMPING", "0.2"))\n',
        '        damping = float(os.environ.get("CCDA_BREAKAWAY_DAMPING", "0.0"))\n',
        "breakaway damping default",
    )

    path.write_text(text)
    print(f"[r2 patch] patched: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    task_path = (
        root
        / "external/deformable-ravens/ravens/tasks/ccda_hidden_contact_cable.py"
    )
    if not task_path.exists():
        raise FileNotFoundError(task_path)

    patch_task(task_path)


if __name__ == "__main__":
    main()
