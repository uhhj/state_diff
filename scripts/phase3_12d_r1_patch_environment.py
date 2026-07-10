#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_environment(path: Path) -> Tuple[str, bool]:
    text = path.read_text()
    original = text

    text = replace_once(
        text,
        '        self._ccda_video_recorder = None\n'
        '        self._ccda_video_label = ""\n',
        '        self._ccda_video_recorder = None\n'
        '        self._ccda_video_label = ""\n'
        '        # Phase3.12d-r1: serialize background physics with snapshot IO and\n'
        '        # surface task-hook failures instead of silently killing the thread.\n'
        '        self._ccda_step_lock = threading.RLock()\n'
        '        self._ccda_physics_hook_error = None\n',
        "environment init hook state",
    )

    old = '''            if self.running:\n                p.stepSimulation()\n            if self.ee is not None:\n                self.ee.step()\n            time.sleep(0.001)\n'''
    new = '''            if self.running:\n                with self._ccda_step_lock:\n                    p.stepSimulation()\n                    if self.ee is not None:\n                        self.ee.step()\n                    task = getattr(self, 'task', None)\n                    hook = getattr(task, 'physics_step_hook', None)\n                    if callable(hook):\n                        try:\n                            hook()\n                        except Exception as exc:\n                            # Do not let an exception silently terminate the daemon\n                            # thread. The rollout wrapper treats this as a hard error.\n                            self._ccda_physics_hook_error = repr(exc)\n            time.sleep(0.001)\n'''
    text = replace_once(text, old, new, "environment physics hook")

    if text != original:
        path.write_text(text)
    return text, text != original


def replace_method(text: str, method_name: str, next_method_name: str, body: str) -> str:
    pattern = re.compile(
        rf"^    def {re.escape(method_name)}\(.*?(?=^    def {re.escape(next_method_name)}\()",
        flags=re.MULTILINE | re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        if body.strip() in text:
            return text
        raise RuntimeError(f"method {method_name}: no match before {next_method_name}")
    if len(matches) != 1:
        raise RuntimeError(f"method {method_name}: expected 1 match, got {len(matches)}")
    m = matches[0]
    return text[:m.start()] + body.rstrip() + "\n\n" + text[m.end():]


def patch_task(path: Path) -> Tuple[str, bool]:
    text = path.read_text()
    original = text

    # __init__ and reset each contain this sequence. The replacement is idempotent.
    old_seq = '''        self._ccda_step_count = 0\n        self._breakaway_released = False\n        self._breakaway_release_step = None\n'''
    new_seq = '''        self._ccda_step_count = 0\n        self._ccda_physics_step_count = 0\n        self._breakaway_released = False\n        self._breakaway_release_step = None\n        self._breakaway_release_physics_step = None\n'''
    if old_seq in text:
        count = text.count(old_seq)
        if count != 2:
            raise RuntimeError(
                f"task counters: expected init+reset occurrences=2, found {count}"
            )
        text = text.replace(old_seq, new_seq)
    elif text.count(new_seq) != 2:
        raise RuntimeError("task counters: neither unpatched nor patched structure found")

    old_reward = '''    def reward(self):\n        """Call original reward, then add CCDA logging fields."""\n        self._ccda_step_count += 1\n        self._maybe_update_breakaway()\n        reward, extras = super().reward()\n        self._maybe_update_breakaway()\n        extras.update(self._ccda_extras())\n        return reward, extras\n'''
    new_reward = '''    def physics_step_hook(self):\n        """Update recoverable hidden contact at every simulated physics step.\n\n        The old implementation checked displacement only from reward(), after a\n        full pick-place primitive and settling. A bead could exceed the release\n        threshold transiently during the pull and return before reward(), which\n        incorrectly kept the pin attached.\n        """\n        self._ccda_physics_step_count += 1\n        self._maybe_update_breakaway(check_source="physics")\n\n    def reward(self):\n        """Call original reward, then add CCDA logging fields."""\n        self._ccda_step_count += 1\n        # Fallback checks keep direct/manual stepping and legacy callers safe.\n        self._maybe_update_breakaway(check_source="reward_pre")\n        reward, extras = super().reward()\n        self._maybe_update_breakaway(check_source="reward_post")\n        extras.update(self._ccda_extras())\n        return reward, extras\n'''
    text = replace_once(text, old_reward, new_reward, "task reward/physics hook")

    update_meta = '''    def _update_breakaway_meta_fields(self) -> None:\n        if self.hidden_condition != "hidden_breakaway_pin":\n            return\n        self.hidden_contact_meta["breakaway_released"] = bool(\n            self._breakaway_released\n        )\n        self.hidden_contact_meta["breakaway_release_step"] = (\n            self._breakaway_release_step\n        )\n        self.hidden_contact_meta["breakaway_release_physics_step"] = (\n            self._breakaway_release_physics_step\n        )\n        self.hidden_contact_meta["breakaway_max_disp_seen"] = float(\n            self._breakaway_max_disp_seen\n        )\n        self.hidden_contact_meta["ccda_physics_step_count"] = int(\n            self._ccda_physics_step_count\n        )\n'''
    text = replace_method(
        text,
        "_update_breakaway_meta_fields",
        "_maybe_update_breakaway",
        update_meta,
    )

    maybe_update = '''    def _maybe_update_breakaway(self, check_source="reward") -> None:\n        if self.hidden_condition != "hidden_breakaway_pin":\n            return\n        if self._breakaway_released:\n            self._update_breakaway_meta_fields()\n            return\n        if self._breakaway_bead_id is None or self._breakaway_anchor_pos is None:\n            self._update_breakaway_meta_fields()\n            return\n\n        try:\n            bead_pos = np.asarray(\n                p.getBasePositionAndOrientation(\n                    int(self._breakaway_bead_id)\n                )[0],\n                dtype=np.float32,\n            )\n            anchor = np.asarray(\n                self._breakaway_anchor_pos, dtype=np.float32\n            )\n            disp = float(np.linalg.norm((bead_pos - anchor)[:2]))\n        except Exception:\n            self._update_breakaway_meta_fields()\n            return\n\n        self._breakaway_max_disp_seen = max(\n            float(self._breakaway_max_disp_seen), disp\n        )\n        threshold = float(\n            os.environ.get("CCDA_BREAKAWAY_DISP", "0.035")\n        )\n        min_physics_steps = int(\n            os.environ.get("CCDA_BREAKAWAY_MIN_PHYSICS_STEPS", "1")\n        )\n\n        # reward() may run before the background thread has advanced. Keep the\n        # old action-step fallback only when explicitly requested; normal task\n        # execution releases according to actual physics-step displacement.\n        physics_ready = (\n            self._ccda_physics_step_count >= min_physics_steps\n        )\n        reward_fallback = (\n            str(check_source).startswith("reward")\n            and self._ccda_step_count\n            >= int(os.environ.get("CCDA_BREAKAWAY_MIN_STEP", "1"))\n        )\n        if disp < threshold or not (physics_ready or reward_fallback):\n            self._update_breakaway_meta_fields()\n            return\n\n        released_constraint_id = self._breakaway_constraint_id\n        try:\n            if released_constraint_id is not None:\n                p.removeConstraint(int(released_constraint_id))\n        except Exception:\n            # If it was already absent, treat the latent contact as released.\n            pass\n\n        self.hidden_constraint_ids = [\n            int(cid)\n            for cid in self.hidden_constraint_ids\n            if int(cid) != int(released_constraint_id)\n        ] if released_constraint_id is not None else list(\n            self.hidden_constraint_ids\n        )\n        self._breakaway_constraint_id = None\n        self._breakaway_released = True\n        self._breakaway_release_step = int(self._ccda_step_count)\n        self._breakaway_release_physics_step = int(\n            self._ccda_physics_step_count\n        )\n        self.hidden_contact_meta["breakaway_release_source"] = str(\n            check_source\n        )\n        self._update_breakaway_meta_fields()\n'''
    text = replace_method(
        text,
        "_maybe_update_breakaway",
        "_apply_hidden_breakaway_pin",
        maybe_update,
    )

    # Ensure newly created breakaway pins start with the new fields.
    old_init = '''        self._breakaway_released = False\n        self._breakaway_release_step = None\n        self._breakaway_max_disp_seen = 0.0\n'''
    new_init = '''        self._breakaway_released = False\n        self._breakaway_release_step = None\n        self._breakaway_release_physics_step = None\n        self._breakaway_max_disp_seen = 0.0\n'''
    text = replace_once(text, old_init, new_init, "breakaway pin initialization")

    old_meta = '''                "breakaway_release_step": None,\n                "breakaway_max_disp_seen": 0.0,\n'''
    new_meta = '''                "breakaway_release_step": None,\n                "breakaway_release_physics_step": None,\n                "ccda_physics_step_count": int(\n                    self._ccda_physics_step_count\n                ),\n                "breakaway_max_disp_seen": 0.0,\n'''
    text = replace_once(text, old_meta, new_meta, "breakaway metadata initialization")

    if text != original:
        path.write_text(text)
    return text, text != original


def validate(environment_text: str, task_text: str) -> None:
    required_environment = [
        "self._ccda_step_lock = threading.RLock()",
        "hook = getattr(task, 'physics_step_hook', None)",
        "self._ccda_physics_hook_error = repr(exc)",
    ]
    required_task = [
        "def physics_step_hook(self):",
        "self._maybe_update_breakaway(check_source=\"physics\")",
        "CCDA_BREAKAWAY_MIN_PHYSICS_STEPS",
        "self._breakaway_release_physics_step",
        "self._breakaway_constraint_id = None",
    ]
    missing = [x for x in required_environment if x not in environment_text]
    missing += [x for x in required_task if x not in task_text]
    if missing:
        raise RuntimeError(f"post-patch validation missing: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    env_path = root / "external/deformable-ravens/ravens/environment.py"
    task_path = (
        root
        / "external/deformable-ravens/ravens/tasks/ccda_hidden_contact_cable.py"
    )
    if not env_path.exists() or not task_path.exists():
        raise SystemExit("Expected DeformableRavens files are missing.")

    if args.check_only:
        validate(env_path.read_text(), task_path.read_text())
        print("[Phase3.12d-r1] environment semantics patch is present")
        return

    env_text, env_changed = patch_environment(env_path)
    task_text, task_changed = patch_task(task_path)
    validate(env_text, task_text)
    print(
        "[Phase3.12d-r1] patched environment={} task={}".format(
            env_changed, task_changed
        )
    )


if __name__ == "__main__":
    main()
