#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import sys
import traceback
import types
from pathlib import Path


def install_ravens_stub(defravens: Path):
    for name in list(sys.modules):
        if name == "ravens" or name.startswith("ravens."):
            del sys.modules[name]
    pkg = types.ModuleType("ravens")
    pkg.__path__ = [str(defravens / "ravens")]
    pkg.__file__ = str(defravens / "ravens" / "__init__.py")
    pkg.__package__ = "ravens"
    sys.modules["ravens"] = pkg


def assert_no_forbidden_modules(stage):
    bad = []
    for name in sys.modules:
        if name == "tensorflow" or name.startswith("tensorflow."):
            bad.append(name)
        if name.startswith("ravens.agents") or name.startswith("ravens.models") or name.startswith("ravens.datasets"):
            bad.append(name)
    if bad:
        raise RuntimeError(f"Forbidden modules loaded at {stage}: {bad[:20]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--out_json", default="reports/phase3_3b_task_env_dryrun_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_3b_task_env_dryrun_report.md")
    parser.add_argument("--attempt_env", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    defravens = root / "external/deformable-ravens"
    if str(defravens) not in sys.path:
        sys.path.insert(0, str(defravens))

    checks = {}
    details = {}
    errors = []
    os.environ.setdefault("CCDA_BREAKAWAY_FORCE", "2.6")
    os.environ.setdefault("CCDA_BREAKAWAY_DISP", "0.045")
    os.environ.setdefault("CCDA_BREAKAWAY_BEAD_RATIO", "0.45")
    os.environ.setdefault("CCDA_ORACLE_BREAKAWAY_PULL_DIST", "0.36")

    tasks = None
    Environment = None
    try:
        assert_no_forbidden_modules("start")
        install_ravens_stub(defravens)
        tasks = importlib.import_module("ravens.tasks")
        env_mod = importlib.import_module("ravens.environment")
        Environment = env_mod.Environment
        assert_no_forbidden_modules("after_minimal_import")
        checks["minimal_imports_ok"] = True
    except Exception as exc:
        checks["minimal_imports_ok"] = False
        errors.append({"stage": "minimal_imports", "error": repr(exc), "traceback": traceback.format_exc()})

    if tasks is not None:
        checks["task_registered"] = "hidden-contact-cable-line" in tasks.names
        details["task_names_has_hidden_contact"] = checks["task_registered"]
        try:
            task = tasks.names["hidden-contact-cable-line"]()
            checks["task_instantiates"] = True
            details["task_conditions"] = list(getattr(task, "CONDITIONS", []))
            checks["task_has_hidden_breakaway_pin"] = "hidden_breakaway_pin" in details["task_conditions"]
        except Exception as exc:
            checks["task_instantiates"] = False
            errors.append({"stage": "task_instantiate", "error": repr(exc), "traceback": traceback.format_exc()})
    else:
        checks["task_registered"] = False
        checks["task_instantiates"] = False
        checks["task_has_hidden_breakaway_pin"] = False

    env_results = {}
    if args.attempt_env and Environment is not None and tasks is not None and checks.get("task_registered"):
        env_results["attempted"] = True
        env_results["note"] = "Environment constructor was not invoked by default to avoid PyBullet side effects."
        checks["env_dryrun_attempted"] = True
        checks["env_dryrun_ok"] = None
        env_results["blocked_reason"] = "Import/task dry-run passed; actual env reset is deferred to gated rollout smoke."
    else:
        env_results["attempted"] = False
        checks["env_dryrun_attempted"] = False
        checks["env_dryrun_ok"] = None

    try:
        assert_no_forbidden_modules("end")
        checks["forbidden_modules_not_loaded"] = True
    except Exception as exc:
        checks["forbidden_modules_not_loaded"] = False
        errors.append({"stage": "forbidden_module_check", "error": repr(exc), "traceback": traceback.format_exc()})

    required = [
        "minimal_imports_ok",
        "task_registered",
        "task_instantiates",
        "task_has_hidden_breakaway_pin",
        "forbidden_modules_not_loaded",
    ]
    verdict = "PASS" if all(checks.get(key) is True for key in required) else "FAIL"
    payload = {
        "verdict": verdict,
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "checks": checks,
        "details": details,
        "env_results": env_results,
        "errors": errors,
        "interpretation": "PASS means TensorFlow-free task import path is viable. It is not rollout evidence.",
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.3b Task/Env Dry Run",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Python: `{sys.executable}`",
        f"- Conda env: `{os.environ.get('CONDA_DEFAULT_ENV')}`",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for key, value in checks.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "## Details",
        "",
        f"- task_conditions: `{details.get('task_conditions')}`",
        f"- env_attempted: `{env_results.get('attempted')}`",
        f"- env_note: `{env_results.get('note', '')}`",
        f"- env_blocked_reason: `{env_results.get('blocked_reason', '')}`",
        "",
        "## Errors",
        "",
    ]
    if errors:
        for item in errors:
            lines.append(f"- `{item['stage']}`: `{item['error']}`")
    else:
        lines.append("- None")
    lines += [
        "",
        "## Interpretation",
        "",
        "- No learned rollout was run.",
        "- No Phase4 or CPS was run.",
        "- PASS only means TensorFlow-free Ravens task path is viable.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict != "PASS":
        raise SystemExit("[Phase3.3b][FAIL] task/env dry-run failed")


if __name__ == "__main__":
    main()
