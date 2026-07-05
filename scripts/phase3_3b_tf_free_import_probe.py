#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import sys
import traceback
import types
from pathlib import Path

SAFE_MODULES = [
    "numpy",
    "torch",
    "pybullet",
    "meshcat",
]

FORBIDDEN_MODULES = [
    "tensorflow",
    "ravens.agents",
    "ravens.models",
    "ravens.datasets",
]


def check_import(name):
    try:
        mod = importlib.import_module(name)
        return {"ok": True, "version": getattr(mod, "__version__", "unknown"), "error": None, "traceback": None}
    except Exception as exc:
        return {"ok": False, "version": None, "error": repr(exc), "traceback": traceback.format_exc()}


def module_loaded_prefix(prefix):
    return sorted([m for m in sys.modules if m == prefix or m.startswith(prefix + ".")])


def install_ravens_stub(defravens: Path):
    for name in list(sys.modules):
        if name == "ravens" or name.startswith("ravens."):
            del sys.modules[name]
    pkg = types.ModuleType("ravens")
    pkg.__path__ = [str(defravens / "ravens")]
    pkg.__file__ = str(defravens / "ravens" / "__init__.py")
    pkg.__package__ = "ravens"
    sys.modules["ravens"] = pkg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--out_json", default="reports/phase3_3b_tf_free_import_probe_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_3b_tf_free_import_probe_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    defravens = root / "external/deformable-ravens"
    if str(defravens) not in sys.path:
        sys.path.insert(0, str(defravens))

    before_forbidden_loaded = {m: module_loaded_prefix(m) for m in FORBIDDEN_MODULES}
    safe_results = {name: check_import(name) for name in SAFE_MODULES}

    task_registry_ok = False
    task_registered = False
    task_error = None
    environment_ok = False
    environment_error = None
    install_ravens_stub(defravens)
    try:
        tasks_mod = importlib.import_module("ravens.tasks")
        task_registry_ok = hasattr(tasks_mod, "names")
        task_registered = task_registry_ok and "hidden-contact-cable-line" in tasks_mod.names
    except Exception as exc:
        task_error = repr(exc)
    try:
        env_mod = importlib.import_module("ravens.environment")
        environment_ok = hasattr(env_mod, "Environment")
    except Exception as exc:
        environment_error = repr(exc)

    after_forbidden_loaded = {m: module_loaded_prefix(m) for m in FORBIDDEN_MODULES}
    checks = {
        "python_env_coord_bimanual": "coord_bimanual" in sys.executable or os.environ.get("CONDA_DEFAULT_ENV") == "coord_bimanual",
        "safe_numpy": safe_results["numpy"]["ok"],
        "safe_torch": safe_results["torch"]["ok"],
        "safe_pybullet": safe_results["pybullet"]["ok"],
        "safe_ravens_tasks": task_registry_ok,
        "safe_environment": environment_ok,
        "task_registry_ok": task_registry_ok,
        "hidden_contact_task_registered": task_registered,
        "tensorflow_not_loaded": not after_forbidden_loaded["tensorflow"],
        "ravens_agents_not_loaded": not after_forbidden_loaded["ravens.agents"],
        "ravens_models_not_loaded": not after_forbidden_loaded["ravens.models"],
        "ravens_datasets_not_loaded": not after_forbidden_loaded["ravens.datasets"],
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "verdict": verdict,
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "root": str(root),
        "deformable_ravens_path": str(defravens),
        "safe_results": safe_results,
        "actual_environment_module": "ravens.environment",
        "requested_environment_module_note": "This DeformableRavens fork exposes Environment at ravens.environment, not ravens.environments.environment.",
        "before_forbidden_loaded": before_forbidden_loaded,
        "after_forbidden_loaded": after_forbidden_loaded,
        "task_registry_ok": task_registry_ok,
        "task_registered": task_registered,
        "task_error": task_error,
        "environment_ok": environment_ok,
        "environment_error": environment_error,
        "checks": checks,
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.3b TensorFlow-Free Import Probe",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Python: `{sys.executable}`",
        f"- Conda env: `{os.environ.get('CONDA_DEFAULT_ENV')}`",
        "- Environment module: `ravens.environment`",
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
        "## Safe Import Results",
        "",
        "| Module | OK | Version | Error |",
        "|---|---:|---|---|",
    ]
    for name, result in safe_results.items():
        lines.append(f"| `{name}` | `{result['ok']}` | `{result['version']}` | `{result['error']}` |")
    lines += [
        "| `ravens.tasks` | `" + str(task_registry_ok) + "` | `unknown` | `" + str(task_error) + "` |",
        "| `ravens.environment` | `" + str(environment_ok) + "` | `unknown` | `" + str(environment_error) + "` |",
        "",
        "## Forbidden Loaded Modules",
        "",
        "| Prefix | Loaded modules |",
        "|---|---|",
    ]
    for name, mods in after_forbidden_loaded.items():
        lines.append(f"| `{name}` | `{mods}` |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- This probe intentionally avoids top-level `import ravens`.",
        "- PASS means rollout runtime may not need TensorFlow.",
        "- FAIL blocks rollout smoke.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict != "PASS":
        raise SystemExit("[Phase3.3b][FAIL] TensorFlow-free import probe failed")


if __name__ == "__main__":
    main()
