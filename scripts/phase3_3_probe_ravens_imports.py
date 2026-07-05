#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

CHECK_MODULES = [
    "numpy",
    "torch",
    "pybullet",
    "meshcat",
    "ravens",
    "ravens.tasks",
]


def check_import(name):
    try:
        mod = importlib.import_module(name)
        return {
            "ok": True,
            "version": getattr(mod, "__version__", "unknown"),
            "error": None,
            "traceback": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "version": None,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--out_json", default="reports/phase3_3_import_probe_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_3_import_probe_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    defravens = root / "external/deformable-ravens"
    if str(defravens) not in sys.path:
        sys.path.insert(0, str(defravens))

    results = {name: check_import(name) for name in CHECK_MODULES}
    payload = {
        "python": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "root": str(root),
        "deformable_ravens_path": str(defravens),
        "sys_path_contains_deformable_ravens": str(defravens) in sys.path,
        "results": results,
        "all_required_ok": all(results[m]["ok"] for m in ["numpy", "torch", "pybullet", "ravens", "ravens.tasks"]),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.3 Ravens Import Probe",
        "",
        f"- Python: `{payload['python']}`",
        f"- Conda env: `{payload['conda_default_env']}`",
        f"- DeformableRavens path injected: `{payload['sys_path_contains_deformable_ravens']}`",
        f"- all_required_ok: `{payload['all_required_ok']}`",
        "",
        "## Import Results",
        "",
        "| Module | OK | Version | Error |",
        "|---|---:|---|---|",
    ]
    for name, result in results.items():
        err = result["error"] if result["error"] else ""
        lines.append(f"| `{name}` | `{result['ok']}` | `{result['version']}` | `{err}` |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- This probe does not run rollout.",
        "- If `ravens` fails because of missing `meshcat`, install meshcat in `coord_bimanual` only.",
        "- Do not modify torch or numpy.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if not payload["all_required_ok"]:
        raise SystemExit("[Phase3.3][WARN] import probe not fully ready")


if __name__ == "__main__":
    main()
