#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--probe_json", default="reports/phase3_3_rollout_runtime_probe_coord_bimanual_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_3_runtime_fix_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_3_runtime_fix_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    probe = read_json(root / args.probe_json)
    checks = probe.get("checks", {})
    learned_ready = bool(probe.get("learned_rollout_ready", False)) or probe.get("verdict") == "PASS"
    required = [
        "imports_torch",
        "imports_ravens",
        "imports_pybullet",
        "imports_numpy",
        "windows_loadable",
        "checkpoint_files_exist",
        "conditions_match",
        "primary_matches",
        "diagnostic_matches",
        "y_action_dim_14",
        "codec_dim_14",
        "codec_no_camera_config",
    ]
    missing_or_false = [key for key in required if checks.get(key) is not True]
    verdict = "PASS" if learned_ready and not missing_or_false else "FAIL"

    details = probe.get("details", {})
    payload = {
        "verdict": verdict,
        "learned_rollout_ready": learned_ready,
        "missing_or_false": missing_or_false,
        "probe_json": str(root / args.probe_json),
        "probe_python": details.get("python") or probe.get("python"),
        "probe_conda_env": details.get("conda_env") or probe.get("conda_env") or probe.get("conda_default_env"),
        "checks": checks,
        "recommendation": (
            "Runtime is ready for a later rollout smoke, but rollout was not run in Phase3.3."
            if verdict == "PASS"
            else "Runtime is still blocked. Do not run rollout or Phase4."
        ),
    }

    (root / args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.3 Runtime Fix Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- learned_rollout_ready: `{learned_ready}`",
        f"- Python: `{payload['probe_python']}`",
        f"- Conda env: `{payload['probe_conda_env']}`",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for key in required:
        lines.append(f"| `{key}` | `{checks.get(key)}` |")

    lines += [
        "",
        "## Missing / False",
        "",
    ]
    if missing_or_false:
        for key in missing_or_false:
            lines.append(f"- `{key}`")
    else:
        lines.append("- None")

    lines += [
        "",
        "## Interpretation",
        "",
        "- Phase3.3 only fixes runtime readiness.",
        "- No rollout was run.",
        "- No Phase4 or CPS was run.",
        "- If PASS, the next step can be a user-approved rollout smoke with explicit gates.",
    ]
    (root / args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict != "PASS":
        raise SystemExit("[Phase3.3][FAIL] runtime still not ready")


if __name__ == "__main__":
    main()
