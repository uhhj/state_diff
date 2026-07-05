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
    parser.add_argument("--import_probe", default="reports/phase3_3b_tf_free_import_probe_summary.json")
    parser.add_argument("--hazard", default="reports/phase3_3b_rollout_import_hazard_summary.json")
    parser.add_argument("--dryrun", default="reports/phase3_3b_task_env_dryrun_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_3b_runtime_verdict_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_3b_runtime_verdict_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    imp = read_json(root / args.import_probe)
    haz = read_json(root / args.hazard)
    dry = read_json(root / args.dryrun)
    checks = {
        "tf_free_import_probe_pass": imp.get("verdict") == "PASS",
        "rollout_import_hazard_pass": haz.get("verdict") == "PASS",
        "task_env_dryrun_pass": dry.get("verdict") == "PASS",
        "tensorflow_not_loaded": (imp.get("checks") or {}).get("tensorflow_not_loaded") is True and (dry.get("checks") or {}).get("forbidden_modules_not_loaded") is True,
        "task_registered": (imp.get("checks") or {}).get("hidden_contact_task_registered") is True and (dry.get("checks") or {}).get("task_registered") is True,
        "hidden_breakaway_available": (dry.get("checks") or {}).get("task_has_hidden_breakaway_pin") is True,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "verdict": verdict,
        "checks": checks,
        "recommendation": (
            "TensorFlow-free rollout runtime path is viable. Next step may be user-approved Phase3.4 rollout smoke."
            if verdict == "PASS"
            else "Runtime path is still blocked. Do not run rollout or Phase4."
        ),
        "no_rollout_run": True,
        "no_phase4": True,
        "no_cps": True,
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    lines = [
        "# Phase3.3b Runtime Verdict",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Recommendation: {payload['recommendation']}",
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
        "## Scope",
        "",
        "- No rollout was run.",
        "- No Phase4 was run.",
        "- No CPS was run.",
        "- No TensorFlow was installed.",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict != "PASS":
        raise SystemExit("[Phase3.3b][FAIL] runtime verdict failed")


if __name__ == "__main__":
    main()
