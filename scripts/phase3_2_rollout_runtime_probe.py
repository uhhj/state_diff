#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
PRIMARY_HIDDEN = "hidden_breakaway_pin"
DIAGNOSTIC_HIDDEN = "hidden_pin"


def import_status(name: str) -> Dict[str, Any]:
    try:
        mod = __import__(name)
        return {"ok": True, "version": getattr(mod, "__version__", "unknown"), "error": None}
    except Exception as exc:
        return {"ok": False, "version": None, "error": repr(exc)}


def load_windows_meta(path: Path) -> Tuple[Any, Dict[str, Any]]:
    import numpy as np
    data = np.load(path, allow_pickle=True)
    meta: Dict[str, Any] = {}
    if "meta_json" in data:
        raw = data["meta_json"]
        try:
            text = str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0])
            meta = json.loads(text)
        except Exception:
            meta = {}
    return data, meta


def checkpoint_status(root: Path, baselines) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "baselines": {}}
    for baseline in baselines:
        cands = sorted((root / baseline).glob("fold_*_seed_*"))
        complete = []
        for cand in cands:
            required = [cand / "config.json", cand / "state_model.pt", cand / "inverse_dynamics.pt"]
            if all(p.exists() for p in required):
                complete.append(str(cand))
        out["baselines"][baseline] = {"num_complete": len(complete), "first_complete": complete[0] if complete else None}
        if not complete:
            out["ok"] = False
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--conditions", nargs="+", default=REQUIRED_CONDITIONS)
    parser.add_argument("--primary_hidden_condition", default=PRIMARY_HIDDEN)
    parser.add_argument("--diagnostic_hidden_condition", default=DIAGNOSTIC_HIDDEN)
    parser.add_argument("--baselines", nargs="+", default=["paper_state", "state_action"])
    parser.add_argument("--out_json", default="reports/phase3_2_rollout_runtime_probe_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_2_rollout_runtime_probe_report.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "external" / "deformable-ravens"))

    details = {
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "torch": import_status("torch"),
        "ravens": import_status("ravens"),
        "pybullet": import_status("pybullet"),
        "numpy": import_status("numpy"),
    }

    windows_path = Path(args.windows)
    if not windows_path.is_absolute():
        windows_path = root / windows_path
    windows_info: Dict[str, Any] = {"path": str(windows_path), "exists": windows_path.exists(), "ok": False}
    codec_info: Dict[str, Any] = {"ok": False}
    meta: Dict[str, Any] = {}
    if windows_path.exists() and details["numpy"]["ok"]:
        try:
            data, meta = load_windows_meta(windows_path)
            windows_info.update({
                "ok": True,
                "conditions": meta.get("conditions"),
                "primary_hidden_condition": meta.get("primary_hidden_condition"),
                "diagnostic_hidden_condition": meta.get("diagnostic_hidden_condition"),
                "y_action_shape": list(data["y_action"].shape) if "y_action" in data else None,
                "action_dim": int(data["action_dim"]) if "action_dim" in data else None,
            })
            from ccda_phase3.data_io import load_action_codec_from_template
            template_path = Path(str(data["action_template_json_or_pickle_path"]))
            codec = load_action_codec_from_template(template_path)
            codec_info = {"ok": True, "path": str(template_path), "summary": codec.summary()}
        except Exception as exc:
            windows_info["error"] = repr(exc)

    ckpt_root = Path(args.checkpoint_root)
    if not ckpt_root.is_absolute():
        ckpt_root = root / ckpt_root
    checkpoints = checkpoint_status(ckpt_root, args.baselines)

    checks = {
        "imports_torch": details["torch"]["ok"],
        "imports_ravens": details["ravens"]["ok"],
        "imports_pybullet": details["pybullet"]["ok"],
        "imports_numpy": details["numpy"]["ok"],
        "windows_loadable": bool(windows_info.get("ok")),
        "checkpoint_files_exist": bool(checkpoints.get("ok")),
        "conditions_match": windows_info.get("conditions") == REQUIRED_CONDITIONS and list(dict.fromkeys(args.conditions)) == REQUIRED_CONDITIONS,
        "primary_matches": windows_info.get("primary_hidden_condition") == args.primary_hidden_condition == PRIMARY_HIDDEN,
        "diagnostic_matches": windows_info.get("diagnostic_hidden_condition") == args.diagnostic_hidden_condition == DIAGNOSTIC_HIDDEN,
        "y_action_dim_14": windows_info.get("y_action_shape", [None, None])[1] == 14,
        "codec_dim_14": codec_info.get("summary", {}).get("dim") == 14,
        "codec_no_camera_config": codec_info.get("summary", {}).get("num_camera_config_paths") == 0,
    }
    learned_rollout_ready = all(checks.values())
    verdict = "PASS" if learned_rollout_ready else "FAIL"
    payload = {
        "verdict": verdict,
        "learned_rollout_ready": learned_rollout_ready,
        "checks": checks,
        "details": details,
        "windows": windows_info,
        "codec": codec_info,
        "checkpoints": checkpoints,
        "required_conditions": REQUIRED_CONDITIONS,
        "primary_hidden_condition": PRIMARY_HIDDEN,
        "diagnostic_hidden_condition": DIAGNOSTIC_HIDDEN,
        "recommendation": "Ready for gated rollout smoke." if learned_rollout_ready else "Do not run learned rollout in this environment.",
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.2 Rollout Runtime Probe",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Learned rollout ready: `{learned_rollout_ready}`",
        f"- Python: `{sys.executable}`",
        f"- Conda env: `{os.environ.get('CONDA_DEFAULT_ENV', '')}`",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for key, val in checks.items():
        lines.append(f"| `{key}` | `{val}` |")
    lines += [
        "",
        "## Imports",
        "",
        "| Package | OK | Version | Error |",
        "|---|---:|---|---|",
    ]
    for key in ["torch", "ravens", "pybullet", "numpy"]:
        d = details[key]
        err = str(d.get("error")).replace("|", "/")
        lines.append(f"| `{key}` | `{d.get('ok')}` | `{d.get('version')}` | `{err}` |")
    lines += [
        "",
        "## Checkpoints",
        "",
        "| Baseline | Complete checkpoints | First complete |",
        "|---|---:|---|",
    ]
    for baseline, info in checkpoints["baselines"].items():
        lines.append(f"| `{baseline}` | {info['num_complete']} | `{info['first_complete']}` |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- PASS means this Python environment can attempt learned rollout smoke.",
        "- FAIL means do not run learned rollout in this environment.",
        "- Do not use NumPy fallback to bypass missing torch.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not learned_rollout_ready:
        raise SystemExit("[Phase3.2][FAIL] rollout runtime not ready")


if __name__ == "__main__":
    main()
