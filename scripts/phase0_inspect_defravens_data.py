#!/usr/bin/env python3
import json
import os
import pickle
import statistics
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
DEFRAVENS_ROOT = Path(os.environ.get("DEFRAVENS_ROOT", ROOT / "external" / "deformable-ravens")).resolve()
TASK = os.environ.get("TASK", "cable-line-notarget")
REPORT_JSON = ROOT / "reports" / "phase0_defravens_summary.json"
REPORT_MD = ROOT / "reports" / "phase0_defravens_setup.md"

def parse_len(path: Path) -> int:
    # Filename format: 000000-4.pkl
    stem = path.stem
    try:
        return int(stem.split("-")[-1])
    except Exception:
        return -1

def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)

def summarize_dataset(base: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": str(base),
        "exists": base.exists(),
    }
    if not base.exists():
        return out

    fields = ["color", "depth", "action", "info", "last_color", "last_depth", "last_info"]
    out["fields"] = {}
    for field in fields:
        fdir = base / field
        files = sorted(fdir.glob("*.pkl")) if fdir.exists() else []
        out["fields"][field] = {
            "exists": fdir.exists(),
            "num_files": len(files),
            "first_file": files[0].name if files else None,
        }

    color_files = sorted((base / "color").glob("*.pkl")) if (base / "color").exists() else []
    lengths: List[int] = [parse_len(p) for p in color_files]
    lengths = [x for x in lengths if x >= 0]

    out["num_episodes"] = len(color_files)
    out["episode_lengths"] = {
        "values": lengths,
        "min": min(lengths) if lengths else None,
        "max": max(lengths) if lengths else None,
        "mean": statistics.mean(lengths) if lengths else None,
    }

    if color_files:
        first = color_files[0]
        ep_len = parse_len(first)
        ep_name = first.name

        sample = {
            "episode_file": ep_name,
            "episode_len": ep_len,
        }

        for field in ["color", "depth", "action", "info", "last_color", "last_depth", "last_info"]:
            p = base / field / ep_name
            if not p.exists():
                sample[field] = {"exists": False}
                continue
            try:
                obj = load_pickle(p)
                item = {"exists": True, "type": type(obj).__name__}
                if hasattr(obj, "shape"):
                    item["shape"] = list(obj.shape)
                    item["dtype"] = str(getattr(obj, "dtype", "unknown"))
                if field in ["action", "info"]:
                    item["len"] = len(obj)
                    if len(obj) > 0:
                        item["first_type"] = type(obj[0]).__name__
                        if isinstance(obj[0], dict):
                            item["first_keys"] = sorted([str(k) for k in obj[0].keys()])
                            if "extras" in obj[0] and isinstance(obj[0]["extras"], dict):
                                item["extras_keys"] = sorted([str(k) for k in obj[0]["extras"].keys()])
                if field == "last_info" and isinstance(obj, dict):
                    item["keys"] = sorted([str(k) for k in obj.keys()])
                    if "extras" in obj and isinstance(obj["extras"], dict):
                        item["extras_keys"] = sorted([str(k) for k in obj["extras"].keys()])
                        item["extras"] = {
                            k: obj["extras"][k]
                            for k in sorted(obj["extras"].keys())
                            if isinstance(obj["extras"][k], (int, float, str, bool))
                        }
                sample[field] = item
            except Exception as exc:
                sample[field] = {"exists": True, "error": repr(exc)}

        out["sample_episode"] = sample

    return out

def main() -> None:
    data_dir = DEFRAVENS_ROOT / "data" / TASK
    goals_dir = DEFRAVENS_ROOT / "goals" / TASK

    summary: Dict[str, Any] = {
        "task": TASK,
        "repo_root": str(ROOT),
        "defravens_root": str(DEFRAVENS_ROOT),
        "data": summarize_dataset(data_dir),
        "goals": summarize_dataset(goals_dir),
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))

    data_eps = summary["data"].get("num_episodes")
    goal_eps = summary["goals"].get("num_episodes")

    md = []
    md.append("# Phase 0 DeformableRavens Setup Report")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(f"- Task: `{TASK}`")
    md.append(f"- DeformableRavens root: `{DEFRAVENS_ROOT}`")
    md.append(f"- Data episodes: `{data_eps}`")
    md.append(f"- Goal episodes: `{goal_eps}`")
    md.append(f"- JSON summary: `{REPORT_JSON}`")
    md.append("")
    md.append("## Pass Criteria")
    md.append("")
    md.append("- `cable-line-notarget` import smoke test passes.")
    md.append("- `data/cable-line-notarget` has 10 smoke demos.")
    md.append("- `goals/cable-line-notarget` has 20 goals.")
    md.append("- Dataset fields include `color`, `depth`, `action`, `info`, `last_color`, `last_depth`, `last_info`.")
    md.append("")
    md.append("## Data Field Counts")
    md.append("")
    for split in ["data", "goals"]:
        md.append(f"### {split}")
        fields = summary[split].get("fields", {})
        for field, info in fields.items():
            md.append(f"- `{field}`: exists={info.get('exists')}, files={info.get('num_files')}, first={info.get('first_file')}")
        md.append("")
    md.append("## Notes")
    md.append("")
    md.append("This is Phase0 only. No hidden contact condition is added here.")
    md.append("Phase1 should fork or subclass the cable task after this original-task smoke test passes.")
    md.append("")

    REPORT_MD.write_text("\n".join(md))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[Phase0] wrote {REPORT_JSON}")
    print(f"[Phase0] wrote {REPORT_MD}")

    if data_eps is None or data_eps < 10:
        raise SystemExit("[Phase0][FAIL] Expected at least 10 data demos.")
    if goal_eps is None or goal_eps < 20:
        raise SystemExit("[Phase0][FAIL] Expected at least 20 goals.")

if __name__ == "__main__":
    main()
