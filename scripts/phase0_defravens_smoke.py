#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFRAVENS_ROOT = Path(os.environ.get("DEFRAVENS_ROOT", ROOT / "external" / "deformable-ravens")).resolve()

if not DEFRAVENS_ROOT.exists():
    raise FileNotFoundError(f"DeformableRavens not found: {DEFRAVENS_ROOT}")

sys.path.insert(0, str(DEFRAVENS_ROOT))
os.chdir(DEFRAVENS_ROOT)

result = {
    "repo_root": str(ROOT),
    "defravens_root": str(DEFRAVENS_ROOT),
    "python": sys.version,
}

try:
    import pybullet as p
    result["pybullet_import"] = True
    result["pybullet_version"] = getattr(p, "__version__", "unknown")
except Exception as exc:
    result["pybullet_import"] = False
    result["pybullet_error"] = repr(exc)
    raise

try:
    import tensorflow as tf
    result["tensorflow_import"] = True
    result["tensorflow_version"] = tf.__version__
    result["tensorflow_gpus"] = [str(x) for x in tf.config.list_physical_devices("GPU")]
except Exception as exc:
    result["tensorflow_import"] = False
    result["tensorflow_error"] = repr(exc)
    raise

try:
    import ravens
    from ravens import tasks, Dataset, Environment
    result["ravens_import"] = True
    result["num_tasks"] = len(tasks.names)
    result["has_cable_line_notarget"] = "cable-line-notarget" in tasks.names
    result["cable_tasks"] = sorted([name for name in tasks.names if name.startswith("cable")])
except Exception as exc:
    result["ravens_import"] = False
    result["ravens_error"] = repr(exc)
    raise

assert result["has_cable_line_notarget"], "Task cable-line-notarget is missing from ravens.tasks.names"

print(json.dumps(result, indent=2, sort_keys=True))
