#!/usr/bin/env python3
"""Runtime environment guard for Phase3.

Phase3 uses two different environments:
  - defravens37: DeformableRavens data generation / optional rollout environment.
  - coord_bimanual: PyTorch training and offline evaluation environment.

This script prevents accidental NumPy fallback when the user intended to run
the PyTorch StateDiff-style baseline.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def try_import(name):
    try:
        mod = __import__(name)
        return True, getattr(mod, "__version__", "unknown"), None
    except Exception as exc:
        return False, None, repr(exc)


def infer_env_name(py, env_name):
    if env_name:
        return env_name
    parts = Path(py).parts
    if "envs" in parts:
        i = parts.index("envs")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        required=True,
        choices=["data_generation", "torch_train_eval", "rollout", "aggregate"],
    )
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--allow_numpy_fallback", action="store_true")
    parser.add_argument("--write_json", default=None)
    args = parser.parse_args()

    py = sys.executable
    env_name = infer_env_name(py, os.environ.get("CONDA_DEFAULT_ENV", ""))

    torch_ok, torch_version, torch_err = try_import("torch")
    ravens_ok, ravens_version, ravens_err = try_import("ravens")

    info = {
        "role": args.role,
        "python": py,
        "conda_env": env_name,
        "torch_ok": torch_ok,
        "torch_version": torch_version,
        "torch_error": torch_err,
        "ravens_ok": ravens_ok,
        "ravens_version": ravens_version,
        "ravens_error": ravens_err,
        "allow_numpy_fallback": bool(args.allow_numpy_fallback),
    }

    if args.write_json:
        Path(args.write_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_json).write_text(json.dumps(info, indent=2, sort_keys=True))

    print(json.dumps(info, indent=2, sort_keys=True))

    if args.role == "data_generation":
        if not ravens_ok:
            raise SystemExit(
                "[Phase3][ENV FAIL] Data generation requires DeformableRavens/ravens. "
                "Activate defravens37."
            )

    elif args.role == "torch_train_eval":
        if not torch_ok and not args.allow_numpy_fallback:
            raise SystemExit(
                "[Phase3][ENV FAIL] PyTorch is required for Phase3 train/eval. "
                "You are likely running in defravens37. Activate coord_bimanual. "
                "Fallback is disabled by default. Use --allow_numpy_fallback only for pipeline smoke."
            )

    elif args.role == "rollout":
        if not ravens_ok:
            raise SystemExit(
                "[Phase3][ENV FAIL] Policy rollout requires DeformableRavens/ravens. "
                "Activate defravens37, or install DeformableRavens into the runtime explicitly."
            )
        if not torch_ok and not args.allow_numpy_fallback:
            raise SystemExit(
                "[Phase3][ENV FAIL] Policy rollout also needs torch to load Phase3 models. "
                "Either install a compatible torch build into defravens37 for rollout, "
                "or run rollout with an explicit fallback only for smoke."
            )

    elif args.role == "aggregate":
        pass


if __name__ == "__main__":
    main()
