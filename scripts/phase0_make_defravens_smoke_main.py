#!/usr/bin/env python3
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFRAVENS_ROOT = Path(os.environ.get("DEFRAVENS_ROOT", ROOT / "external" / "deformable-ravens")).resolve()

src = DEFRAVENS_ROOT / "main.py"
dst = DEFRAVENS_ROOT / "main_phase0_smoke.py"

if not src.exists():
    raise FileNotFoundError(src)

text = src.read_text()
new_text, n = re.subn(
    r"MAX_ORDER\s*=\s*\d+",
    "MAX_ORDER = 1  # Phase0 smoke: 10 demos",
    text,
    count=1,
)

if n != 1:
    raise RuntimeError("Could not patch MAX_ORDER in main.py")

dst.write_text(new_text)
print(f"[Phase0] wrote smoke main: {dst}")
print("[Phase0] MAX_ORDER=1 means 10 demos.")
