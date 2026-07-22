#!/usr/bin/env python3
"""Single Stage-L Resume1 deterministic execution entrypoint."""
from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r258_stagel_resume1_execution_recovery import main


if __name__ == "__main__":
    raise SystemExit(main())
