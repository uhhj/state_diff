# Phase3.2 Runtime Blocked Next Step

Phase3.2 rollout code hardening was completed, but learned rollout smoke was not run because no current environment passed runtime readiness.

Required in one environment:
- torch
- ravens
- pybullet
- numpy
- Phase3 checkpoints
- Phase3 windows

Probe results:
- coord_bimanual: torch OK, pybullet OK, windows/checkpoints OK, ravens import FAIL due missing meshcat.
- defravens37: ravens OK, pybullet OK, windows/checkpoints OK, torch import FAIL.

Do not use NumPy fallback. Next options:
1. Add torch to defravens37, or
2. Add ravens/PyBullet runtime dependencies to coord_bimanual, or
3. Build a dedicated rollout environment cloned from defravens37 with torch installed.

No learned rollout / Phase4 / CPS evidence has been produced.
