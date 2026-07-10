# Phase3.12d-r1 Start

- Objective: environment semantics repair + query-local snapshot candidate oracle
- No training
- No Phase4
- No CPS
- Cross-process dynamic prefix replay is retired
- One query is executed in one worker
- Prefix is executed once per query
- All candidates use PyBullet saveState/restoreState from the same query state

## Preserved Existing Work

The uncommitted Phase3.12d diagnostics and existing runtime artifact directories
were present before Phase3.12d-r1 and are intentionally preserved. They are not
part of this repair unless explicitly staged later.
