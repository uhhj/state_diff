# Phase3.12d-r2 Comparator Correction

The historical r2 environment report compared hidden armed geometry after N
no-action physics steps with the free geometry at zero steps. That test measures
absolute post-reset motion, not condition-specific visible leakage.

Phase3.12d-r2.1 retains the historical report unchanged and replaces only the
hard-gate comparator with same-horizon controls:

- free replicate(N) versus free(N)
- hidden unarmed(N) versus free(N)
- hidden armed(N) versus free(N)
- armed motion(N) versus free motion(N)

No task or physics parameter is changed by r2.1.
