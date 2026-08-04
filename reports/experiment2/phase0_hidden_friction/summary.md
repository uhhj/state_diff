# Phase 0A Hidden-Friction Cable Pilot

- Groups: 3
- Seeds: 70001..70003
- Config hash: `8d4fc91b932d717549d82474f518ca53ff30052c720cd866883941437279b0dd`
- Main repository: `c7d73045752bf90b6143055a4eb11429b67b5d92`
- Submodule: `39238061cd98124db210c5483c014f27d53ee563`
- Command: `python scripts/experiment2/phase0/run_hidden_friction_pairs.py --config configs/experiment2/phase0/hidden_friction_cable.json --output reports/experiment2/phase0_hidden_friction --groups 3`
- Status: pilot evidence only; no statistical or formal CCDA claim.
- Visual inspection: PASS for `pair_hf_070001_overview.png` and `pair_hf_070001.gif`; fixed axes, no default privileged-region overlay, zero free resistance, positive hidden resistance, and a visible main-pull branch.

Observed problems:

- The proposed seed `70000` placed both outward endpoint pulls beyond the configured workspace, so the runner correctly rejected it. The pilot start was minimally adjusted to `70001`; action semantics and distances were unchanged.
- Independent threaded resets are not bit-exact despite identical seeds: the largest initial XY mismatch was `0.00007031 m` across these three pairs.
- The largest preload-end coordinate mismatch was `0.02000544 m`; this pilot records it without treating three groups as a statistical or formal gate result.
- The server's older Matplotlib required the compatible two-call tick-label API; generated data were unaffected.
