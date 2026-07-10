# Phase3.12d-r2.1 Runtime Configuration

- Conda environment: `coord_bimanual`
- Conditions: `free hidden_breakaway_pin hidden_high_friction`
- Visible seeds: `312000 312001 312002 312003 312500 312501 312502 312503`
- Evaluation steps: `0 1 5 20`
- Raw DDPM samples/query: `8`

## Corrected Audit Thresholds

- free reproducibility max/MAE: `1e-7 / 1e-8`
- common evolution max/MAE: `1e-6 / 1e-7`
- paired visible max/MAE: `1e-4 / 1e-5`
- fraction difference: `1e-9`
- curve difference: `1e-5`
- controlled divergence: `0.003`
- free goal actionable dense gain: `0.003`

No physical task parameter or submodule file was modified by r2.1.
