# Phase3.9b Geometry IDM Robustness + Loss Ablation Start

- Timestamp: 2026-07-06T20:28:50+08:00
- Main branch: Experiment1
- Main HEAD: 85cd6b578ffee68484c426ac208268fe280c5f71
- origin/Experiment1: 85cd6b578ffee68484c426ac208268fe280c5f71

## Main status

?? checkpoints/
?? reports/phase3_9b_ablation/
?? reports/phase3_9b_ablation_progress.json
?? reports/phase3_9b_ablation_raw_summary.json
?? reports/phase3_9b_ablation_report.md
?? reports/phase3_9b_ablation_summary.json
?? reports/phase3_9b_geometry_ablation_preflight_report.md
?? reports/phase3_9b_geometry_ablation_preflight_summary.json
?? reports/phase3_9b_geometry_ablation_runtime_config.md
?? reports/phase3_9b_geometry_ablation_start_audit.md
?? reports/phase3_9b_no_phase4_confirmation.md
?? scripts/phase3_9b_analyze_ablation_matrix.py
?? scripts/phase3_9b_geometry_ablation_preflight.py
?? scripts/phase3_9b_run_ablation_matrix.py
?? scripts/phase3_9b_run_geometry_ablation.sh

## Submodule

 388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)

## Objective

Run a larger one-step matched-prefix controlled retry and loss ablation for geometry-aware IDM:
- inverse dynamics only
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- checkpoints local-only
