# Phase3.14b DDPM Training

- Verdict: `PASS`
- Root cause: `phase314b_ddpm_training_supported`
- Runs: `12`
- Device: `cuda`
- Selected: `mlp_ddpm / paper_state / seed=31421`
- Validation best-of-8 final Chamfer: `12.06921725`

| Family | Input | Seed | Epoch | K1 | Best-of-8 | Diversity |
|---|---|---:|---:|---:|---:|---:|
| `mlp_ddpm` | `paper_state` | 31421 | 750 | 27.55321291 | 12.06921725 | 28.84365369 |
| `mlp_ddpm` | `paper_state` | 31422 | 750 | 25.20862694 | 12.48567988 | 28.22613502 |
| `mlp_ddpm` | `paper_state` | 31423 | 750 | 27.54107296 | 12.34373040 | 28.27900128 |
| `mlp_ddpm` | `state_action` | 31421 | 700 | 27.97433677 | 12.30857269 | 28.83505566 |
| `mlp_ddpm` | `state_action` | 31422 | 725 | 24.89892000 | 12.20439994 | 27.91812052 |
| `mlp_ddpm` | `state_action` | 31423 | 750 | 27.90849610 | 12.35047865 | 28.59429440 |
| `temporal_unet_ddpm` | `paper_state` | 31421 | 750 | 58.76241355 | 28.71755409 | 62.37500034 |
| `temporal_unet_ddpm` | `paper_state` | 31422 | 750 | 58.94843245 | 29.72054823 | 59.24340943 |
| `temporal_unet_ddpm` | `paper_state` | 31423 | 750 | 56.57164055 | 31.85352387 | 57.38236920 |
| `temporal_unet_ddpm` | `state_action` | 31421 | 750 | 60.55145217 | 34.10842565 | 52.92200710 |
| `temporal_unet_ddpm` | `state_action` | 31422 | 750 | 60.98882744 | 33.43478745 | 58.79145990 |
| `temporal_unet_ddpm` | `state_action` | 31423 | 750 | 58.88581480 | 32.66894378 | 58.72530833 |

- Test split was not used for selection.
- IDM, candidate execution, Phase4, and CPS were not run.
