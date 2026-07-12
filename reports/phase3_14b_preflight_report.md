# Phase3.14b Preflight

- Verdict: `PASS`
- Root cause: `phase314b_preflight_supported`
- Cache SHA256: `3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8`
- CUDA available: `True`
- Fixed validation rows: `102`

| Model | Parameters | Output |
|---|---:|---|
| `mlp_ddpm:paper_state` | 1081692 | `[2, 4, 87]` |
| `temporal_unet_ddpm:paper_state` | 3624151 | `[2, 4, 87]` |
| `mlp_ddpm:state_action` | 1103196 | `[2, 4, 87]` |
| `temporal_unet_ddpm:state_action` | 3753175 | `[2, 4, 87]` |

- No DDPM training was run by preflight.
- IDM, candidate execution, Phase4, and CPS remain blocked.
