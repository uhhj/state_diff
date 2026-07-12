# Phase3.14a Immutable Training Cache

- Verdict: `PASS`
- Root cause: `phase314a_immutable_training_cache_supported`
- Rows: `4256`
- Cache SHA256: `3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8`
- Source rows relinked after staging promotion: `4256`

| Split | Rows | Visible seeds |
|---|---:|---:|
| `train` | 2440 | 256 |
| `val` | 608 | 64 |
| `test` | 1208 | 128 |

- Future padding fraction: `0.30439380`
- No model training or candidate execution was run during cache construction.
- Phase4 and CPS remain blocked.
