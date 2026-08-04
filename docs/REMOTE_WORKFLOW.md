# Experiment2 Remote Workflow

Server access is blocked until the cloud-provider console or another trusted out-of-band channel authenticates the server host-key fingerprints. A network `ssh-keyscan` alone is not identity proof.

After trusted confirmation, compare freshly collected keys with the approved fingerprints, store only exact matching public keys, and configure `ccda-gpu` with strict host-key checking and the dedicated `id_ed25519_ccda_experiment2` identity. Never use `sshpass`, embedded passwords, or `StrictHostKeyChecking=no`.

Before cloning, inspect `/data`, `/workspace`, and `/root`, confirm that the selected Experiment2 path does not exist, and do not touch any Experiment1 directory. Prefer `/data/state_diff_experiment2`; use `/root/state_diff_experiment2` only when necessary.

Clone the published branch recursively:

```bash
git clone --branch Experiment2 --single-branch --recurse-submodules \
  https://github.com/uhhj/state_diff.git /data/state_diff_experiment2
```

Then compare the remote superproject and submodule commits against the published local evidence and run `bash scripts/remote/verify_remote.sh`. Do not create or modify Conda environments during bootstrap; inventory the existing simulation environment only.
