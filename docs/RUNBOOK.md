# Initial runbook

## Status

No batch jobs, downloads, environment creation or benchmark runs have occurred. Environment versions will be pinned before installation after checking the available modules and package compatibility.

## Infrastructure preflight

Run `scripts/preflight.ps1` in a local PowerShell window, entering the password directly at SSH's prompt. If the host key is new, compare its fingerprint with the cluster Quick Start before accepting. Expected documented fingerprint at review time: `SHA256:AhYlWQBTsN/4GAzYTVTiZzrVhPhcurFMu1sBMglqvdM`.

The script checks exact project links, permissions, filesystem capacity, exported cache configuration, existence of a few standard directories, Kaggle credential-file permissions, user jobs, account/QoS, node availability and monthly accounting. It does not inspect dataset contents or print secrets. Filesystem free space is not the user's project allowance: also verify project quota with storagemgr.

SSH authentication does not persist to later agent commands. For autonomous follow-up, configure a dedicated SSH key from the local machine with the user entering the password in their own terminal. Never put the SSH password in a script, chat, environment variable, or command line.

## Cache configuration

After confirming the SSD target, install `scripts/cacheblock.sh` under the SSD code directory and source its absolute path from `.bashrc`. Preserve existing `.bashrc` content and make the addition idempotent. This has not been performed. Source the same block explicitly in batch jobs; noninteractive SSH must not be assumed to read `.bashrc`.

For each batch job, set TMPDIR to a unique directory under `$P/.tmp` using SLURM_JOB_ID. Cleanup must check that the resolved deletion target is inside that scratch directory. No environments or package caches go in HOME or /tmp.

## Data boundary

Resolve dataset provenance and reuse terms before data downloads. Run all inspection, extraction, report parsing and modeling as Slurm jobs, outputting aggregate diagnostics only. No patient records, paths containing patient identifiers, report fragments, images or per-patient predictions may be returned through AI tool output.
