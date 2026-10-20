# Initial runbook

## Status

SSH key access, project tiers, private target permissions, cache configuration and quotas have been verified. CPU environment installation is complete. Job 215024 passed 22 synthetic tests and pip dependency checks. No clinical benchmark results exist yet.

Active workflow as of 20 October 2026: data acquisition 215029 (queued), L40 GPU smoke 215030 (queued), source discovery 215035 (depends on successful completion of 215029). Additional acquisition/discovery tests run before the corresponding long CPU jobs.

Confirmed assignments: SSD 100 GB, HDD 200 GB. Conservative acquisition plan is 79.8 GB of SSD usage including reserves. GPU smoke requests one L40 for at most five minutes (at most 20 SU at the documented rate). No GPU SU had been consumed at last check.

Local dedicated key path is ~/.ssh/icubenchmark. Commands must use explicit identity selection and BatchMode; never read or print private key contents. For Windows SSH, encode non-secret shell scripts as UTF-8 base64 and pipe to bash -l remotely to preserve quoting and initialize modules. Do not encode secrets into shell command lines.

Code mirror is a committed Git checkout at /projects/icumultimodal/code. Latest synchronized revision when acquisition/discovery were queued: 5b56cc1. Use Git bundles for synchronization; do not manually edit source files on the cluster.

## Infrastructure preflight

Run `scripts/preflight.ps1` in a local PowerShell window, entering the password directly at SSH's prompt. If the host key is new, compare its fingerprint with the cluster Quick Start before accepting. Expected documented fingerprint at review time: `SHA256:AhYlWQBTsN/4GAzYTVTiZzrVhPhcurFMu1sBMglqvdM`.

The script checks exact project links, permissions, filesystem capacity, exported cache configuration, existence of a few standard directories, Kaggle credential-file permissions, user jobs, account/QoS, node availability and monthly accounting. It does not inspect dataset contents or print secrets. Filesystem free space is not the user's project allowance: also verify project quota with storagemgr.

SSH authentication does not persist to later agent commands. For autonomous follow-up, configure a dedicated SSH key from the local machine with the user entering the password in their own terminal. Never put the SSH password in a script, chat, environment variable, or command line.

## Cache configuration

Cache configuration is installed at /projects/icumultimodal/cacheconfig.sh and sourced from .bashrc; its previous content was backed up on SSD. Source the same block explicitly in batch jobs; noninteractive SSH must not be assumed to read .bashrc.

For each batch job, set TMPDIR to a unique directory under `$P/.tmp` using SLURM_JOB_ID. Cleanup must check that the resolved deletion target is inside that scratch directory. No environments or package caches go in HOME or /tmp.

## Data boundary

The user confirmed required access authorization after reviewing the conflict between uploader license labels and the included PhysioNet license. Preserve this provenance caveat. Run all inspection, extraction, report parsing and modeling as Slurm jobs, outputting aggregate diagnostics only. No patient records, paths containing patient identifiers, report fragments, images or per-patient predictions may be returned through AI tool output.
