# Current blocker — cluster connectivity

After the successful acquisition/discovery status check, two SCP attempts to
10.97.216.128 timed out. No source synchronization or cohort submission occurred.
Completed data and discovery outputs are unaffected; the GPU smoke job was
pending at the last successful check.

Local cleanup and candidate-cohort implementation are committed. Reconnect to
NTUSECURE/VPN, verify SSH, sync the clean checkout, and run the expanded synthetic
suite before the cohort job. Do not describe the new cohort tests as passed yet.
