# Run in a normal PowerShell terminal. SSH reads your password directly.
# This script only reads infrastructure metadata; it does not read patient data.
$ErrorActionPreference = 'Stop'
$remote = @'
set -eu
umask 077
printf 'HOST '; hostname
printf 'USER '; id -un
P=/projects/icumultimodal
A=/projects/icumultimodalarchive
for d in "$P" "$A"; do
  if [ ! -d "$d" ]; then printf 'MISSING %s\n' "$d"; exit 2; fi
  printf 'PROJECT '; readlink -f "$d"
  stat -c 'PERMISSIONS %a %U %G' "$d"
  test -r "$d" && test -w "$d" || exit 3
done
case "$(readlink -f "$P")" in /projects/_ssd/*) ;; *) printf 'SSD TIER MISMATCH\n'; exit 4;; esac
case "$(readlink -f "$A")" in /projects/_hdd/*) ;; *) printf 'HDD TIER MISMATCH\n'; exit 4;; esac
printf 'Filesystem capacity (not project quota):\n'
df -h "$P" "$A"
for name in TMPDIR PIP_CACHE_DIR CONDA_PKGS_DIRS HF_HOME TRITON_CACHE_DIR XDG_CACHE_HOME TORCH_HOME; do
  value="${!name-}"
  case "$value" in "$P"/.tmp|"$P"/.tmp/*) printf 'CACHE %s SSD_CONFIGURED\n' "$name";; '') printf 'CACHE %s UNSET\n' "$name";; *) printf 'CACHE %s NEEDS_REVIEW\n' "$name";; esac
done
for d in code envs raw results .tmp; do
  if [ -d "$P/$d" ]; then printf 'EXISTS %s\n' "$d"; else printf 'ABSENT %s\n' "$d"; fi
done
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
  stat -c 'KAGGLE_FILE_PERMISSIONS %a' "$HOME/.kaggle/kaggle.json"
else
  printf 'KAGGLE_FILE_ABSENT\n'
fi
printf 'ACCOUNT\n'
sacctmgr show assoc user="$USER" format=Account,QOS%40,DefaultQOS
printf 'MY_JOBS\n'
squeue --me
printf 'AVAILABLE_NODES\n'
sinfo -o '%P %a %l %D %t %G'
printf 'MONTHLY_ACCOUNTING_BASELINE\n'
sshare -U -nP -o GrpTRESMins,GrpTRESRaw
printf 'PREFLIGHT_COMPLETE (project quota still needs storagemgr verification)\n'
'@
$remote = $remote.Replace("`r`n", "`n")
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
# Encoding is solely to preserve script quoting across Windows SSH.
& ssh -o ServerAliveInterval=60 -o ConnectTimeout=15 shreyas010@10.97.216.128 "printf %s $encoded | base64 -d | bash"
if ($LASTEXITCODE -ne 0) { throw "Preflight stopped with SSH/remote exit code $LASTEXITCODE. Do not retry by changing cluster rules." }
