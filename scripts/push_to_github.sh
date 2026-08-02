#!/usr/bin/env bash
# Push local Physics-Aligned World2WAM to GitHub.
#
# Setup (pick one, one-time):
#   A) HTTPS token (works on this server; SSH to GitHub is blocked):
#      bash scripts/setup_github_credentials.sh ghp_xxxx
#   B) SSH key (good for other servers / when GitHub SSH is reachable):
#      bash scripts/setup_github_ssh.sh
#      # add printed key at https://github.com/settings/keys
#
# Then anytime:
#   bash scripts/push_to_github.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -d .git ]]; then
  echo "ERROR: .git missing. Run from Physics-Aligned World2WAM with git initialized."
  exit 1
fi

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  git remote set-url origin "https://${GITHUB_TOKEN}@github.com/LirPan/World2WAM.git"
  git push origin main
  git remote set-url origin "https://github.com/LirPan/World2WAM.git"
  echo "Pushed to https://github.com/LirPan/World2WAM"
  exit 0
fi

CRED_FILE="${HOME}/.config/world2wam/git-credentials"
if [[ -f "${CRED_FILE}" ]]; then
  git -C "${REPO_ROOT}" config credential.helper "store --file=${CRED_FILE}"
  git remote set-url origin https://github.com/LirPan/World2WAM.git
  git push origin main
  echo "Pushed via stored HTTPS credentials."
  exit 0
fi

if git remote get-url origin | grep -q '^git@github.com:'; then
  if ssh -T -o BatchMode=yes -o ConnectTimeout=8 git@github.com 2>&1 | grep -qi "successfully authenticated"; then
    git push origin main
    echo "Pushed via SSH."
    exit 0
  fi
fi

echo "GitHub auth not configured. Run once:"
echo "  bash scripts/setup_github_credentials.sh ghp_xxxx   # recommended on this server"
echo "  bash scripts/setup_github_ssh.sh                      # if GitHub SSH works"
exit 1
