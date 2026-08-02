#!/usr/bin/env bash
# Store GitHub HTTPS credentials once; future git push/pull need no input.
# Usage: bash scripts/setup_github_credentials.sh ghp_xxxxxxxxxxxx
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/setup_github_credentials.sh <GITHUB_TOKEN>"
  echo "Create token: https://github.com/settings/tokens (repo scope)"
  exit 1
fi

TOKEN="$1"
CRED_DIR="${HOME}/.config/world2wam"
CRED_FILE="${CRED_DIR}/git-credentials"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${CRED_DIR}"
chmod 700 "${CRED_DIR}"

# GitHub PAT over HTTPS: use x-access-token as username (works for classic + fine-grained).
printf 'https://x-access-token:%s@github.com\n' "${TOKEN}" > "${CRED_FILE}"
chmod 600 "${CRED_FILE}"

git -C "${REPO_ROOT}" config credential.helper "store --file=${CRED_FILE}"
git -C "${REPO_ROOT}" remote set-url origin https://github.com/LirPan/World2WAM.git

echo "Saved credentials to ${CRED_FILE}"
echo "Remote set to HTTPS. Test with:"
echo "  bash scripts/push_to_github.sh"
