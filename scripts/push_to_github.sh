#!/usr/bin/env bash
# Push local Physics-Aligned World2WAM to GitHub (requires auth once).
#
# Option A — Personal Access Token (recommended):
#   export GITHUB_TOKEN=ghp_xxxxxxxx
#   bash scripts/push_to_github.sh
#
# Option B — SSH (after ssh-keygen && adding key to GitHub):
#   git remote set-url origin git@github.com:LirPan/World2WAM.git
#   git push origin main
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

echo "No GITHUB_TOKEN set. Push manually:"
echo "  cd \"${REPO_ROOT}\""
echo "  git push origin main"
echo ""
echo "Or set token:"
echo "  export GITHUB_TOKEN=ghp_xxxx"
echo "  bash scripts/push_to_github.sh"
