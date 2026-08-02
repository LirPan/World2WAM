#!/usr/bin/env bash
# One-time GitHub SSH setup for passwordless git push/pull on this server.
set -euo pipefail

KEY="${HOME}/.ssh/id_ed25519"
PUB="${KEY}.pub"

mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"

if [[ ! -f "${KEY}" ]]; then
  ssh-keygen -t ed25519 -C "$(whoami)@$(hostname)" -f "${KEY}" -N ""
fi

if [[ ! -f "${HOME}/.ssh/config" ]] || ! grep -q "Host github.com" "${HOME}/.ssh/config" 2>/dev/null; then
  cat >> "${HOME}/.ssh/config" <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
EOF
fi

chmod 600 "${HOME}/.ssh/config" "${KEY}"
chmod 644 "${PUB}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git -C "${REPO_ROOT}" remote set-url origin git@github.com:LirPan/World2WAM.git

echo "=== Add this public key to GitHub (one time) ==="
echo "https://github.com/settings/keys → New SSH key"
echo ""
cat "${PUB}"
echo ""
echo "Then test:"
echo "  ssh -T git@github.com"
echo "  bash scripts/push_to_github.sh"
