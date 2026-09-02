#!/usr/bin/env bash
set -euo pipefail

ROOT="${WORLD2WAM_SPRINT_ROOT:-/DATA/disk0/yjh/world2wam_iclr2027}"
DEPLOY="$ROOT/deploy"
MARKER="$ROOT/status/bootstrap.complete.json"
MANIFEST="$ROOT/manifests/queue_v2.json"
RUN_ROOT="$ROOT/runs/paper_sprint_v2"
PY=/usr/bin/python3
BOOTSTRAP="$DEPLOY/scripts/bootstrap_iclr2027_new_yjh.sh"
BOOTSTRAP_ATTEMPTS="$ROOT/status/bootstrap_v2_supervisor_attempts"

mkdir -p "$ROOT/manifests" "$RUN_ROOT"
while [[ ! -s "$MARKER" ]]; do
  if ! pgrep -u "$(id -u)" -f "^bash ${BOOTSTRAP}$" >/dev/null 2>&1; then
    attempts=0
    [[ -s "$BOOTSTRAP_ATTEMPTS" ]] && attempts="$(<"$BOOTSTRAP_ATTEMPTS")"
    if (( attempts >= 3 )); then
      printf '[%s] bootstrap exhausted three supervised attempts\n' "$(date -Is)" >&2
      exit 1
    fi
    attempts=$((attempts + 1))
    printf '%s\n' "$attempts" > "$BOOTSTRAP_ATTEMPTS"
    printf '[%s] restarting bootstrap attempt %s/3\n' "$(date -Is)" "$attempts"
    bash "$BOOTSTRAP" >> "$ROOT/logs/bootstrap_v2.log" 2>&1 &
  fi
  printf '[%s] waiting for verified bootstrap marker\n' "$(date -Is)"
  sleep 30
done

"$PY" - "$MARKER" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
required = {
    "complete": True,
    "fasterwam_commit": "83667817df0d4f823f39d90700e61ea2f432ac45",
    "libero_plus_commit": "4976dc3",
}
for key, expected in required.items():
    if payload.get(key) != expected:
        raise SystemExit(f"invalid bootstrap marker: {key}={payload.get(key)!r}")
PY

"$PY" "$DEPLOY/scripts/build_iclr2027_manifest.py" \
  --root "$ROOT" \
  --protocol "$DEPLOY/protocols/iclr2027_paper_sprint.json" \
  --output "$MANIFEST"
"$PY" "$DEPLOY/scripts/paper_sprint.py" plan \
  --manifest "$MANIFEST" --run-root "$RUN_ROOT"
exec "$PY" "$DEPLOY/scripts/paper_sprint.py" run \
  --manifest "$MANIFEST" --run-root "$RUN_ROOT"
