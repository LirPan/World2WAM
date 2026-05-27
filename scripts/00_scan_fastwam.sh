#!/usr/bin/env bash
# Read-only reminder: see notes/code_scan_report.md
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Code scan report: ${ROOT}/notes/code_scan_report.md"
test -f "${ROOT}/notes/code_scan_report.md" && head -n 40 "${ROOT}/notes/code_scan_report.md"
