#!/usr/bin/env bash

# Deprecated compatibility entry point; use evaluation.sh --mode video instead.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$REPO_ROOT/evaluation.sh" --mode video "$@"
