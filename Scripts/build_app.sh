#!/bin/bash
set -euo pipefail
SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
exec ruby "$SCRIPT_DIRECTORY/build_app.rb" "$@"
