#!/usr/bin/env bash
#
# Out-of-tree plugin dependency check (TT-133, docs/plan-engine-plugins.md):
# proves a separate SwiftPM package can depend on TinyTitanLib, implement an
# EnginePlugin (TPP/1), register it with PluginRegistry, and resolve a matching
# install without editing any TinyTitan source file.
#
#   tools/plugin-dependency-check.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE="$ROOT/examples/plugin"

step() { printf '\n== %s ==\n' "$1"; }

step "out-of-tree plugin fixture (path dependency on the working tree)"
(cd "$FIXTURE" && swift build)
(cd "$FIXTURE" && swift run PluginDemo)

step "ok: out-of-tree plugin builds, registers, and resolves"
