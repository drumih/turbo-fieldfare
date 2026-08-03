#!/usr/bin/env bash
# Install the TurboFieldfare Mac app into /Applications.
#
# Usage:
#   Scripts/install-app.sh [--force] [--no-codesign] [--version <shortVersion>]
#
# The app is built (via Scripts/build-app.sh), codesigned with an ad-hoc
# signature unless --no-codesign is given, and installed to /Applications.
# An already-installed copy blocks the install unless --force is passed.

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
application_dir="/Applications"
app_name="TurboFieldfare.app"
destination="$application_dir/$app_name"

force=false
no_codesign=false
version=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)      force=true;      shift ;;
    --no-codesign) no_codesign=true; shift ;;
    --version)
      if [[ $# -lt 2 ]]; then
        echo "--version requires a value" >&2
        exit 1
      fi
      version="$2"
      shift 2
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 1
      ;;
  esac
done

build_args=()
if [[ "$no_codesign" == false ]]; then
  build_args+=(--codesign)
fi
if [[ -n "$version" ]]; then
  build_args+=(--version "$version")
fi

source_bundle="$(cd -- "$script_directory/.." && pwd)/.build/release/$app_name"

echo "▶ Building release bundle…"
"$script_directory/build-app.sh" "${build_args[@]}"

if [[ -d "$destination" && "$force" == false ]]; then
  existing_id="$(plutil -extract CFBundleIdentifier raw "$destination/Contents/Info.plist" 2>/dev/null || echo "")"
  echo "⛔ $destination already exists (bundle id: ${existing_id:-unknown})." >&2
  echo "   Run with --force to replace it." >&2
  exit 1
fi

if pgrep -f "$app_name" >/dev/null; then
  echo "⛔ $app_name is currently running; quit it before installing." >&2
  exit 1
fi

[[ "$force" == true ]] && echo "▶ Replacing existing app in $application_dir…"
rm -rf "$destination"
cp -R "$source_bundle" "$destination"

codesign --verify --deep "$destination"
echo "✔ Installed to $destination"
