#!/usr/bin/env bash
# Build the complete TurboFieldfare Mac app bundle.
#
# - Builds all release products (app + decode service library)
# - Packages them into .build/release/TurboFieldfare.app
# - Optionally codesigns the bundle with an ad-hoc signature
#
# Usage:
#   Scripts/build-app.sh [--version <shortVersion>] [--codesign]
#
# With no --version tag set, the version is derived from the most recent
# git tag, falling back to a dev build (last commit short hash) when the
# repository has no tags. Use `--version` to override both.

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_directory/.." && pwd)"
cd "$repo_root"

version=""
codesign_bundle=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      if [[ $# -lt 2 ]]; then
        echo "--version requires a value" >&2
        exit 1
      fi
      version="$2"
      shift 2
      ;;
    --codesign)
      codesign_bundle=true
      shift
      ;;
    --no-codesign)
      codesign_bundle=false
      shift
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Resolve version and build number from git, with dev fallbacks.
# ---------------------------------------------------------------------------
if [[ -z "$version" ]]; then
  if git_tag=$(git describe --tags --abbrev=0 2>/dev/null); then
    version="$git_tag"
  else
    version="dev-$(git rev-parse --short HEAD)"
  fi
fi
build_number="$(git rev-list --count HEAD)"
bundle_version="$version+$build_number"

app_bundle=".build/release/TurboFieldfare.app"
macos_dir="$app_bundle/Contents/MacOS"
resources_dir="$app_bundle/Contents/Resources"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "▶ Building release products…"
swift build -c release
for product in TurboFieldfareMac TurboFieldfareDecodeService; do
  if [[ ! -f ".build/release/$product" ]]; then
    echo "Missing binary: .build/release/$product" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Assemble the bundle
# ---------------------------------------------------------------------------
echo "▶ Assembling app bundle ($version)…"
rm -rf "$app_bundle"
mkdir -p "$macos_dir" "$resources_dir"
cp .build/release/TurboFieldfareMac            "$macos_dir/"
cp .build/release/TurboFieldfareDecodeService  "$macos_dir/"
cp Sources/TurboFieldfareApp/Mac/Resources/turbofieldfare-app-icon.png \
   "$resources_dir/"
printf 'APPL????' > "$app_bundle/Contents/PkgInfo"

cat > "$app_bundle/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>TurboFieldfareMac</string>
    <key>CFBundleIdentifier</key>
    <string>com.turbofieldfare.mac</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>TurboFieldfare</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$version</string>
    <key>CFBundleVersion</key>
    <string>$bundle_version</string>
    <key>CFBundleIconFile</key>
    <string>turbofieldfare-app-icon</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>NSSupportsAutomaticTermination</key>
    <true/>
    <key>NSSupportsSuddenTermination</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# ---------------------------------------------------------------------------
# Optional ad-hoc code signing (avoids Gatekeeper refusal on first launch)
# ---------------------------------------------------------------------------
if [[ "$codesign_bundle" == true ]]; then
  echo "▶ Code signing (adhoc)…"
  codesign --force --deep --sign - "$app_bundle"
fi

plutil -lint "$app_bundle/Contents/Info.plist"
echo "✔ Bundle built: $app_bundle ($version)"
