#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/.build/release"
APP_DIR="$ROOT_DIR/.build/TurboFieldfare.app"
MACOS_DIR="$APP_DIR/Contents/MacOS"
RESOURCES_DIR="$APP_DIR/Contents/Resources"
ICON_SOURCE="$ROOT_DIR/Sources/TurboFieldfareApp/Mac/Resources/turbofieldfare-app-icon.png"
ICON_FILE="$RESOURCES_DIR/TurboFieldfare.icns"

ICONSET_TMP="$(mktemp -d "${TMPDIR:-/tmp}/turbofieldfare-icon.XXXXXX")"
ICONSET_DIR="${ICONSET_TMP}.iconset"
mv "$ICONSET_TMP" "$ICONSET_DIR"

cleanup() {
    rm -rf "$ICONSET_DIR"
}
trap cleanup EXIT

INSTALL=false

if [[ "${1:-}" == "--install" ]]; then
    INSTALL=true
elif [[ $# -gt 0 ]]; then
    printf 'Usage: %s [--install]\n' "$0"
    exit 1
fi

cd "$ROOT_DIR"

swift build -c release

rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

cp "$BUILD_DIR/TurboFieldfareMac" "$MACOS_DIR/TurboFieldfareMac"
cp "$BUILD_DIR/TurboFieldfareDecodeService" "$MACOS_DIR/TurboFieldfareDecodeService"
cp "$BUILD_DIR/TurboFieldfareCLI" "$MACOS_DIR/TurboFieldfareCLI"
cp "$BUILD_DIR/TurboFieldfareRepack" "$MACOS_DIR/TurboFieldfareRepack"
cp "$BUILD_DIR/TurboFieldfareServer" "$MACOS_DIR/TurboFieldfareServer"
cp -R "$BUILD_DIR"/*.bundle "$RESOURCES_DIR/"

for size in 16 32 128 256 512; do
    double_size=$((size * 2))
    sips -z "$size" "$size" "$ICON_SOURCE" --out "$ICONSET_DIR/icon_${size}x${size}.png" >/dev/null
    sips -z "$double_size" "$double_size" "$ICON_SOURCE" \
        --out "$ICONSET_DIR/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET_DIR" -o "$ICON_FILE"

plutil -create xml1 "$APP_DIR/Contents/Info.plist"
plutil -insert CFBundleDisplayName -string TurboFieldfare "$APP_DIR/Contents/Info.plist"
plutil -insert CFBundleName -string TurboFieldfare "$APP_DIR/Contents/Info.plist"
plutil -insert CFBundleIdentifier -string com.turbofieldfare.app "$APP_DIR/Contents/Info.plist"
plutil -insert CFBundleExecutable -string TurboFieldfareMac "$APP_DIR/Contents/Info.plist"
plutil -insert CFBundleIconFile -string TurboFieldfare.icns "$APP_DIR/Contents/Info.plist"
plutil -insert CFBundlePackageType -string APPL "$APP_DIR/Contents/Info.plist"
plutil -insert CFBundleSignature -string '????' "$APP_DIR/Contents/Info.plist"
plutil -insert CFBundleShortVersionString -string 0.7.2 "$APP_DIR/Contents/Info.plist"
plutil -insert CFBundleVersion -string 0.7.2 "$APP_DIR/Contents/Info.plist"
plutil -insert LSMinimumSystemVersion -string 26.0 "$APP_DIR/Contents/Info.plist"
plutil -insert NSHighResolutionCapable -bool true "$APP_DIR/Contents/Info.plist"
plutil -insert NSPrincipalClass -string NSApplication "$APP_DIR/Contents/Info.plist"
codesign --force --deep --sign - "$APP_DIR"
codesign --verify --deep --strict --verbose=2 "$APP_DIR"
printf 'Packaged %s\n' "$APP_DIR"

if $INSTALL; then
    INSTALL_DIR="/Applications"
    INSTALL_APP="$INSTALL_DIR/TurboFieldfare.app"
    rm -rf "$INSTALL_APP"
    ditto "$APP_DIR" "$INSTALL_APP"
    printf 'Installed %s\n' "$INSTALL_APP"
fi
