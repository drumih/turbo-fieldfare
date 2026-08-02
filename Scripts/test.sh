#!/usr/bin/env bash
# Serial test runner. Shared Metal state makes in-process parallel tests
# unreliable. Pass any extra arguments through, for example --filter.

if [[ "${1:-}" == "--package-path" ]]; then
  shift 2
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory/.."

# SwiftPM does not add the developer Frameworks directory to the compiler's
# import search paths when the active toolchain is Command Line Tools. Tests
# use the system Testing framework, so supply the path for both compilation
# and linking without hard-coding a particular Xcode installation.
developer_directory="$(xcode-select -p)"
frameworks_directory="$developer_directory/Library/Developer/Frameworks"
developer_library_directory="$developer_directory/Library/Developer/usr/lib"
testing_flags=()
if [[ -d "$frameworks_directory/Testing.framework" ]]; then
  testing_flags=(
    -Xswiftc -F -Xswiftc "$frameworks_directory"
    -Xlinker -F -Xlinker "$frameworks_directory"
    -Xlinker -framework -Xlinker Testing
    -Xlinker -rpath -Xlinker "$frameworks_directory"
    -Xlinker -rpath -Xlinker "$developer_library_directory"
  )
fi

#Preview macros are an Xcode design-time feature. Command Line Tools ships the
# SwiftUI declaration but not the PreviewsMacros plugin, while `swift test`
# still builds the Mac executable in DEBUG. Keep previews available to app
# builds and omit only their DEBUG declarations for package tests.
preview_flags=(-Xswiftc -D -Xswiftc TURBOFIELDFARE_NO_PREVIEWS)

exec swift test --no-parallel "${testing_flags[@]}" "${preview_flags[@]}" "$@"
