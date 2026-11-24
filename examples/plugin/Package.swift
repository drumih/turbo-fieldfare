// swift-tools-version: 6.4
import PackageDescription

// An out-of-tree engine plugin consumer package (TT-133, TPP/1).
//
// Proves that a separate SwiftPM package outside TinyTitan can depend on
// `TinyTitanLib`, implement `EnginePlugin` in its own library target, register
// it with `PluginRegistry`, and resolve a matching install without modifying
// any TinyTitan source file. `tools/plugin-dependency-check.sh` builds and runs
// this package in CI.
let package = Package(
    name: "PluginDemo",
    platforms: [
        .macOS(.v26)
    ],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .target(
            name: "EchoPlugin",
            dependencies: [
                .product(name: "TinyTitanLib", package: "TinyTitan")
            ]
        ),
        .executableTarget(
            name: "PluginDemo",
            dependencies: [
                "EchoPlugin",
                .product(name: "TinyTitanLib", package: "TinyTitan"),
            ]
        ),
    ]
)
