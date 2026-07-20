// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "TurboFieldfare",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
    ],
    products: [
        .library(name: "TurboFieldfare", targets: ["TurboFieldfare"]),
        .executable(name: "TurboFieldfareRepack", targets: ["TurboFieldfareRepack"]),
        .executable(name: "TurboFieldfareCLI", targets: ["TurboFieldfareCLI"]),
        .executable(name: "TurboFieldfareMac", targets: ["TurboFieldfareMac"]),
        .executable(name: "TurboFieldfareDecodeService", targets: ["TurboFieldfareDecodeService"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "TurboFieldfare",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/TurboFieldfare",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "TurboFieldfareRepackCore",
            path: "Sources/TurboFieldfareRepack/Core"
        ),
        .executableTarget(
            name: "TurboFieldfareRepack",
            dependencies: ["TurboFieldfareRepackCore"],
            path: "Sources/TurboFieldfareRepack/Command"
        ),
        .target(
            name: "TurboFieldfareCLICore",
            dependencies: ["TurboFieldfare"],
            path: "Sources/TurboFieldfareCLI",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "TurboFieldfareCLI",
            dependencies: ["TurboFieldfareCLICore"],
            path: "Sources/TurboFieldfareCLI/Command"
        ),
        .target(
            name: "TurboFieldfareAppCore",
            dependencies: ["TurboFieldfare", "TurboFieldfareRepackCore", "TurboFieldfareDecodeProtocol"],
            path: "Sources/TurboFieldfareApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
            ]
        ),
        .target(
            name: "TurboFieldfareMacPresentation",
            dependencies: ["TurboFieldfareAppCore"],
            path: "Sources/TurboFieldfareApp/MacPresentation"
        ),
        .target(
            name: "TurboFieldfareDecodeProtocol",
            path: "Sources/TurboFieldfareDecodeProtocol"
        ),
        .executableTarget(
            name: "TurboFieldfareDecodeService",
            dependencies: ["TurboFieldfareAppCore", "TurboFieldfareDecodeProtocol"],
            path: "Sources/TurboFieldfareDecodeService"
        ),
        .executableTarget(
            name: "TurboFieldfareMac",
            dependencies: ["TurboFieldfareAppCore", "TurboFieldfareMacPresentation"],
            path: "Sources/TurboFieldfareApp/Mac",
            resources: [
                .copy("Resources/turbofieldfare-app-icon.png"),
            ]
        ),
        .target(
            name: "TurboFieldfareValidationSupport",
            dependencies: ["TurboFieldfare"],
            path: "Sources/TurboFieldfareValidation/Support"
        ),
        .testTarget(
            name: "TurboFieldfareTestsCore",
            dependencies: ["TurboFieldfare", "TurboFieldfareValidationSupport", "TurboFieldfareRepackCore", "TurboFieldfareCLICore"],
            path: "Tests/TurboFieldfare/Core"
        ),
        .testTarget(
            name: "TurboFieldfareRepackTests",
            dependencies: ["TurboFieldfareRepackCore"],
            path: "Tests/TurboFieldfareRepack/Core"
        ),
        .testTarget(
            name: "TurboFieldfareAppCoreTests",
            dependencies: ["TurboFieldfareAppCore", "TurboFieldfare", "TurboFieldfareRepackCore", "TurboFieldfareDecodeProtocol"],
            path: "Tests/TurboFieldfareApp/Core"
        ),
        .testTarget(
            name: "TurboFieldfareMacPresentationTests",
            dependencies: ["TurboFieldfareMacPresentation"],
            path: "Tests/TurboFieldfareApp/MacPresentation"
        ),
    ]
)
