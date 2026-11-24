// swift-tools-version: 6.4
import PackageDescription

/// The language standard this package is written to.
///
/// Swift 6 language mode is set below (`swiftLanguageModes: [.v6]`); these are
/// the upcoming features that are not yet default in that mode and that the
/// tree is clean under. Enforced here rather than documented, so a target
/// added later cannot quietly opt out. The ones deliberately *not* adopted
/// (and why, with their measured diagnostic counts) are recorded in
/// `docs/swift-language-standard.md`.
///
/// `-warnings-as-errors` is part of the standard, not a preference: a warning
/// that only appears in a build log is a check nobody runs, and the release
/// script's log scan did not cover `swift test` at all. Every target carries
/// this array (25 of 25 at the time of writing), so the flag cannot be dodged
/// by a new target either. The tree builds and tests clean with it
/// (`swift build --build-tests`, `swift test --no-parallel`).
let tinytitanLanguageStandard: [SwiftSetting] = [
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("ImmutableWeakCaptures"),
    .enableUpcomingFeature("MemberImportVisibility"),
    .unsafeFlags(["-warnings-as-errors"]),
]

let package = Package(
    name: "TinyTitan",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        // Product 1 of 2: the library. `TinyTitanLib` is the supported surface
        // another Swift program depends on; `TinyTitan`, `TinyTitanFormat` and
        // `ContinuityCore` are its building blocks and stay targets rather than
        // products, so there is exactly one library to promise anything about.
        // See `docs/plan-embedded-library.md`.
        //
        // The product is static, which is the right default for a source
        // consumer: nothing has to be embedded beside their binary. The dynamic
        // form below is the same library, and it exists because a release ships
        // both (`tools/build_library.sh` links it with an install name of
        // `@rpath/libTinyTitanLib.dylib` so the published file can carry the
        // plain name).
        // `TinyTitanVariants` is the engine-factory surface: it selects the
        // 7.0 variant for an install (or the 6.x engine where none ships) and
        // is in the product so an embedder and the CLI can both construct an
        // engine without importing the server. `TinyTitanLib` itself still
        // imports no engine module; the product is the bundle.
        .library(
            name: "TinyTitanLib",
            targets: [
                "TinyTitanLib", "TinyTitanLegacyEngine", "TinyTitanVariants", "TinyTitanCPUEngine",
                "TinyTitanPluginAPI",
            ]),
        .library(
            name: "TinyTitanLibDynamic", type: .dynamic,
            targets: [
                "TinyTitanLib", "TinyTitanLegacyEngine", "TinyTitanVariants", "TinyTitanCPUEngine",
                "TinyTitanPluginAPI",
            ]),
        // Each engine plugin is a library of its own (TPP/1): built and tested
        // standalone, compiled into the products above.
        .library(name: "TemplatePlugin", targets: ["TemplatePlugin"]),
        // Product 2 of 2: the engine, embedding the library above.
        .executable(name: "TinyTitanRepack", targets: ["TinyTitanRepack"]),
        .executable(name: "TinyTitanCLI", targets: ["TinyTitanCLI"]),
        .executable(name: "TinyTitanServer", targets: ["TinyTitanServer"]),
        .executable(name: "TinyTitanBench", targets: ["TinyTitanBench"]),
        .executable(name: "ContinuityDemo", targets: ["ContinuityDemo"]),
        .executable(name: "tinytitan-memory", targets: ["TinyTitanMemoryTool"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/ibireme/yyjson.git", exact: "0.12.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.100.0"),
    ],
    targets: [
        .target(
            name: "TinyTitanFormat",
            path: "sources/TinyTitanFormat",
            swiftSettings: tinytitanLanguageStandard
        ),
        // C99 + NEON for the inner loops where Swift's vector types do not
        // lower well. Kept deliberately small: one file, one entry point,
        // covered by the same tests as the Swift path it replaced.
        //
        // `-O2` for these kernels, which is *not* the build system's default:
        // SwiftPM's `swiftbuild` system compiles every C target at `-Os` in
        // release (the older native planner used `-O2`), and the difference on
        // the CPU int8 GEMV is 1.24x -- 2.06 against 1.66 ms per pass, min of
        // six interleaved rounds an arm, checksum identical
        // (390266.62). `.unsafeFlags` is the only way to set it, and it does
        // *not* stop this package being consumed as a dependency: measured on
        // 2026-10-02 in Swift 6.4, a package that depends on the released tag
        // resolves, builds and links, and `tools/embedded-dependency-check.sh`
        // builds `examples/embedded` to keep that property from rotting. The
        // toolchain still prints its "contains unsafe build flags" diagnostic
        // for the consumer, which is noise, not a refusal. Raising Swift to
        // `-O3` was tried and rejected, having measured the same as the release
        // default (0.675 vs 0.680 ms).
        //
        // The language standard and the hardening warnings are enforced here,
        // not merely declared: `-std=c99` comes from `cLanguageStandard` above,
        // and these settings add the flags `-Wall -Wextra` does not imply plus
        // `-Werror`, so a new kernel cannot land with a shadowed variable, a
        // narrowing conversion, a dropped qualifier, a non-literal format or a
        // missing prototype. All three C files compile clean under the full set
        // (proof recorded when the flags landed, AUD-003), and `-pedantic-errors` rejects the
        // implicit declarations and GNU extensions C99 does not have.
        .target(
            name: "TinyTitanKernelsC",
            path: "sources/TinyTitanKernelsC",
            cSettings: [
                .unsafeFlags([
                    "-O2",
                    "-pedantic-errors",
                    "-Wall", "-Wextra",
                    "-Wshadow", "-Wconversion", "-Wsign-conversion", "-Wcast-qual",
                    "-Wwrite-strings", "-Wformat=2", "-Wstrict-prototypes",
                    "-Wmissing-prototypes",
                    "-Werror",
                ])
            ],
            swiftSettings: tinytitanLanguageStandard,
            linkerSettings: [
                // The batched CPU prefill's GEMM wrapper in dequant_affine.c.
                .linkedFramework("Accelerate")
            ]
        ),
        // The generic library the 7.0 variants share: bounded reads, concurrency,
        // hardware and the tool-call parsers. Model knowledge stays out —
        // docs/plan-tinytitan-7-engine.md §1.3 — and the core-purity gate enforces
        // it. The tokenizer cluster stays in `TinyTitan` until its internal
        // GF* coupling is split (tracker row TT-073 records the finding).
        .target(
            name: "TinyTitanCore",
            dependencies: [
                "TinyTitanFormat",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "yyjson", package: "yyjson"),
            ],
            path: "sources/TinyTitanCore",
            resources: [.copy("Metal")],
            swiftSettings: tinytitanLanguageStandard
        ),
        // The first 7.0 engine variant (plan task 5). A variant target imports
        // only the generic library — the variant-isolation gate enforces it.
        .target(
            name: "Qwen35Dense2B4Bit",
            dependencies: ["TinyTitanCore", "TinyTitanFormat", "TinyTitanKernelsC"],
            path: "sources/Engines/Qwen35Dense2B4Bit",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "Qwen35Dense2B8Bit",
            dependencies: ["TinyTitanCore", "TinyTitanFormat", "TinyTitanKernelsC"],
            path: "sources/Engines/Qwen35Dense2B8Bit",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "Qwen35Dense4B4Bit",
            dependencies: ["TinyTitanCore", "TinyTitanFormat", "TinyTitanKernelsC"],
            path: "sources/Engines/Qwen35Dense4B4Bit",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "Qwen35Dense4B8Bit",
            dependencies: ["TinyTitanCore", "TinyTitanFormat", "TinyTitanKernelsC"],
            path: "sources/Engines/Qwen35Dense4B8Bit",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "Qwen35Dense9B4Bit",
            dependencies: ["TinyTitanCore", "TinyTitanFormat", "TinyTitanKernelsC"],
            path: "sources/Engines/Qwen35Dense9B4Bit",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "Qwen35Dense9B8Bit",
            dependencies: ["TinyTitanCore", "TinyTitanFormat", "TinyTitanKernelsC"],
            path: "sources/Engines/Qwen35Dense9B8Bit",
            swiftSettings: tinytitanLanguageStandard
        ),
        // The shared config-driven dense-GPU engine (TT-097 Task 12): one
        // target for the whole dense Qwen 3.5 family, geometry taken from each
        // install's manifest (`numExperts == 0` makes the same graph dense),
        // refusing non-dense installs by name. A spec-sanctioned copy of the
        // shipped variant's engine — the variant-isolation rule forbids a
        // cross-variant import.
        .target(
            name: "DenseGPU",
            dependencies: [
                "TinyTitanCore", "TinyTitanFormat", "TinyTitanKernelsC",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "sources/Engines/DenseGPU",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "Ornith35B4Bit",
            dependencies: [
                "TinyTitanCore", "TinyTitanFormat", "TinyTitanKernelsC",
                .product(name: "Tokenizers", package: "swift-transformers"),
                "TinyTitanPluginAPI",
            ],
            path: "sources/Engines/Ornith35B4Bit",
            swiftSettings: tinytitanLanguageStandard
        ),
        // TT-121 phase 2: Inkling 2 Mini from its Q4_K_XL GGUF, served from a
        // verified `.ssdai` install. The graph is a spec-sanctioned copy of the
        // GGUF route's decoder (variant-isolation rule).
        // TinyTitan Plugin Protocol v1 (docs/plugin-api-v1.md): the promise to
        // a plugin author. Depends on Core only.
        .target(
            name: "TinyTitanPluginAPI",
            dependencies: ["TinyTitanCore"],
            path: "sources/TinyTitanPluginAPI",
            swiftSettings: tinytitanLanguageStandard
        ),
        // The golden base every engine plugin is copied from: a standalone
        // library over the plugin API, compiled in, serving no shipped model.
        .target(
            name: "TemplatePlugin",
            dependencies: ["TinyTitanPluginAPI", "TinyTitanCore"],
            path: "sources/Plugins/TemplatePlugin",
            exclude: ["README.md"],
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "Inkling2GGUFQ4",
            dependencies: ["TinyTitanCore", "TinyTitanFormat", "TinyTitanPluginAPI"],
            path: "sources/Engines/Inkling2GGUFQ4",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitan",
            dependencies: [
                "TinyTitanCore",
                "TinyTitanFormat",
                "TinyTitanKernelsC",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "sources/TinyTitan",
            swiftSettings: tinytitanLanguageStandard
        ),
        // The supported facade an embedder links: the generation orchestrator,
        // prompt shaping and templating, with no NIO and no HTTP. It depends on
        // the engine and the format reader and on nothing else in this package.
        // `TinyTitanServerCore` is rebuilt on top of it, so the facade cannot
        // drift from what the server actually needs.
        .target(
            name: "TinyTitanLib",
            dependencies: ["TinyTitanCore", "TinyTitanFormat"],
            path: "sources/TinyTitanLib",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanLegacyEngine",
            dependencies: ["TinyTitanCore", "TinyTitan", "TinyTitanLib"],
            path: "sources/TinyTitanLegacyEngine",
            swiftSettings: tinytitanLanguageStandard
        ),
        // The factory surface a single-model front end and an embedder use:
        // the 7.0 variant selection table plus the CPU dense selection. It
        // lives outside `TinyTitanServerCore` because both the CLI and the
        // library product need it, and the server must not be the only way to
        // reach a variant engine.
        .target(
            name: "TinyTitanVariants",
            dependencies: [
                "TinyTitanCore", "TinyTitan", "TinyTitanLib", "TinyTitanLegacyEngine",
                "Ornith35B4Bit", "DenseGPU", "Inkling2GGUFQ4",
                "TinyTitanPluginAPI", "TemplatePlugin",
                "Qwen35Dense2B4Bit", "Qwen35Dense2B8Bit",
                "Qwen35Dense4B4Bit", "Qwen35Dense4B8Bit",
                "Qwen35Dense9B4Bit", "Qwen35Dense9B8Bit",
            ],
            path: "sources/TinyTitanVariants",
            swiftSettings: tinytitanLanguageStandard
        ),
        // The CPU dense backend, shared by the server and the library's CPU
        // facade. It needs the 6.x tokenizer, so it is its own target rather
        // than a dependency of `TinyTitanLib`; the library reaches it through
        // an injected backend, never an import.
        .target(
            name: "TinyTitanCPUEngine",
            dependencies: [
                "TinyTitanCore", "TinyTitanLib", "TinyTitan",
                "TinyTitanLegacyEngine", "TinyTitanVariants",
            ],
            path: "sources/TinyTitanCPUEngine",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanRepackCore",
            dependencies: ["TinyTitanCore", "TinyTitanFormat"],
            path: "sources/TinyTitanRepack/Core",
            swiftSettings: tinytitanLanguageStandard
        ),
        .executableTarget(
            name: "TinyTitanRepack",
            dependencies: ["TinyTitanRepackCore"],
            path: "sources/TinyTitanRepack/Command",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanCLICore",
            // The CLI is a front end of the library (phase A2 of
            // `docs/plan-embedded-library.md`), so it depends on the facade it
            // drives. `TinyTitan` stays for the few value types `Args` parses.
            dependencies: [
                "TinyTitanCore", "TinyTitan", "TinyTitanLib", "TinyTitanLegacyEngine",
                "TinyTitanVariants",
                "TinyTitanCPUEngine", "TinyTitanInkling2",
            ],
            path: "sources/TinyTitanCLI",
            exclude: ["Command"],
            swiftSettings: tinytitanLanguageStandard
        ),
        .executableTarget(
            name: "TinyTitanCLI",
            dependencies: ["TinyTitanCLICore"],
            path: "sources/TinyTitanCLI/Command",
            swiftSettings: tinytitanLanguageStandard
        ),
        // Continuity: sessions, task memory and context assembly, in this
        // process. Depends on nothing at all, not even NIO, so it cannot
        // reach the network and cannot be reached from one.
        .target(
            name: "ContinuityCore",
            path: "sources/ContinuityCore",
            // Documentation that lives next to the code it describes. SwiftPM
            // treats any undeclared file under a target path as unhandled and
            // warns on every clean plan; excluding it says so explicitly and
            // leaves the file where it is. (`sources/TinyTitanCLICore`'s
            // `exclude: ["Command"]` is the same mechanism.)
            exclude: ["README.md"],
            swiftSettings: tinytitanLanguageStandard
        ),
        // Worked examples and a scale check for ContinuityCore. Not part of
        // the server; it exists so the package's claims can be run.
        .executableTarget(
            name: "ContinuityDemo",
            dependencies: ["ContinuityCore"],
            path: "sources/ContinuityDemo",
            swiftSettings: tinytitanLanguageStandard
        ),
        // Agent memory: the model-facing surface (keys, tools, prompt
        // fragment, journal filter) over ContinuityCore. Depends on nothing
        // in the engine, so the serving path can use it without the memory
        // subsystem being able to reach back into inference, and on no
        // networking, so it cannot reach off the machine.
        .target(
            name: "TinyTitanMemory",
            dependencies: ["ContinuityCore"],
            path: "sources/TinyTitanMemory",
            swiftSettings: tinytitanLanguageStandard
        ),
        // See and correct what the server remembers: list, show, delete.
        // Reads take no lock; writes need the workspace.
        .executableTarget(
            name: "TinyTitanMemoryTool",
            dependencies: ["TinyTitanMemory", "ContinuityCore"],
            path: "sources/TinyTitanMemoryTool",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanServerCore",
            dependencies: [
                "TinyTitanCore",
                "TinyTitanLib", "TinyTitanPluginAPI",
                "TinyTitan",
                "TinyTitanLegacyEngine",
                "TinyTitanMemory",
                // The variant selection table lives in `TinyTitanVariants`
                // now, so the server, the CLI and the library product share
                // one factory surface. Server→variant imports are allowed by
                // the variant-isolation rule; only a variant may not import
                // the server.
                "TinyTitanVariants",
                // TT-121: the Inkling 2 Q4 variant, served through the Inkling
                // backend until the shared session takes Inkling (phase 4).
                "Inkling2GGUFQ4",
                "TinyTitanCPUEngine", "TinyTitanInkling2",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "sources/TinyTitanServer/Core",
            swiftSettings: tinytitanLanguageStandard
        ),
        .executableTarget(
            name: "TinyTitanServer",
            dependencies: [
                "TinyTitanServerCore", "TinyTitanVariants", "TinyTitanCPUEngine",
                "TinyTitanPluginAPI",
            ],
            path: "sources/TinyTitanServer/Command",
            swiftSettings: tinytitanLanguageStandard
        ),
        .executableTarget(
            name: "TinyTitanBench",
            dependencies: ["TinyTitanCore", "TinyTitan", "TinyTitanLib", "TinyTitanLegacyEngine"],
            path: "sources/TinyTitanBench",
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanValidationSupport",
            dependencies: ["TinyTitan"],
            path: "sources/TinyTitanValidation/Support",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanTests",
            dependencies: [
                "TinyTitanVariants",
                .product(name: "yyjson", package: "yyjson"),
                "Qwen35Dense2B4Bit",
                "Qwen35Dense2B8Bit",
                "Qwen35Dense4B4Bit", "Qwen35Dense4B8Bit",
                "Qwen35Dense9B4Bit", "Qwen35Dense9B8Bit",
                "TinyTitanCore",
                "TinyTitan", "TinyTitanFormat", "TinyTitanValidationSupport",
                "TinyTitanRepackCore", "TinyTitanCLICore", "TinyTitanLib",
            ],
            path: "tests/TinyTitan",
            resources: [
                .copy("Tokenization/Fixtures"),
                .copy("Runtime/qwen38_tensor_names.txt"),
                .copy("Runtime/ple_golden.json"),
                .copy("Infrastructure/ModelIO/GGUFDequantVectors.txt"),
                .copy("Infrastructure/ModelIO/GGUFTokenizerVectors.json"),
                .copy("Inkling2GGUF/Inkling2GGUFReferences"),
            ],
            swiftSettings: tinytitanLanguageStandard
        ),
        .target(
            name: "TinyTitanInkling2",
            dependencies: ["TinyTitanCore"],
            path: "sources/Engines/Inkling2",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanRepackTests",
            // `TinyTitanFormat` directly: the manifest and resident-index validation
            // tests assert on those types rather than on JSON dictionaries.
            dependencies: [
                "TinyTitanRepackCore", "TinyTitanFormat",
                .product(name: "yyjson", package: "yyjson"),
            ],
            path: "tests/TinyTitanRepack/Core",
            resources: [.copy("Support/qwen38_tensor_names.txt")],
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanInkling2Tests",
            dependencies: ["TinyTitanInkling2"],
            path: "tests/TinyTitanInkling2",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "ContinuityCoreTests",
            dependencies: ["ContinuityCore", .product(name: "yyjson", package: "yyjson")],
            path: "tests/ContinuityCore",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanCoreTests",
            dependencies: ["TinyTitanCore", .product(name: "yyjson", package: "yyjson")],
            path: "tests/TinyTitanCore",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "Qwen35Dense2B4BitTests",
            dependencies: [
                "Qwen35Dense2B4Bit", "TinyTitanCore", .product(name: "yyjson", package: "yyjson"),
            ],
            path: "tests/Engines/Qwen35Dense2B4Bit",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "Ornith35B4BitTests",
            dependencies: [
                "Ornith35B4Bit", "TinyTitan", "TinyTitanCore",
                .product(name: "yyjson", package: "yyjson"),
            ],
            path: "tests/Engines/Ornith35B4Bit",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanPluginAPITests",
            dependencies: ["TinyTitanPluginAPI", "TinyTitanCore"],
            path: "tests/TinyTitanPluginAPI",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TemplatePluginTests",
            dependencies: ["TemplatePlugin", "TinyTitanPluginAPI", "TinyTitanCore"],
            path: "tests/Plugins/TemplatePlugin",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "Inkling2GGUFQ4Tests",
            dependencies: ["Inkling2GGUFQ4", "TinyTitan", "TinyTitanCore", "TinyTitanPluginAPI"],
            path: "tests/Engines/Inkling2GGUFQ4",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "DenseGPUTests",
            dependencies: [
                "DenseGPU", "TinyTitanCore", "TinyTitan",
                .product(name: "yyjson", package: "yyjson"),
            ],
            path: "tests/Engines/DenseGPU",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanMemoryTests",
            dependencies: [
                "TinyTitanMemory", "ContinuityCore", .product(name: "yyjson", package: "yyjson"),
            ],
            path: "tests/TinyTitanMemory",
            swiftSettings: tinytitanLanguageStandard
        ),
        .testTarget(
            name: "TinyTitanServerTests",
            dependencies: [
                .product(name: "yyjson", package: "yyjson"),
                "TinyTitanCore",
                "TinyTitanServerCore",
                "TinyTitanPluginAPI",
                "TinyTitanLib",
                "TinyTitanMemory",
                // `GenerationDefaults.Sampling`, so the mapper tests can pin
                // that an omitted field follows the served model's profile
                // rather than a hardcoded house default.
                "TinyTitan",
                // The factory surface the selection tests pin; its own
                // target names the variant modules, so the tests import one
                // module instead of eight.
                "TinyTitanVariants",
                "TinyTitanCPUEngine",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "tests/TinyTitanServer",
            resources: [.copy("Fixtures")],
            swiftSettings: tinytitanLanguageStandard
        ),
        // The library's own contract: what the facade promises an embedder,
        // tested against the kit rather than through the server that consumes
        // it. `tests/TinyTitanServer` covers the server's use of the kit; this
        // target is where the facade's own guarantees live.
        .testTarget(
            name: "TinyTitanLibTests",
            dependencies: [
                .product(name: "yyjson", package: "yyjson"),
                "TinyTitanCore", "TinyTitanLib", "TinyTitan", "TinyTitanLegacyEngine",
                // The contract suite builds engines the way an embedder does,
                // through the variant selection table.
                "TinyTitanVariants",
            ],
            path: "tests/TinyTitanLib",
            swiftSettings: tinytitanLanguageStandard
        ),
    ],
    swiftLanguageModes: [.v6],
    // The C in this package is written to strict C99; declaring it here makes
    // the compiler enforce it instead of documenting an intention. Together
    // with the TinyTitanKernelsC cSettings below this is the C language
    // standard in force, and a deliberate violation fails the build (AUD-002/003).
    cLanguageStandard: .c99
)
