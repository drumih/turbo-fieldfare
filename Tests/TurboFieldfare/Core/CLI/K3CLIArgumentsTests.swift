import Testing
@testable import TurboFieldfareCLICore

/// Argument parsing for the K3 (.gturbo v2) CLI flags.
@Suite struct K3CLIArgumentsTests {
    @Test func k3FlagsDefaultToProductionValues() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.reasoningEffort == nil)
        #expect(!arguments.noThinking)
        #expect(arguments.prefill == "chunked")
        #expect(arguments.prefillChunk == 32)
        #expect(arguments.expertPredict)
        #expect(arguments.expertIOWorkers == "auto")
        #expect(arguments.expertIOSplits == 1)
        #expect(arguments.expertIOCache == "auto")
        #expect(arguments.modelVerification == "full-sha256")
        #expect(!arguments.k3ActivationDiagnostics)
        #expect(!arguments.verbose)
    }

    @Test func k3FlagsParse() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--messages-file", "chat.json",
            "--reasoning-effort", "high", "--no-thinking",
            "--prefill", "serial", "--prefill-chunk", "64",
            "--expert-predict", "off", "--expert-io-workers", "8",
            "--expert-io-splits", "4",
            "--expert-io-cache", "uncached",
            "--model-verification", "trusted-install",
            "--k3-activation-diagnostics", "--verbose",
        ])
        #expect(arguments.reasoningEffort == "high")
        #expect(arguments.noThinking)
        #expect(arguments.prefill == "serial")
        #expect(arguments.prefillChunk == 64)
        #expect(!arguments.expertPredict)
        #expect(arguments.expertIOWorkers == "8")
        #expect(arguments.expertIOSplits == 4)
        #expect(arguments.expertIOCache == "uncached")
        #expect(arguments.modelVerification == "trusted-install")
        #expect(arguments.k3ActivationDiagnostics)
        #expect(arguments.verbose)
    }

    @Test func reasoningEffortAcceptsOnlyK3Values() throws {
        for value in ["low", "high", "max"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--reasoning-effort", value,
            ])
            #expect(arguments.reasoningEffort == value)
        }
        for value in ["medium", "none", "LOW"] {
            #expect(throws: ArgsError.invalidValue(flag: "--reasoning-effort", value: value)) {
                _ = try Args.parse([
                    "--model", "m.gturbo", "--prompt", "hi", "--reasoning-effort", value,
                ])
            }
        }
    }

    @Test func prefillAcceptsSerialOrChunked() throws {
        for value in ["serial", "chunked"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--prefill", value,
            ])
            #expect(arguments.prefill == value)
        }
        #expect(throws: ArgsError.invalidValue(flag: "--prefill", value: "fast")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--prefill", "fast",
            ])
        }
    }

    @Test func prefillChunkMustBeSupportedGeometry() throws {
        for value in ["32", "64", "128", "256"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--prefill-chunk", value,
            ])
            #expect(arguments.prefillChunk == Int(value))
        }
        for value in ["0", "100", "512"] {
            #expect(throws: ArgsError.invalidValue(flag: "--prefill-chunk", value: value)) {
                _ = try Args.parse([
                    "--model", "m.gturbo", "--prompt", "hi", "--prefill-chunk", value,
                ])
            }
        }
    }

    @Test func expertPredictTakesOnOrOff() throws {
        let on = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--expert-predict", "on",
        ])
        #expect(on.expertPredict)
        let off = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--expert-predict", "off",
        ])
        #expect(!off.expertPredict)
        #expect(throws: ArgsError.invalidValue(flag: "--expert-predict", value: "maybe")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--expert-predict", "maybe",
            ])
        }
    }

    @Test func expertIOWorkersAcceptsAutoOrBoundedCount() throws {
        for value in ["auto", "1", "8", "32"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--expert-io-workers", value,
            ])
            #expect(arguments.expertIOWorkers == value)
        }
        for value in ["0", "33", "fast"] {
            #expect(throws: ArgsError.invalidValue(
                flag: "--expert-io-workers", value: value)) {
                _ = try Args.parse([
                    "--model", "m.gturbo", "--prompt", "hi",
                    "--expert-io-workers", value,
                ])
            }
        }
    }

    @Test func expertIOSplitsAcceptOnlyMeasuredGeometries() throws {
        for value in ["1", "2", "4", "8"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--expert-io-splits", value,
            ])
            #expect(arguments.expertIOSplits == Int(value))
        }
        for value in ["0", "3", "16", "auto"] {
            #expect(throws: ArgsError.invalidValue(
                flag: "--expert-io-splits", value: value)) {
                _ = try Args.parse([
                    "--model", "m.gturbo", "--prompt", "hi",
                    "--expert-io-splits", value,
                ])
            }
        }
    }

    @Test func modelVerificationRequiresExplicitKnownPolicy() throws {
        for value in ["full-sha256", "trusted-install"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--model-verification", value,
            ])
            #expect(arguments.modelVerification == value)
        }
        #expect(throws: ArgsError.invalidValue(
            flag: "--model-verification", value: "none")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--model-verification", "none",
            ])
        }
    }

    @Test func expertIOCacheAcceptsAutoBufferedOrUncached() throws {
        for value in ["auto", "buffered", "uncached"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--expert-io-cache", value,
            ])
            #expect(arguments.expertIOCache == value)
        }
        #expect(throws: ArgsError.invalidValue(
            flag: "--expert-io-cache", value: "direct")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--expert-io-cache", "direct",
            ])
        }
    }

    @Test func k3FlagsRequireValues() {
        for flag in ["--reasoning-effort", "--prefill", "--prefill-chunk",
                     "--expert-predict", "--expert-io-workers",
                     "--expert-io-splits", "--expert-io-cache",
                     "--model-verification"] {
            #expect(throws: ArgsError.missingValue(flag: flag)) {
                _ = try Args.parse([
                    "--model", "m.gturbo", "--prompt", "hi", flag,
                ])
            }
        }
    }
}
