import Testing
@testable import TurboFieldfareCLICore

@Suite struct CLIArgumentsTests {
    @Test func defaultsUseProductionGenerationValues() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.model == "m.gturbo")
        #expect(arguments.prompt == "hi")
        #expect(arguments.messagesFile == nil)
        #expect(arguments.maxNew == 1_024)
        #expect(arguments.maxContext == 4096)
        #expect(arguments.temperature == 0.2)
        #expect(arguments.topK == 64)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1)
        #expect(arguments.seed == nil)
        #expect(arguments.stops.isEmpty)
        #expect(!arguments.quiet)
        #expect(arguments.expertCacheSlots == 16)
        #expect(arguments.expertCachePolicy == "lfu")
        #expect(arguments.prefillEnabled)
        #expect(arguments.prefillChunkTokens == 128)
        #expect(arguments.rdadvisePolicy == "off")
    }

    @Test func generationOptionsParseAndStopsRepeat() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--max-new", "32", "--max-context", "512",
            "--temperature", "0", "--top-k", "40", "--top-p", "0.95",
            "--repetition-penalty", "1.1", "--seed", "42",
            "--stop", "A", "--stop", "B", "--quiet",
        ])
        #expect(arguments.maxNew == 32)
        #expect(arguments.maxContext == 512)
        #expect(arguments.temperature == 0)
        #expect(arguments.topK == 40)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1.1)
        #expect(arguments.seed == 42)
        #expect(arguments.stops == ["A", "B"])
        #expect(arguments.quiet)
    }

    @Test func topKZeroRequiresTopPToBeDisabled() throws {
        let disabled = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--top-k", "0", "--top-p", "1",
        ])
        #expect(disabled.topK == nil)
        #expect(disabled.topP == 1)

        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "0",
            ])
        }
    }

    @Test func topKAboveKernelLimitRejected() {
        #expect(throws: ArgsError.invalidValue(flag: "--top-k", value: "257")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "257",
            ])
        }
    }

    @Test func helpListsExactlyThePublicOptions() {
        let expected: Set<String> = [
            "--model", "--prompt", "--messages-file", "--max-new", "--max-context",
            "--temperature", "--top-k", "--top-p", "--repetition-penalty",
            "--seed", "--stop", "--quiet",
            "--expert-cache-slots", "--expert-cache-policy",
            "--prefill", "--prefill-chunk-tokens", "--rdadvise",
            "--help",
        ]
        let words = Args.usage.split { $0.isWhitespace || $0 == "(" || $0 == ")" }
        let options = Set(words.map(String.init).filter { $0.hasPrefix("--") })
        #expect(options == expected)
    }

    @Test func unsupportedSelectorsAreRejected() {
        for flag in ["--runtime-profile", "--experiment-id", "-h"] {
            #expect(throws: ArgsError.unknownFlag(flag)) {
                _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", flag])
            }
        }
    }

    @Test func modelAndPromptAreRequired() {
        #expect(throws: ArgsError.requiredMissing("--model")) {
            _ = try Args.parse(["--prompt", "hi"])
        }
        #expect(throws: ArgsError.modeMissing) {
            _ = try Args.parse(["--model", "m.gturbo"])
        }
    }

    @Test func messagesFileSelectsChatMode() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--messages-file", "chat.json",
        ])
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == "chat.json")
    }

    @Test func promptAndMessagesFileAreMutuallyExclusive() {
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--messages-file")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--messages-file", "chat.json",
            ])
        }
    }

    @Test func runtimeOptionsParse() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--expert-cache-slots", "24", "--expert-cache-policy", "lru",
            "--prefill", "off", "--prefill-chunk-tokens", "64",
            "--rdadvise", "adaptive",
        ])
        #expect(arguments.expertCacheSlots == 24)
        #expect(arguments.expertCachePolicy == "lru")
        #expect(!arguments.prefillEnabled)
        #expect(arguments.prefillChunkTokens == 64)
        #expect(arguments.rdadvisePolicy == "adaptive")
    }

    @Test func runtimeOptionsAcceptDocumentedValues() throws {
        for slots in [8, 16, 24, 32] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--expert-cache-slots", "\(slots)",
            ])
            #expect(arguments.expertCacheSlots == slots)
        }
        for policy in ["lfu", "lru"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--expert-cache-policy", policy,
            ])
            #expect(arguments.expertCachePolicy == policy)
        }
        for tokens in [32, 64, 128] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--prefill-chunk-tokens", "\(tokens)",
            ])
            #expect(arguments.prefillChunkTokens == tokens)
        }
        for policy in ["off", "default", "bounded", "adaptive"] {
            let arguments = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--rdadvise", policy,
            ])
            #expect(arguments.rdadvisePolicy == policy)
        }
    }

    @Test func runtimeOptionsRejectUnsupportedValues() {
        for (flag, value) in [
            ("--expert-cache-slots", "7"),
            ("--expert-cache-policy", "fifo"),
            ("--prefill", "yes"),
            ("--prefill-chunk-tokens", "256"),
            ("--rdadvise", "aggressive"),
        ] {
            #expect(throws: ArgsError.invalidValue(flag: flag, value: value)) {
                _ = try Args.parse([
                    "--model", "m.gturbo", "--prompt", "hi", flag, value,
                ])
            }
        }
    }
}
