import Testing
import TinyTitan

@testable import TinyTitanCLICore
import TinyTitanCore

@Suite struct CLIArgumentsTests {
    @Test func defaultsUseProductionGenerationValues() throws {
        let arguments = try Args.parse(["--model", "m.ssdai", "--prompt", "hi"])
        #expect(arguments.model == "m.ssdai")
        #expect(arguments.prompt == "hi")
        #expect(arguments.messagesFile == nil)
        #expect(arguments.maxNew == 1_024)
        #expect(arguments.maxContext == RuntimeConfiguration.nativeMaximumContextTokens)
        #expect(arguments.temperature == 0.6)
        #expect(arguments.topK == 20)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1)
        #expect(arguments.seed == nil)
        #expect(arguments.stops.isEmpty)
        #expect(!arguments.quiet)
        #expect(arguments.prefillChunk == nil)
        #expect(arguments.kvCachePrecision == .int8)
        #expect(arguments.ropeScalingMode == .none)
        #expect(arguments.thinkingMode == .off)
    }

    @Test func kvPrecisionAndYaRNOptionsParse() throws {
        let yarn = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi",
            "--kv-bits", "4", "--rope-scaling", "yarn",
        ])
        #expect(yarn.kvCachePrecision == .int4)
        #expect(yarn.ropeScalingMode == .yarn)
        #expect(yarn.maxContext == 1_048_576)
        let halfMillion = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi",
            "--rope-scaling", "yarn", "--max-context", "524288",
        ])
        #expect(halfMillion.maxContext == 524_288)
        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.ssdai", "--prompt", "hi",
                "--rope-scaling", "yarn", "--max-context", "262144",
            ])
        }
    }

    @Test func prefillChunkParsesFixedAndAutoValues() throws {
        let fixed = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi", "--prefill-chunk", "4096",
        ])
        #expect(fixed.prefillChunk == .fixed(4_096))

        let automatic = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi", "--prefill-chunk", "auto",
        ])
        #expect(automatic.prefillChunk == .auto)

        #expect(throws: ArgsError.invalidValue(flag: "--prefill-chunk", value: "8192")) {
            _ = try Args.parse([
                "--model", "m.ssdai", "--prompt", "hi", "--prefill-chunk", "8192",
            ])
        }
    }

    @Test func generationOptionsParseAndStopsRepeat() throws {
        let arguments = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi",
            "--max-new", "32", "--max-context", "512",
            "--temperature", "0", "--top-k", "40", "--top-p", "0.95",
            "--repetition-penalty", "1.1", "--presence-penalty", "1.5",
            "--seed", "42",
            "--stop", "A", "--stop", "B", "--quiet",
        ])
        #expect(arguments.maxNew == 32)
        #expect(arguments.maxContext == 512)
        #expect(arguments.temperature == 0)
        #expect(arguments.topK == 40)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1.1)
        #expect(arguments.presencePenalty == 1.5)
        #expect(arguments.presencePenaltyWasSet)
        #expect(arguments.seed == 42)
        #expect(arguments.stops == ["A", "B"])
        #expect(arguments.quiet)
    }

    @Test func presencePenaltyIsBoundedToTheOpenAIRange() throws {
        // Qwen3.8's instruct row is 1.5, so the flag has to accept it; outside
        // OpenAI's -2...2 is refused rather than clamped.
        let accepted = try Args.parse([
            "--model", "m", "--prompt", "x",
            "--presence-penalty", "-0.5",
        ])
        #expect(accepted.presencePenalty == -0.5)
        #expect(accepted.presencePenaltyWasSet)
        #expect(throws: ArgsError.self) {
            try Args.parse(["--model", "m", "--prompt", "x", "--presence-penalty", "2.5"])
        }
    }

    @Test func contextArgumentAcceptsQwenMaximumAndRejectsLargerValues() throws {
        let maximum = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi",
            "--max-context", "262144",
        ])
        #expect(maximum.maxContext == 262_144)
        #expect(throws: ArgsError.invalidValue(flag: "--max-context", value: "262145")) {
            _ = try Args.parse([
                "--model", "m.ssdai", "--prompt", "hi",
                "--max-context", "262145",
            ])
        }
    }

    @Test func topKZeroRequiresTopPToBeDisabled() throws {
        let disabled = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi",
            "--top-k", "0", "--top-p", "1",
        ])
        #expect(disabled.topK == nil)
        #expect(disabled.topP == 1)

        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.ssdai", "--prompt", "hi", "--top-k", "0",
            ])
        }
    }

    @Test func topKAboveKernelLimitRejected() {
        #expect(throws: ArgsError.invalidValue(flag: "--top-k", value: "257")) {
            _ = try Args.parse([
                "--model", "m.ssdai", "--prompt", "hi", "--top-k", "257",
            ])
        }
    }

    @Test func conciseFlagParsesAndDefaultsOff() throws {
        let on = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi", "--concise",
        ])
        #expect(on.concise)
        let off = try Args.parse(["--model", "m.ssdai", "--prompt", "hi"])
        #expect(!off.concise)
    }

    @Test func thinkingModeParsesOnlyTheOfficialBinaryValues() throws {
        let on = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi", "--thinking", "on",
        ])
        #expect(on.thinkingMode == .on)
        #expect(throws: ArgsError.invalidValue(flag: "--thinking", value: "medium")) {
            _ = try Args.parse([
                "--model", "m.ssdai", "--prompt", "hi", "--thinking", "medium",
            ])
        }
    }

    @Test func reasoningEffortParsesOnlyTheTemplateLevels() throws {
        let parsed = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi",
            "--thinking", "on", "--reasoning-effort", "low",
        ])
        #expect(parsed.reasoningEffort == .low)
        let unset = try Args.parse(["--model", "m.ssdai", "--prompt", "hi"])
        #expect(unset.reasoningEffort == nil)
        // The flag parses the union of the templates' levels; "high" is
        // Qwen3.8's invalid one but K2-Horizon's default, and the family is
        // checked against the installed manifest later.
        let k2Level = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi",
            "--thinking", "on", "--reasoning-effort", "high",
        ])
        #expect(k2Level.reasoningEffort == .high)
    }

    @Test func reasoningEffortRequiresThinkingOn() {
        #expect(
            throws: ArgsError.invalidValue(
                flag: "--reasoning-effort", value: "low requires --thinking on")
        ) {
            _ = try Args.parse([
                "--model", "m.ssdai", "--prompt", "hi",
                "--reasoning-effort", "low",
            ])
        }
    }

    @Test func helpListsExactlyThePublicOptions() {
        let expected: Set<String> = [
            "--model", "--prompt", "--messages-file", "--max-new", "--max-context",
            "--temperature", "--top-k", "--top-p", "--repetition-penalty",
            "--presence-penalty",
            "--seed", "--stop", "--quiet", "--help",
            "--rdadvise", "--expert-cache-slots", "--ram-budget", "--prefill-chunk",
            "--concise",
            "--kv-bits", "--rope-scaling", "--thinking", "--reasoning-effort",
            "--calibration-profile", "--check", "--out",
            "--json", "--models-dir",
            "--markdown", "--registry",
        ]
        // The usage text also spells the `calibrate` subcommand's flags, so
        // the bracket and pipe delimiters around them are part of the scan.
        let words = Args.usage.split {
            $0.isWhitespace || "()[]|".contains($0)
        }
        let options = Set(words.map(String.init).filter { $0.hasPrefix("--") })
        #expect(options == expected)
    }

    @Test func unsupportedSelectorsAreRejected() {
        for flag in ["--runtime-profile", "--experiment-id", "-h"] {
            #expect(throws: ArgsError.unknownFlag(flag)) {
                _ = try Args.parse(["--model", "m.ssdai", "--prompt", "hi", flag])
            }
        }
    }

    @Test func modelAndPromptAreRequired() {
        #expect(throws: ArgsError.requiredMissing("--model")) {
            _ = try Args.parse(["--prompt", "hi"])
        }
        #expect(throws: ArgsError.modeMissing) {
            _ = try Args.parse(["--model", "m.ssdai"])
        }
    }

    @Test func messagesFileSelectsChatMode() throws {
        let arguments = try Args.parse([
            "--model", "m.ssdai", "--messages-file", "chat.json",
        ])
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == "chat.json")
    }

    @Test func promptAndMessagesFileAreMutuallyExclusive() {
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--messages-file")) {
            _ = try Args.parse([
                "--model", "m.ssdai", "--prompt", "hi",
                "--messages-file", "chat.json",
            ])
        }
    }

    /// `--ram-budget` names the whole process, so the parser keeps the number
    /// and the loader derives the cache from it. Resolving it here would
    /// duplicate the loader's arithmetic and turn the `ram target` line into a
    /// cache-budget line, which is the one thing the flag exists to avoid.
    @Test func ramBudgetParsesAsATarget() throws {
        let arguments = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi", "--ram-budget", "12G",
        ])
        #expect(arguments.ramBudgetBytes == 12 << 30)
        #expect(arguments.expertCacheSlots == nil)
    }

    /// The same floor the server refuses below: a streaming install holds its
    /// weights plus the 8-slot minimum cache, so 1-3G is a target the process
    /// cannot honour.
    @Test func ramBudgetRefusesTargetsBelowTheFloor() {
        for tiny in ["1G", "2G", "3G", "3221225472"] {
            #expect(throws: ArgsError.invalidValue(flag: "--ram-budget", value: tiny)) {
                _ = try Args.parse([
                    "--model", "m.ssdai", "--prompt", "hi", "--ram-budget", tiny,
                ])
            }
        }
        #expect(throws: ArgsError.invalidValue(flag: "--ram-budget", value: "nope")) {
            _ = try Args.parse([
                "--model", "m.ssdai", "--prompt", "hi", "--ram-budget", "nope",
            ])
        }
    }

    /// Both flags reach the engine so the loader can apply its precedence and
    /// report the gap as the override's, not the install's.
    @Test func ramBudgetAndExplicitSlotsBothSurviveTheParse() throws {
        let arguments = try Args.parse([
            "--model", "m.ssdai", "--prompt", "hi",
            "--ram-budget", "12G", "--expert-cache-slots", "16",
        ])
        #expect(arguments.ramBudgetBytes == 12 << 30)
        #expect(arguments.expertCacheSlots == 16)
    }
}
