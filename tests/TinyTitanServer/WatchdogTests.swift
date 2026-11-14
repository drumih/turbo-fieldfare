import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan
@testable import TinyTitanLib
@testable import TinyTitanServerCore

/// The four watchdogs, on synthetic streams.
///
/// Every case here is a shape this project actually produced -- a 0.8B
/// looping on one sentence, a port that never emitted a token, forty tokens
/// of empty tool markup where a program belonged, a tool called with the
/// same arguments until the rounds ran out -- or one of the innocent shapes
/// that must never be mistaken for them. The false-positive side is the half
/// that matters: a watchdog that stops a good generation costs the user the
/// whole answer.
@Suite struct WatchdogTests {

    private var observing: WatchdogConfiguration {
        WatchdogConfiguration(isEnabled: true)
    }

    private func feed(
        _ text: String,
        configuration: WatchdogConfiguration
    ) -> WatchdogVerdict {
        var watchdog = LoopWatchdog(configuration: configuration)
        var verdict = WatchdogVerdict.fine
        // One byte at a time: chunk boundaries must not change the answer,
        // and a token is rarely a tidy unit anyway.
        for character in text {
            let result = watchdog.observe(String(character), at: .now)
            if result != .fine {
                verdict = result
                break
            }
        }
        return verdict
    }

    // MARK: loop

    /// The measured failure, reproduced: Qwen 0.8B repeated this sentence
    /// until its 600-token budget ran out.
    ///
    /// The thresholds are the calibrated ones -- a 64-byte window, six
    /// repeats -- so what these tests measure is what will ship.
    private let looped = "Wait, I need to check the facts again. "

    @Test func loopTripsOnARepeatedPhrase() {
        #expect(
            feed(
                String(repeating: looped, count: 8),
                configuration: observing) != .fine)
    }

    /// A model can think in circles until its budget is gone. Since thinking
    /// left the answer channel the answer's detector never sees it, so
    /// reasoning has a loop detector of its own, and says where it tripped.
    @Test func aLoopInReasoningTrips() {
        var set = WatchdogSet(configuration: observing)
        for character in String(repeating: looped, count: 8) {
            set.observeReasoning(String(character))
        }
        #expect(set.trips.map(\.kind) == [.loop])
        #expect(set.trips.first?.message.hasPrefix("in reasoning: ") == true)
    }

    /// An answer that restates its thought is not a loop. Four repeats in
    /// the thought and four in the answer would be eight in one window --
    /// a trip -- so this holds only because the two are watched apart.
    @Test func anAnswerThatRepeatsItsThoughtIsNotALoop() {
        var set = WatchdogSet(configuration: observing)
        let said = String(repeating: looped, count: 4)
        for character in said { set.observeReasoning(String(character)) }
        for character in said { set.observe(String(character)) }
        #expect(set.trips.isEmpty)
    }

    /// A handful of repeats is not a loop. Prose repeats a sentence for
    /// emphasis and code repeats a call, and a threshold that fired on
    /// either would be useless. Four was the plan's proposed threshold and
    /// it fired on 8.1% of the recorded corpus.
    @Test func loopDoesNotTripOnAFewRepeats() {
        for count in 2...4 {
            #expect(
                feed(
                    String(repeating: looped, count: count),
                    configuration: observing) == .fine,
                Comment(rawValue: "\(count) repeats should not be a loop"))
        }
    }

    /// A phrase shorter than the window costs one repetition to warm the
    /// window up: six occurrences of a 64-byte window need five periods plus
    /// a window, which is a seventh repetition of a 39-byte phrase. Worth
    /// pinning, because it is the difference between "repeats six times" as
    /// written and as measured.
    @Test func loopNeedsOneExtraRepetitionForAShortPhrase() {
        #expect(feed(String(repeating: looped, count: 6), configuration: observing) == .fine)
        #expect(feed(String(repeating: looped, count: 7), configuration: observing) != .fine)
    }

    /// Ordinary prose of the same length is untouched.
    @Test func loopIgnoresOrdinaryProse() {
        let prose = """
            The town of Ashgrove sits where the river bends, and the ferry has \
            run on Sundays since the bridge went down. Marcus keeps the ledger \
            in the back room of the inn, which burned in the autumn of the year \
            the photograph was taken, and Rosa has never once agreed with him \
            about what the photograph shows. The road north is closed until the \
            thaw, so nobody has left the valley since the first snow.
            """
        #expect(feed(prose, configuration: observing) == .fine)
    }

    /// B3's worst case: legitimately repetitive code. Thirty near-identical
    /// switch cases are the largest false-positive risk this detector has.
    @Test func loopIgnoresRepetitiveCode() {
        var code = "switch token {\n"
        for index in 0..<30 {
            code +=
                "    case .symbol\(index): return Token(kind: .symbol\(index), "
                + "offset: offset, length: \(index))\n"
        }
        code += "    default: return nil\n}\n"
        #expect(feed(code, configuration: observing) == .fine)
    }

    /// A table rule, a line of dashes and a block of indentation all repeat
    /// perfectly. None is a loop, and the variety test is what says so.
    @Test func loopIgnoresRulesBordersAndIndentation() {
        #expect(
            feed(
                String(repeating: "|---", count: 200),
                configuration: observing) == .fine)
        #expect(
            feed(
                String(repeating: "-", count: 800),
                configuration: observing) == .fine)
        #expect(
            feed(
                String(repeating: " ", count: 800),
                configuration: observing) == .fine)
        #expect(
            feed(
                String(repeating: "= ", count: 400),
                configuration: observing) == .fine)
    }

    /// A phrase that recurs naturally across a long document ages out of the
    /// history instead of accumulating towards a trip.
    @Test func loopForgetsRepeatsOutsideTheHistory() {
        let phrase = "The ferry runs on Sundays and not otherwise at all. "
        let text = (0..<6).map { phrase + Self.variedProse(seed: UInt64($0), bytes: 1_600) }
            .joined()
        #expect(feed(text, configuration: observing) == .fine)
    }

    /// A known limitation, pinned rather than hidden: templated output --
    /// a numbered list where only the numbers change -- repeats its
    /// invariant span exactly, and no local rule can tell that from a loop.
    ///
    /// This is why `loop` is never in the default acting list (B3). The
    /// detector says "something repeated", which is true; whether that is a
    /// failure is a judgement it does not have the information to make.
    @Test func loopCannotTellTemplatedOutputFromALoop() {
        let list = (0..<10).map { index in
            "Chapter \(index): the ledger recorded a crossing on the \(index)th, "
                + "and the weather that week was below what the almanac promised.\n"
        }.joined()
        #expect(
            feed(list, configuration: observing) != .fine,
            "a template trips it; the calibration corpus measures how often")
    }

    /// Prose with a real vocabulary, so a test about one property is not
    /// quietly testing another.
    static func variedProse(seed: UInt64, bytes: Int) -> String {
        let words = [
            "ledger", "crossing", "almanac", "thaw", "ferry", "ashgrove",
            "photograph", "chapter", "inn", "valley", "snow", "river",
            "bridge", "autumn", "marcus", "rosa", "harbour", "lantern",
            "orchard", "mill", "quarry", "shepherd", "meadow", "tide",
        ]
        var state = seed &+ 0x9E37_79B9_7F4A_7C15
        var text = ""
        while text.utf8.count < bytes {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            text += words[Int(state >> 59) % words.count]
            text += (state >> 40) % 9 == 0 ? ".\n" : " "
        }
        return text
    }

    /// B5: the cost per byte must not grow with the output. A detector that
    /// got slower as it watched would tax exactly the long generations most
    /// likely to need watching.
    @Test func loopCostDoesNotGrowWithOutputLength() {
        func nanosPerByte(_ bytes: Int) -> Double {
            var watchdog = LoopWatchdog(configuration: observing)
            // Pseudo-random varied text: no loop to find, so every byte pays
            // the full path.
            var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
            var text = ""
            text.reserveCapacity(bytes)
            for _ in 0..<bytes {
                seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                text.append(Character(UnicodeScalar(UInt8(32 + seed >> 58))))
            }
            let started = ContinuousClock.now
            _ = watchdog.observe(text, at: .now)
            let elapsed = started.duration(to: .now)
            let seconds =
                Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
            return seconds * 1e9 / Double(bytes)
        }
        // The best of several, not one reading. A single timing on a busy
        // machine measures the scheduler: this failed once with a 125B server
        // generating beside the suite, load average 4.9, because the long run
        // happened to be preempted and the short one did not. Contention only
        // ever adds time, so the minimum is the reading closest to the code.
        func bestOf(_ bytes: Int) -> Double {
            (0..<5).map { _ in nanosPerByte(bytes) }.min() ?? .infinity
        }
        _ = nanosPerByte(1 << 12)  // warm the allocator
        let small = bestOf(1 << 13)
        let large = bestOf(1 << 17)  // sixteen times as long
        #expect(
            large < small * 4,
            Comment(rawValue: "per-byte cost grew from \(small) to \(large) ns"))
    }

    // MARK: stall

    /// B1: the clock starts at the first token. A long prefill emits nothing
    /// for minutes -- this project has measured 652 s of it -- and a
    /// watchdog that counted from the request would stop every long prompt.
    @Test func stallIgnoresPrefill() {
        var watchdog = StallWatchdog(
            configuration:
                WatchdogConfiguration(isEnabled: true, stallSeconds: 1))
        let start = ContinuousClock.now
        #expect(watchdog.check(at: start.advanced(by: .seconds(600))) == .fine)
    }

    @Test func stallFiresAfterTheThresholdAndNotBefore() {
        var watchdog = StallWatchdog(
            configuration:
                WatchdogConfiguration(isEnabled: true, stallSeconds: 90))
        let start = ContinuousClock.now
        _ = watchdog.observe("first token", at: start)
        #expect(watchdog.check(at: start.advanced(by: .seconds(89))) == .fine)
        #expect(watchdog.check(at: start.advanced(by: .seconds(91))) != .fine)
    }

    /// The slowest decode this project has measured is 6.7 tok/s. A stream
    /// that is merely slow must never trip.
    @Test func stallIgnoresASlowButProgressingStream() {
        var watchdog = StallWatchdog(
            configuration:
                WatchdogConfiguration(isEnabled: true, stallSeconds: 90))
        var now = ContinuousClock.now
        for _ in 0..<200 {
            _ = watchdog.observe("token ", at: now)
            now = now.advanced(by: .seconds(10))
            #expect(watchdog.check(at: now) == .fine)
        }
    }

    /// It reports once. A stall that persisted would otherwise fill the log
    /// with one line per tick.
    @Test func stallReportsOnce() {
        var watchdog = StallWatchdog(
            configuration:
                WatchdogConfiguration(isEnabled: true, stallSeconds: 10))
        let start = ContinuousClock.now
        _ = watchdog.observe("x", at: start)
        #expect(watchdog.check(at: start.advanced(by: .seconds(20))) != .fine)
        #expect(watchdog.check(at: start.advanced(by: .seconds(40))) == .fine)
    }

    // MARK: stub

    /// Ornith's C99 stage returned forty tokens of empty tool markup where a
    /// program belonged, and the request reported a clean stop.
    @Test func stubFiresOnAnEmptyNormalFinish() {
        var watchdog = StubWatchdog(configuration: observing)
        #expect(
            watchdog.finish(
                visibleBytes: 20, requestBytes: 700,
                finishReason: "stop") != .fine)
    }

    /// The case that found this rule's missing half. On the first request of
    /// the first observation run the harness asked "Say OK." and the model
    /// answered "OK.", and the watchdog called it a failure. A short answer
    /// to a short question is an answer, and no rule that ignores the
    /// question can tell the two apart.
    @Test func stubIgnoresAShortAnswerToAShortQuestion() {
        var watchdog = StubWatchdog(configuration: observing)
        #expect(
            watchdog.finish(
                visibleBytes: 3, requestBytes: 7,
                finishReason: "stop") == .fine)
    }

    /// It is the last user message that counts, not the prompt: an agent
    /// harness puts a long system prompt in front of every question,
    /// including the one-word ones.
    @Test func stubMeasuresWhatWasAskedNotWhatWasSent() {
        var watchdog = StubWatchdog(
            configuration:
                WatchdogConfiguration(isEnabled: true, stubAskedBytes: 200))
        #expect(
            watchdog.finish(
                visibleBytes: 4, requestBytes: 199,
                finishReason: "stop") == .fine)
        #expect(
            watchdog.finish(
                visibleBytes: 4, requestBytes: 200,
                finishReason: "stop") != .fine)
    }

    /// A short reply that ended in a tool call is a normal turn of a tool
    /// loop, not a stub.
    @Test func stubIgnoresAToolCall() {
        var watchdog = StubWatchdog(configuration: observing)
        #expect(
            watchdog.finish(
                visibleBytes: 20, requestBytes: 700,
                finishReason: "tool_calls") == .fine)
    }

    /// A truncated reply already tells the client what happened through
    /// `length`; saying it twice adds nothing.
    @Test func stubIgnoresALengthFinish() {
        var watchdog = StubWatchdog(configuration: observing)
        #expect(
            watchdog.finish(
                visibleBytes: 20, requestBytes: 700,
                finishReason: "length") == .fine)
    }

    @Test func stubIgnoresARealAnswer() {
        var watchdog = StubWatchdog(configuration: observing)
        let answer = String(repeating: "a real answer. ", count: 40)
        #expect(
            watchdog.finish(
                visibleBytes: answer.utf8.count, requestBytes: 700,
                finishReason: "stop") == .fine)
    }

    /// The 151-token book reply is deliberately outside this watchdog's
    /// reach: telling a short answer from a wrong-length one needs to know
    /// what was asked, and that judgement is not this detector's to make.
    @Test func stubDoesNotReachForTheShortAnswerCase() {
        var watchdog = StubWatchdog(configuration: observing)
        #expect(
            watchdog.finish(
                visibleBytes: 600, requestBytes: 700,
                finishReason: "stop") == .fine)
    }

    // MARK: ping-pong

    private func message(callingTool name: String, arguments: String) -> GFTokenizer.Message {
        // The arguments here are literals written by this file. A decode failure
        // is a fixture typo, reported as a test issue so the suite keeps running
        // (and this test fails) instead of taking the process down.
        let decoded = try? JSONDecoder().decode(JSONValue.self, from: Data(arguments.utf8))
        if decoded == nil {
            Issue.record("tool-call arguments did not decode: \(arguments)")
        }
        return GFTokenizer.Message(
            role: .assistant,
            content: nil,
            toolCalls: [
                GFTokenizer.HistoricalToolCall(
                    id: UUID().uuidString, name: name, arguments: decoded ?? .null)
            ])
    }

    @Test func pingPongFiresOnTheSameCallThreeTimes() {
        let messages = (0..<3).map { _ in
            message(callingTool: "read_file", arguments: #"{"path":"main.swift"}"#)
        }
        #expect(PingPongWatchdog.inspect(messages, configuration: observing) != .fine)
    }

    /// A long session legitimately reads the same file more than once. Only
    /// a consecutive run is a loop; identical calls with work in between are
    /// a tool being used.
    @Test func pingPongIgnoresIdenticalCallsWithWorkBetweenThem() {
        let messages = [
            message(callingTool: "read_file", arguments: #"{"path":"main.swift"}"#),
            message(callingTool: "grep", arguments: #"{"q":"publish"}"#),
            message(callingTool: "read_file", arguments: #"{"path":"main.swift"}"#),
            message(callingTool: "write_file", arguments: #"{"path":"main.swift"}"#),
            message(callingTool: "read_file", arguments: #"{"path":"main.swift"}"#),
        ]
        #expect(PingPongWatchdog.inspect(messages, configuration: observing) == .fine)
    }

    /// A user turn between identical calls means the conversation moved on.
    /// Counting only within the tool-call subsequence would call three reads
    /// in three separate turns a loop.
    @Test func pingPongRunIsBrokenByAUserTurn() {
        var messages: [GFTokenizer.Message] = []
        for _ in 0..<3 {
            messages.append(
                message(
                    callingTool: "read_file",
                    arguments: #"{"path":"main.swift"}"#))
            messages.append(GFTokenizer.Message(role: .user, content: "and now?"))
        }
        #expect(PingPongWatchdog.inspect(messages, configuration: observing) == .fine)
    }

    /// A tool result does not break the run: it is the answer to the call,
    /// and the model asking again identically right after it is the failure.
    @Test func pingPongSurvivesTheToolResultsBetweenCalls() {
        var messages: [GFTokenizer.Message] = []
        for _ in 0..<3 {
            messages.append(
                message(
                    callingTool: "read_file",
                    arguments: #"{"path":"main.swift"}"#))
            messages.append(
                GFTokenizer.Message(
                    role: .tool, content: "not found",
                    toolCallID: "x"))
        }
        #expect(PingPongWatchdog.inspect(messages, configuration: observing) != .fine)
    }

    /// The same tool with different arguments is a tool being used.
    @Test func pingPongIgnoresDifferentArguments() {
        let messages = (0..<6).map { index in
            message(callingTool: "read_file", arguments: #"{"path":"file\#(index).swift"}"#)
        }
        #expect(PingPongWatchdog.inspect(messages, configuration: observing) == .fine)
    }

    /// Argument order is the client's choice, not the model's: two calls
    /// that differ only in key order are the same call.
    @Test func pingPongSeesThroughKeyOrder() {
        let messages = [
            message(callingTool: "grep", arguments: #"{"a":1,"b":2}"#),
            message(callingTool: "grep", arguments: #"{"b":2,"a":1}"#),
            message(callingTool: "grep", arguments: #"{"a":1,"b":2}"#),
        ]
        #expect(PingPongWatchdog.inspect(messages, configuration: observing) != .fine)
    }

    @Test func pingPongIsSilentWhenWatchdogsAreOff() {
        let messages = (0..<9).map { _ in
            message(callingTool: "read_file", arguments: #"{"path":"main.swift"}"#)
        }
        #expect(PingPongWatchdog.inspect(messages, configuration: .off) == .fine)
    }

    // MARK: the set

    /// Off is off: no detector is consulted and nothing is recorded, which
    /// is what makes shipping this disabled by default meaningful.
    @Test func disabledSetSeesNothing() {
        var set = WatchdogSet(configuration: .off)
        set.observe(String(repeating: "Wait, I need to check again. ", count: 40))
        set.finish(visibleBytes: 0, requestBytes: 700, finishReason: "stop")
        set.record(pingPong: .concern("ignored"))
        #expect(set.trips.isEmpty)
        #expect(set.wantsStop == false)
    }

    /// Observation is the default even when watchdogs are on. A trip is
    /// recorded, and the generation continues.
    @Test func observationRecordsWithoutStopping() {
        var set = WatchdogSet(configuration: WatchdogConfiguration(isEnabled: true))
        set.observe(String(repeating: "Wait, I need to check the facts again. ", count: 8))
        #expect(set.trips.count == 1)
        #expect(set.trips.first?.kind == .loop)
        #expect(set.trips.first?.acted == false)
        #expect(set.wantsStop == false)
        #expect(set.explanation == nil)
    }

    /// Naming a watchdog in the acting list is the only thing that turns a
    /// concern into a stop, and the reason reaches the content (B4).
    @Test func actingTurnsAConcernIntoAStop() {
        var set = WatchdogSet(
            configuration:
                WatchdogConfiguration(isEnabled: true, acting: [.loop]))
        set.observe(String(repeating: "Wait, I need to check the facts again. ", count: 8))
        #expect(set.wantsStop)
        #expect(set.trips.first?.acted == true)
        #expect(set.explanation?.contains("stopped this generation") == true)
    }

    /// Acting on one kind does not enable another.
    @Test func actingIsPerKind() {
        var set = WatchdogSet(
            configuration:
                WatchdogConfiguration(isEnabled: true, acting: [.stall]))
        set.observe(String(repeating: "Wait, I need to check the facts again. ", count: 8))
        #expect(set.trips.count == 1)
        #expect(set.wantsStop == false)
    }

    /// Ping-pong reports and never intervenes, whatever the operator asks
    /// for. The obvious intervention -- answer this turn with no tools --
    /// leaves a tool-templated prompt with a decoder that allows none, and
    /// fails the request outright. Naming it in the acting list is therefore
    /// dropped at parse time rather than honoured into a worse failure.
    @Test func pingPongNeverActs() {
        let configuration = WatchdogConfiguration.fromEnvironment([
            "TINYTITAN_WATCHDOGS": "1",
            "TINYTITAN_WATCHDOG_ACT": "pingpong,loop",
        ])
        #expect(configuration.acting == [.loop])
        #expect(configuration.acts(.pingpong) == false)

        var set = WatchdogSet(
            configuration:
                WatchdogConfiguration(isEnabled: true, acting: Set(WatchdogKind.allCases)))
        set.record(pingPong: .concern("tool read_file called 3 times in a row"))
        #expect(set.trips.count == 1)
        #expect(set.trips.first?.acted == false)
        #expect(set.wantsStop == false)
        #expect(set.resolve(content: "the answer", finishReason: "stop").note == nil)
    }

    /// One report per kind: a loop that continues must not fill the log with
    /// a line per byte.
    @Test func aKindReportsOnce() {
        var set = WatchdogSet(configuration: WatchdogConfiguration(isEnabled: true))
        for _ in 0..<20 {
            set.observe("Wait, I need to check the facts again. ")
        }
        #expect(set.trips.count == 1)
    }

    // MARK: what the client is told

    /// A generation nothing tripped on is returned exactly as it was. This
    /// is the case that must never regress: the feature is off for almost
    /// everyone, and even on it should be invisible until something breaks.
    @Test func aCleanGenerationIsUntouched() {
        var set = WatchdogSet(
            configuration:
                WatchdogConfiguration(isEnabled: true, acting: Set(WatchdogKind.allCases)))
        set.observe("The ferry runs on Sundays.")
        set.finish(visibleBytes: 400, requestBytes: 700, finishReason: "stop")
        let outcome = set.resolve(
            content: "The ferry runs on Sundays.",
            finishReason: "stop")
        #expect(outcome.note == nil)
        #expect(outcome.content == "The ferry runs on Sundays.")
        #expect(outcome.finishReason == "stop")
    }

    /// B4: a stopped generation is reported as `length` -- the nearest
    /// honest reason either protocol offers -- and the truth is told in the
    /// content, which is the one place that cannot break a client.
    @Test func aStoppedGenerationIsReportedAsLengthAndSaysWhy() {
        var set = WatchdogSet(
            configuration:
                WatchdogConfiguration(isEnabled: true, acting: [.loop]))
        set.observe(String(repeating: "Wait, I need to check the facts again. ", count: 8))
        let outcome = set.resolve(content: "partial answer", finishReason: "stop")
        #expect(outcome.finishReason == "length")
        #expect(outcome.content.hasPrefix("partial answer"))
        #expect(outcome.content.contains("TinyTitan stopped this generation"))
        #expect(outcome.content.contains("loop"))
    }

    /// Observation changes nothing a client can see. This is what makes
    /// `TINYTITAN_WATCHDOGS=1` safe to run on a real workload while the
    /// thresholds are being judged.
    @Test func observationIsInvisibleToTheClient() {
        var set = WatchdogSet(configuration: WatchdogConfiguration(isEnabled: true))
        set.observe(String(repeating: "Wait, I need to check the facts again. ", count: 8))
        set.finish(visibleBytes: 4, requestBytes: 700, finishReason: "stop")
        let outcome = set.resolve(content: "x", finishReason: "stop")
        #expect(set.trips.count == 2, "both loop and stub saw something")
        #expect(outcome.note == nil)
        #expect(outcome.content == "x")
        #expect(outcome.finishReason == "stop")
    }

    // MARK: the shared fixture

    /// The calibration script carries a Python port of `LoopWatchdog`, and
    /// its numbers are only worth anything if the port agrees with what
    /// ships. Both sides read this file; `watchdog_calibrate.py --selftest`
    /// is the other half of this test.
    @Test func swiftAgreesWithTheCalibrationFixture() throws {
        struct Case: Decodable {
            let name: String
            let text: String
            let repeatCount: Int?
            let loopTrips: Bool
            enum CodingKeys: String, CodingKey {
                case name, text, loopTrips
                case repeatCount = "repeat"
            }
        }
        struct Fixture: Decodable {
            let defaults: [String: Int]
            let cases: [Case]
        }
        let url = try #require(
            Bundle.module.url(
                forResource: "watchdog-cases",
                withExtension: "json",
                subdirectory: "Fixtures"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        // The fixture also records the thresholds it was measured at, so a
        // default changed on one side and not the other fails here rather
        // than silently invalidating the corpus numbers.
        let configuration = WatchdogConfiguration(isEnabled: true)
        #expect(fixture.defaults["window"] == configuration.loopWindowBytes)
        #expect(fixture.defaults["repeats"] == configuration.loopRepeats)
        #expect(fixture.defaults["history"] == configuration.loopHistoryBytes)
        #expect(fixture.defaults["minimumPeriodBytes"] == LoopWatchdog.minimumPeriodBytes)
        #expect(fixture.defaults["minimumDistinctBytes"] == LoopWatchdog.minimumDistinctBytes)
        for testCase in fixture.cases {
            let text = String(repeating: testCase.text, count: testCase.repeatCount ?? 1)
            let tripped = feed(text, configuration: configuration) != .fine
            #expect(
                tripped == testCase.loopTrips,
                Comment(rawValue: "\(testCase.name): swift said \(tripped)"))
        }
    }

    // MARK: configuration

    @Test func configurationDefaultsToOff() {
        let configuration = WatchdogConfiguration.fromEnvironment([:])
        #expect(configuration.isEnabled == false)
        #expect(configuration.summary == "watchdogs=off")
    }

    @Test func configurationReadsTheEnvironment() {
        let configuration = WatchdogConfiguration.fromEnvironment([
            "TINYTITAN_WATCHDOGS": "1",
            "TINYTITAN_WATCHDOG_ACT": "stall, loop",
            "TINYTITAN_WATCHDOG_STALL_SECONDS": "45",
            "TINYTITAN_WATCHDOG_LOOP_REPEATS": "6",
        ])
        #expect(configuration.isEnabled)
        #expect(configuration.acting == [.stall, .loop])
        #expect(configuration.stallSeconds == 45)
        #expect(configuration.loopRepeats == 6)
        #expect(configuration.summary == "watchdogs=act(loop,stall)")
    }

    /// An unknown name is dropped rather than failing the launch: a typo in
    /// an env var must not stop a server from starting.
    @Test func configurationDropsUnknownActingNames() {
        let configuration = WatchdogConfiguration.fromEnvironment([
            "TINYTITAN_WATCHDOGS": "1",
            "TINYTITAN_WATCHDOG_ACT": "loop,nonsense",
        ])
        #expect(configuration.acting == [.loop])
    }

    @Test func enabledWithoutActingReportsObservation() {
        let configuration = WatchdogConfiguration.fromEnvironment(["TINYTITAN_WATCHDOGS": "on"])
        #expect(configuration.summary == "watchdogs=observe")
    }
}
