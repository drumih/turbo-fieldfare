import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitanLib

/// The supervisor that owns the watchdog state a generation task and a ticker
/// both write to (`WatchdogSupervisor.swift`).
///
/// `WatchdogTests` covers the detectors themselves. What is covered here is
/// everything the class between them and the generation loop can get wrong on
/// its own: the disabled path (including the promise that it starts no tasks),
/// the lock-wrapped delegation that must not drop or duplicate state, the
/// first-stop-wins rule, and what the client is finally told. The concurrency
/// case is the reason to be here at all -- the type is `@unchecked Sendable`
/// on the note "every stored property is guarded by `lock`", and a note is a
/// claim until something exercises it from two threads.
@Suite("Watchdog supervisor")
struct WatchdogSupervisorTests {

    /// Enabled, watching for stalls, and allowed to act on them.
    private var actingOnStall: WatchdogConfiguration {
        WatchdogConfiguration(isEnabled: true, acting: [.stall], stallSeconds: 5)
    }

    /// Enabled and watching, but allowed to intervene in nothing.
    private var observingOnly: WatchdogConfiguration {
        WatchdogConfiguration(
            isEnabled: true, acting: [], stallSeconds: 5,
            loopRepeats: 2, loopWindowBytes: 16)
    }

    /// A disabled feature must not appear on the per-token path at all, and it
    /// must not start a task. Both are promises in the source, not style.
    @Test("Off means off: no lock, no trips, no task")
    func disabledSupervisorIsInert() async throws {
        let supervisor = WatchdogSupervisor(configuration: .off)
        #expect(supervisor.isActive == false)
        #expect(supervisor.wantsStop == false)
        #expect(supervisor.stopMessage == nil)
        #expect(supervisor.explanation == nil)
        // The shapes every detector hunts, fed to a disabled supervisor.
        supervisor.observe("abcdefghijklmnopqrstuvwxyz0123", at: .now)
        supervisor.observe("abcdefghijklmnopqrstuvwxyz0123", at: .now)
        supervisor.observe("abcdefghijklmnopqrstuvwxyz0123", at: .now)
        supervisor.check(at: .now.advanced(by: .seconds(3600)))
        supervisor.finish(visibleBytes: 1, requestBytes: 10_000, finishReason: "stop")
        supervisor.record(pingPong: .concern("would be a loop"))
        #expect(supervisor.trips.isEmpty)
        #expect(supervisor.wantsStop == false)
        #expect(supervisor.startTicker() == nil)
        #expect(WatchdogSupervisor.inert.isActive == false)
        #expect(WatchdogSupervisor.inert.wantsStop == false)
    }

    /// The clock is injected, so this runs in no real time at all: a token at
    /// t0, nothing for a threshold, and `check` at t0 + threshold finds it.
    @Test("A stall reaches wantsStop, the message, and the client's text")
    func stallTravelsThroughTheSupervisorToIntervention() async throws {
        let supervisor = WatchdogSupervisor(configuration: actingOnStall)
        #expect(supervisor.isActive)
        let firstToken = ContinuousClock.now
        supervisor.observe("the answer begins", at: firstToken)
        #expect(supervisor.wantsStop == false, "a stall needs a check, not a token")

        supervisor.check(at: firstToken.advanced(by: .seconds(6)))
        #expect(supervisor.wantsStop)
        let stop = try #require(supervisor.stopMessage)
        #expect(stop.contains("stall"))
        #expect(stop.contains("no visible token for 5s"))
        let trips = supervisor.trips
        #expect(trips.count == 1)
        #expect(trips[0].kind == .stall)
        #expect(trips[0].acted)

        // B4: the honest finish reason does not exist in either protocol, so
        // the reason maps to the nearest one and the truth goes in the text.
        let outcome = supervisor.resolve(content: "half an answer", finishReason: "stop")
        #expect(outcome.finishReason == "length")
        #expect(outcome.content.hasPrefix("half an answer"))
        #expect(outcome.content != "half an answer")
        #expect(outcome.note == supervisor.explanation)
    }

    /// The second `check` must not re-report, and a later trip must not
    /// rewrite the stop message the client was already promised.
    @Test("The first stop wins and is not duplicated")
    func firstStopWins() async throws {
        let supervisor = WatchdogSupervisor(configuration: actingOnStall)
        let firstToken = ContinuousClock.now
        supervisor.observe("token", at: firstToken)
        supervisor.check(at: firstToken.advanced(by: .seconds(6)))
        let stop = supervisor.stopMessage
        supervisor.check(at: firstToken.advanced(by: .seconds(600)))
        #expect(supervisor.trips.count == 1, "a stall reports once, not every tick")
        // A different watchdog can still report; it cannot take the first one's place.
        supervisor.record(pingPong: .concern("same tool call again"))
        #expect(supervisor.trips.count == 2)
        #expect(supervisor.stopMessage == stop)
    }

    /// Observation-only mode is the default the environment flag reaches for
    /// `on`, and it must cost the client nothing: the trip is logged, the
    /// completion is handed back exactly as it arrived.
    @Test("Watching without acting changes neither the content nor the reason")
    func observationDoesNotAlterTheCompletion() async throws {
        let supervisor = WatchdogSupervisor(configuration: observingOnly)
        let chunk = "abcdefghijklmnopqrstuvwxyz0123"
        for _ in 0..<3 { supervisor.observe(chunk, at: .now) }
        supervisor.finish(visibleBytes: 10, requestBytes: 657, finishReason: "stop")
        #expect(supervisor.trips.isEmpty == false, "the loop and stub should have reported")
        #expect(supervisor.trips.allSatisfy { $0.acted == false })
        #expect(supervisor.wantsStop == false)
        #expect(supervisor.explanation == nil)
        let outcome = supervisor.resolve(content: "a good answer", finishReason: "stop")
        #expect(outcome.content == "a good answer")
        #expect(outcome.finishReason == "stop")
        #expect(outcome.note == nil)
    }

    /// Reasoning is watched by its own loop detector, so an answer that
    /// restates the end of its thought is not a loop -- and a thought that
    /// circles is caught even though it never reached the answer channel.
    @Test("Reasoning loops are caught, and do not poison the answer window")
    func reasoningHasItsOwnWindow() async throws {
        let looping = WatchdogSupervisor(configuration: observingOnly)
        let chunk = "abcdefghijklmnopqrstuvwxyz0123"
        for _ in 0..<3 { looping.observeReasoning(chunk, at: .now) }
        let kinds = looping.trips.map(\.kind)
        #expect(kinds == [.loop])
        #expect(looping.trips[0].message.contains("in reasoning"))

        // The same text split across the two channels does not trip while each
        // channel is below its own repeat count -- which is the point of two
        // windows, since a shared one would already have seen two occurrences.
        let split = WatchdogSupervisor(configuration: observingOnly)
        split.observeReasoning(chunk, at: .now)
        split.observe(chunk, at: .now)
        #expect(split.trips.isEmpty)
        // One more on the answer channel trips the answer, and only the answer.
        split.observe(chunk, at: .now)
        #expect(split.trips.map(\.kind) == [.loop])
        #expect(split.trips[0].message.contains("in reasoning") == false)
    }

    /// The `unchecked-invariant` note says every stored property is guarded by
    /// `lock`. Two writers and three readers at once is how that claim gets
    /// checked; the count is exact, so a lost update under contention fails
    /// this rather than merely being slower.
    @Test("Concurrent writers lose nothing and readers stay consistent")
    func concurrentAccess() async throws {
        let supervisor = WatchdogSupervisor(configuration: observingOnly)
        let writers = 64
        try await withThrowingTaskGroup(of: Void.self) { group in
            for index in 0..<writers {
                group.addTask {
                    supervisor.record(pingPong: .concern("write \(index)"))
                }
            }
            // Readers on the same lock, hammering while the writers run.
            for _ in 0..<4 {
                group.addTask {
                    for _ in 0..<200 {
                        _ = supervisor.wantsStop
                        _ = supervisor.stopMessage
                        _ = supervisor.explanation
                        supervisor.check()
                        _ = supervisor.trips
                    }
                }
            }
            try await group.waitForAll()
        }
        #expect(supervisor.trips.count == writers)
        let messages = Set(supervisor.trips.map(\.message))
        #expect(messages.count == writers, "every writer's report survived")
        // pingpong can act on nothing, so a full set of trips still stops nothing.
        #expect(supervisor.wantsStop == false)
    }

    /// The ticker exists so the clock keeps moving while the generation task is
    /// not running; it must be cancellable and must not outlive the supervisor.
    @Test("An enabled supervisor's ticker starts and stops on demand")
    func tickerLifecycle() async throws {
        let supervisor = WatchdogSupervisor(configuration: actingOnStall)
        let ticker = try #require(supervisor.startTicker())
        #expect(supervisor.isActive)
        ticker.cancel()
        _ = await ticker.value  // returns once cancelled; must not hang
        // `weak self` means dropping the supervisor ends the ticker too: the
        // task's guard fails and it returns rather than ticking on unowned.
        var transient: WatchdogSupervisor? = WatchdogSupervisor(configuration: actingOnStall)
        let orphan = try #require(transient?.startTicker())
        transient = nil
        _ = await orphan.value
    }
}
