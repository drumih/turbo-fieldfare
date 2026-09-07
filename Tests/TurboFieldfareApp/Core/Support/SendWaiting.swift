import Foundation
@testable import TurboFieldfareAppCore

/// Waiting for a send, which is a pipeline rather than a call.
///
/// `send()` takes the composer synchronously — so a second Generate is refused
/// on the same run-loop turn — and then runs its stages in its own task: the
/// replay, the ticket, the request, the retained images, the generation, the
/// commit or the hand-back. `isRunning` is therefore true a hop or two after
/// the click rather than on the line below it, and a test that asserted either
/// state immediately after `send()` was asserting the scheduler.
@MainActor
enum SendWaiting {
    /// Until the runtime is generating, or the send ended without reaching a
    /// generation at all — which is what a refused request does.
    static func generationStarts(_ model: AppModel) async {
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline, model.isTurnInFlight, !model.isRunning {
            await Task.yield()
        }
    }

    /// Until the whole send has resolved: committed, refused, or handed back.
    ///
    /// Polled rather than yielded: a turn whose double answers a token a second
    /// is a real case here, and spinning the main actor through it costs more
    /// than the wait.
    static func turnEnds(_ model: AppModel) async {
        let deadline = Date().addingTimeInterval(30)
        while Date() < deadline, model.isTurnInFlight {
            try? await Task.sleep(for: .milliseconds(1))
        }
        await model.persistenceTail?.value
    }
}
