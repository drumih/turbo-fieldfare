import Foundation
import Metal
import Testing

import TinyTitanLib
import TinyTitanCore

/// The concurrency and lifetime contract of §5 of
/// `docs/plan-embedded-library.md`, exercised against a real install.
///
/// Gated rather than always-on, like the other tests in this tree that need a
/// model: the package suite runs without one, and a contract nobody can run is a
/// contract nobody has. Point `TINYTITAN_LIBRARY_CONTRACT_MODEL` at a `.ssdai`
/// install to run it:
///
///     TINYTITAN_LIBRARY_CONTRACT_MODEL=models/qwen3.5_4B_4Bit \
///         swift test --no-parallel --filter LibraryContractTests
///
/// Serialized because each test loads an install and, in one case, two.
@Suite(.serialized, .enabled(if: contractModel() != nil))
struct LibraryContractTests {
    /// A small window on purpose: two engines are resident at once in the
    /// two-engine test, and the contract is about who may run, not about how
    /// much context fits.
    private static let window = 8_192

    private func engine() async throws -> Engine {
        let model = try #require(contractModel())
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try await Engine(
            directory: model,
            device: device,
            configuration: EngineConfiguration(contextWindow: Self.window))
    }

    private func ask(
        _ session: Session,
        _ prompt: String,
        maxTokens: Int = 8
    ) async throws -> GenerationSummary {
        try await session.respond(
            to: [ChatMessage(role: .user, content: prompt)],
            options: GenerationOptions(maxTokens: maxTokens, temperature: 0)
        ) { _ in }
    }

    /// §5.3 — the caller owns the device, and two engines on it must work.
    @Test func twoEnginesOnOneDeviceRunAlternately() async throws {
        let first = try await engine()
        let second = try await engine()
        let a = await first.session()
        let b = await second.session()

        #expect(try await ask(a, "Say one.").completionTokens > 0)
        #expect(try await ask(b, "Say two.").completionTokens > 0)
        // Back to the first: the second engine's work must not have disturbed it.
        #expect(try await ask(a, "Say three.").completionTokens > 0)

        await first.unload()
        await second.unload()
    }

    /// §5.2 — a second generation on one session waits on the engine's slot
    /// pool rather than racing the first. The `.busy` alternative is still open
    /// in the plan; what this pins is that today's answer is "waits", and that
    /// both turns come back intact.
    @Test func twoGenerationsOnOneSessionBothComplete() async throws {
        let engine = try await engine()
        defer { Task { await engine.unload() } }
        let session = await engine.session()

        async let first = ask(session, "Count to five.", maxTokens: 24)
        async let second = ask(session, "Name a colour.", maxTokens: 24)
        let (one, two) = try await (first, second)

        #expect(one.completionTokens > 0)
        #expect(two.completionTokens > 0)
        #expect(!one.text.isEmpty)
        #expect(!two.text.isEmpty)
    }

    /// §5.4 — cancellation is cooperative and prompt, and it arrives as
    /// `.cancelled` rather than as a half-finished summary.
    @Test func cancellingMidDecodeEndsTheGeneration() async throws {
        let engine = try await engine()
        defer { Task { await engine.unload() } }
        let session = await engine.session()

        let work = Task {
            try await session.respond(
                to: [ChatMessage(role: .user, content: "Write a long essay about rivers.")],
                options: GenerationOptions(maxTokens: 2_048, temperature: 0.6)
            ) { _ in }
        }
        // Long enough that the model is decoding, not still prefilling: a
        // 512-token decode on the 4B takes tens of seconds.
        try await Task.sleep(for: .seconds(3))
        await session.cancel()

        do {
            let summary = try await work.value
            Issue.record(
                "a cancelled generation returned a summary: \(summary.completionTokens) tokens")
        } catch let error as TinyTitanError {
            guard case .cancelled = error else {
                Issue.record("expected .cancelled, got \(error)")
                return
            }
        }
    }

    /// §5.4 sibling: the CLI's SIGINT handler cancels the task that awaits
    /// `respond`, not the session. That cancellation has to reach the
    /// generation task, or the process keeps decoding to its token limit after
    /// the operator asked it to stop (AUD2-035).
    @Test func cancellingTheAwaitingTaskEndsTheGeneration() async throws {
        let engine = try await engine()
        defer { Task { await engine.unload() } }
        let session = await engine.session()

        let work = Task {
            try await session.respond(
                to: [ChatMessage(role: .user, content: "Write a long essay about rivers.")],
                options: GenerationOptions(maxTokens: 2_048, temperature: 0.6)
            ) { _ in }
        }
        // Same window as the session-cancel test: the model is decoding, not
        // still prefilling.
        try await Task.sleep(for: .seconds(3))
        work.cancel()

        do {
            let summary = try await work.value
            Issue.record(
                "a caller-cancelled generation returned a summary: \(summary.completionTokens) tokens"
            )
        } catch let error as TinyTitanError {
            guard case .cancelled = error else {
                Issue.record("expected .cancelled, got \(error)")
                return
            }
        }
    }

    /// §5.5 — `unload()` releases the model, and every session that outlives it
    /// says so instead of failing somewhere deeper.
    @Test func unloadingShutsLiveSessionsDown() async throws {
        let engine = try await engine()
        let session = await engine.session()
        #expect(try await ask(session, "Say hi.").completionTokens > 0)

        await engine.unload()

        do {
            _ = try await ask(session, "Say hi again.")
            Issue.record("a live session should report engineShutDown after unload")
        } catch let error as TinyTitanError {
            guard case .engineShutDown = error else {
                Issue.record("expected .engineShutDown, got \(error)")
                return
            }
        }

        // A session made *after* the unload is inert in the same way.
        let after = await engine.session()
        do {
            _ = try await ask(after, "And again.")
            Issue.record("a session made after unload should also report engineShutDown")
        } catch let error as TinyTitanError {
            guard case .engineShutDown = error else {
                Issue.record("expected .engineShutDown, got \(error)")
                return
            }
        }
    }
}

/// The install the model-gated suites run against, or `nil` when they skip.
func contractModel() -> URL? {
    guard
        let path = ProcessInfo.processInfo.environment["TINYTITAN_LIBRARY_CONTRACT_MODEL"],
        !path.isEmpty
    else { return nil }
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
        isDirectory.boolValue
    else { return nil }
    return URL(fileURLWithPath: path)
}
