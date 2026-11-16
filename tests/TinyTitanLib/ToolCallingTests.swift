import Foundation
import Metal
import Testing

import TinyTitanLib
@testable import TinyTitanCore

/// The tool loop, against a real install.
///
/// Gated the same way the §5 contract is: point
/// `TINYTITAN_LIBRARY_CONTRACT_MODEL` at a `.ssdai` install to run it.
///
///     TINYTITAN_LIBRARY_CONTRACT_MODEL=models/qwen3.5_4B_4Bit \
///         swift test --no-parallel --filter ToolCallingTests
///
/// What this can prove deterministically is the *round trip*: a conversation
/// that offers a tool, replays the assistant turn that called it and feeds the
/// result back must render and generate. Whether a given model chooses to call
/// the tool is the model's business, not an assertion this test can make.
@Suite(.serialized, .enabled(if: contractModel() != nil))
struct ToolCallingTests {
    private static let schema =
        #"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#

    private func engine() async throws -> Engine {
        let model = try #require(contractModel())
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try await Engine(
            directory: model,
            device: device,
            configuration: EngineConfiguration(contextWindow: 8_192))
    }

    @Test func aToolLoopRoundTrips() async throws {
        let engine = try await engine()
        defer { Task { await engine.unload() } }
        let session = await engine.session()

        let call = ToolCall(
            id: "call_1", name: "get_weather", argumentsJSON: #"{"city":"Paris"}"#)
        let summary = try await session.respond(
            to: [
                ChatMessage(role: .user, content: "What is the weather in Paris right now?"),
                ChatMessage(role: .assistant, content: "", toolCalls: [call]),
                ChatMessage(role: .tool, content: "18 degrees and sunny", toolCallID: "call_1"),
            ],
            tools: [
                ToolDefinition(
                    name: "get_weather",
                    description: "Look up the current weather for a city.",
                    parametersJSON: Self.schema)
            ],
            options: GenerationOptions(maxTokens: 32, temperature: 0)
        ) { _ in }

        // The point is that it ran at all: the tool schema rendered, the replayed
        // call and its result were accepted, and the model answered.
        #expect(summary.completionTokens > 0)
        #expect(!summary.text.isEmpty)
    }
}
