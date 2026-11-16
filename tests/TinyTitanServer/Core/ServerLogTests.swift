//
//  ServerLogTests.swift
//  TinyTitanServer
//
//  The residency line's `prompt_cache` field.
//

import Testing

@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

/// A whole catalog shares one resident slot, so the `prompt_cache` a load
/// reports has to come from the backend that just loaded. Naming the server's
/// own flag instead would report the previous model's cache after a switch --
/// most visibly to a CPU entry, which has no cache at all.
@Suite("Prompt cache reporting")
struct PromptCacheReportingTests {
    /// A backend that can describe its cache, the way a GPU session can.
    private struct Describing: ServerInferenceBackend, PromptCacheDescribing {
        let promptCacheMode: ServerPromptCacheMode
        var maximumContext: Int { 4096 }

        func generate(
            _ request: ValidatedChatRequest,
            onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
        ) async throws -> ServerCompletion {
            throw ServerRequestError.unsupportedOperation("stub")
        }
    }

    /// A backend that cannot: an engine with no cache, and every test double.
    private struct Silent: ServerInferenceBackend {
        var maximumContext: Int { 4096 }

        func generate(
            _ request: ValidatedChatRequest,
            onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
        ) async throws -> ServerCompletion {
            throw ServerRequestError.unsupportedOperation("stub")
        }
    }

    @Test func aDescribingBackendNamesItsOwnMode() {
        #expect(
            ServerLog.promptCacheField(for: Describing(promptCacheMode: .multiPrefix))
                == " prompt_cache=multi-prefix")
        #expect(
            ServerLog.promptCacheField(for: Describing(promptCacheMode: .singlePrefix))
                == " prompt_cache=single-prefix")
        #expect(
            ServerLog.promptCacheField(for: Describing(promptCacheMode: .off))
                == " prompt_cache=off")
    }

    /// The field is added by the caller, so a backend that cannot describe its
    /// cache contributes an empty string rather than a wrong mode.
    @Test func aBackendThatCannotDescribeAddsNothing() {
        #expect(ServerLog.promptCacheField(for: Silent()).isEmpty)
    }
}
