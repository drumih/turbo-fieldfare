import Foundation
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

/// Retention across interleaved conversations, against a real model.
///
/// Everything else about slots is covered without one: selection, eviction,
/// key reservation and the byte accounting are all decided by
/// `ServerPromptCache` alone, and `ServerPromptCacheTests` exercises them on
/// synthetic entries. What no synthetic test can reach is the binding those
/// decisions rest on — that the slot the cache names is the KV the runner
/// resumes on. That needs a loaded model, so it lives here and skips itself
/// when there is not one.
@Suite("Server prompt cache slots, real model", .serialized)
struct ServerPromptCacheSlotModelTests {
    /// A conversation, and the fact it is expected to remember.
    private struct Lineage {
        let secret: String
        var messages: [GFTokenizer.Message]

        init(secret: String) {
            self.secret = secret
            self.messages = [GFTokenizer.Message(
                role: .user,
                content: "My secret number is \(secret). Reply with exactly OK.")]
        }
    }

    /// Two conversations the KV position check cannot tell apart, each asked to
    /// recall its own secret after the other has run.
    ///
    /// The shape of this test is the point, so it is worth stating plainly.
    ///
    /// **Why the secrets are digits.** `prepareForContinuation(expectedPosition:)`
    /// is the only runtime guard on the entry-to-KV binding, and it compares a
    /// token count. Two conversations of different lengths are therefore partly
    /// protected by that guard: a mis-bound slot would fail it loudly and never
    /// reach the model. Digits tokenise one per token, so two five-digit
    /// numbers in the same sentence produce histories of *identical* length,
    /// the guard cannot discriminate them, and the binding is the only thing
    /// left holding them apart. That is the condition worth testing under.
    ///
    /// **Why the assertion is the content, not `cachedTokens`.** A resume onto
    /// the wrong lineage reports a hit exactly like a resume onto the right
    /// one: the count of reused tokens says a prefix was reused, never which.
    /// So the cache counter shows the feature engaged and the recalled number
    /// shows it engaged correctly. Both are required; neither alone says much.
    ///
    /// Run against a checkout with an installed model, or point
    /// `TURBO_FIELDFARE_TEST_MODEL` at one.
    @Test(.enabled(if: Self.modelURL != nil,
                   "no installed model; set TURBO_FIELDFARE_TEST_MODEL to run"))
    func interleavedConversationsKeepTheirOwnKVAcrossEachOther() async throws {
        let modelURL = try #require(Self.modelURL)
        let session = try await ServerModelSession.load(
            modelDirectory: modelURL,
            maxContext: 4_096,
            promptCacheMode: .singlePrefix,
            promptCacheSlots: 2,
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))

        // Both variants share the one model load; loading it again to separate
        // them would double the slowest part of the suite to prove nothing the
        // second pass does not already prove.
        //
        // Each conversation carries its *own* key in the keyed pass. A key
        // names one lineage, so handing both the same one is a claim to be the
        // same conversation, and they would correctly evict each other.
        let variants: [(label: String, alpha: String?, beta: String?)] = [
            ("unkeyed", nil, nil),
            ("keyed", "conversation-alpha", "conversation-beta"),
        ]
        for variant in variants {
            var alpha = Lineage(secret: "41729")
            var beta = Lineage(secret: "58306")
            let label = variant.label

            // Turn one for each, interleaved. Neither can hit; both are new.
            let opened = try await open(
                &alpha, &beta, in: session,
                alphaKey: variant.alpha, betaKey: variant.beta)
            #expect(opened.alpha == opened.beta,
                    "\(label): openings are \(opened.alpha) and \(opened.beta) tokens; they must match, or the position check is doing the work this test means to give the slot binding")

            // Turn two for each, each preceded by the other's request. On one
            // slot this is where both prefixes are already gone.
            for (lineage, key) in [(alpha, variant.alpha), (beta, variant.beta)] {
                var asked = lineage
                asked.messages.append(GFTokenizer.Message(
                    role: .user,
                    content: "What is my secret number? Answer with the digits only."))
                let completion = try await complete(
                    asked.messages, in: session, key: key)

                #expect(completion.usage.promptTokensDetails.cachedTokens > 0,
                        "\(label): \(lineage.secret) did not continue across the other conversation's request")
                #expect(completion.content.contains(lineage.secret),
                        "\(label): \(lineage.secret) was not recalled; got \(completion.content)")
                let other = lineage.secret == alpha.secret
                    ? beta.secret : alpha.secret
                #expect(!completion.content.contains(other),
                        "\(label): resumed onto the other conversation's KV; \(lineage.secret) answered with \(other)")
            }
        }
    }

    /// Opens both conversations and feeds each assistant reply back verbatim.
    ///
    /// Verbatim because the cache holds the tokens the model generated:
    /// `assistantMatches` correctly refuses a turn the model never produced, so
    /// a fabricated reply tests the refusal path and nothing else.
    private func open(
        _ alpha: inout Lineage,
        _ beta: inout Lineage,
        in session: ServerModelSession,
        alphaKey: String?,
        betaKey: String?
    ) async throws -> (alpha: Int, beta: Int) {
        let first = try await complete(alpha.messages, in: session, key: alphaKey)
        let second = try await complete(beta.messages, in: session, key: betaKey)
        alpha.messages.append(GFTokenizer.Message(
            role: .assistant, content: first.content))
        beta.messages.append(GFTokenizer.Message(
            role: .assistant, content: second.content))
        return (first.usage.promptTokens, second.usage.promptTokens)
    }

    private func complete(
        _ messages: [GFTokenizer.Message],
        in session: ServerModelSession,
        key: String?,
        mode: ServerPromptCacheMode? = nil
    ) async throws -> ServerCompletion {
        try await session.generate(
            ValidatedChatRequest(
                messages: messages,
                tools: [],
                stream: false,
                includeUsage: true,
                // Greedy, so the reply fed back into the next turn is the same
                // one every run.
                generationConfig: GenerationConfig(
                    maxNewTokens: 24, temperature: 0),
                maximumCompletionTokens: 24,
                promptCacheKey: key,
                promptCacheMode: mode),
            onEvent: { _ in })
    }

    /// What `prompt_cache_mode: off` has to be worth, against a real model.
    ///
    /// A single-shot caller — a summariser that will never continue anything —
    /// still had to prefill somewhere, and prefilling into a slot destroys the
    /// conversation in it. So before this, traffic that could not benefit from
    /// the cache still cost a conversation its prefix just by running, which is
    /// the same eviction the slots work removed between two conversations.
    ///
    /// Both halves are asserted, because either alone is satisfiable by doing
    /// nothing useful: the opted-out request must not publish (it cannot be
    /// continued afterwards) and must not evict (the named conversations must
    /// still continue across it).
    @Test(.enabled(if: Self.modelURL != nil,
                   "no installed model; set TURBO_FIELDFARE_TEST_MODEL to run"))
    func anOptedOutRequestNeitherPublishesNorEvicts() async throws {
        let modelURL = try #require(Self.modelURL)
        let session = try await ServerModelSession.load(
            modelDirectory: modelURL,
            maxContext: 4_096,
            promptCacheMode: .singlePrefix,
            promptCacheSlots: 2,
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))

        var alpha = Lineage(secret: "41729")
        var beta = Lineage(secret: "58306")
        _ = try await open(&alpha, &beta, in: session,
                           alphaKey: "conversation-alpha",
                           betaKey: "conversation-beta")

        // Both slots are now claimed. Fire opted-out traffic through the
        // server: enough of it to have evicted both several times over if it
        // shared the pool.
        let single = [GFTokenizer.Message(
            role: .user, content: "Summarise nothing. Reply with exactly OK.")]
        for _ in 1...4 {
            let completion = try await complete(
                single, in: session, key: nil, mode: .off)
            // No publish: an opted-out request is never continuable, so its own
            // repeat cannot reuse anything either. Were it publishing, the
            // second and later of these identical requests would report a hit.
            #expect(completion.usage.promptTokensDetails.cachedTokens == 0,
                    "an opted-out request reused or published a prefix")
        }

        // No eviction: both named conversations continue across all of it.
        for (lineage, key) in [(alpha, "conversation-alpha"),
                               (beta, "conversation-beta")] {
            var asked = lineage
            asked.messages.append(GFTokenizer.Message(
                role: .user,
                content: "What is my secret number? Answer with the digits only."))
            let completion = try await complete(
                asked.messages, in: session, key: key)
            #expect(completion.usage.promptTokensDetails.cachedTokens > 0,
                    "opted-out traffic evicted \(lineage.secret)")
            #expect(completion.content.contains(lineage.secret),
                    "\(lineage.secret) was not recalled; got \(completion.content)")
        }
    }

    /// An installed model, or nil.
    ///
    /// `TURBO_FIELDFARE_TEST_MODEL` comes first so a checkout without its own
    /// `scratch/` can borrow one — a second checkout beside the first is the
    /// ordinary way to work on a branch, and a sibling is not on the walk-up
    /// path. Matches `TURBO_FIELDFARE_TOKENIZER_DIR`, which the tokenizer reads
    /// for the same reason.
    ///
    /// The walk-up starts at this source file rather than the working
    /// directory, which is wherever the suite happens to be run from.
    static func modelURL(from file: StaticString = #filePath,
                         environment: [String: String] =
                            ProcessInfo.processInfo.environment) -> URL? {
        let manager = FileManager.default
        if let override = environment["TURBO_FIELDFARE_TEST_MODEL"],
           !override.isEmpty {
            let url = URL(fileURLWithPath: override).standardizedFileURL
            // An explicit path is an operator's statement that a model is
            // there. A typo must skip loudly rather than silently falling back
            // to a different model than the one named.
            guard manager.fileExists(atPath: url.path) else { return nil }
            return url
        }
        var directory = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
        while directory.path != "/" {
            let candidate = directory
                .appendingPathComponent("scratch")
                .appendingPathComponent("gemma4.gturbo")
            if manager.fileExists(atPath: candidate.path) {
                return candidate
            }
            directory = directory.deletingLastPathComponent()
        }
        return nil
    }

    static var modelURL: URL? { modelURL() }
}
