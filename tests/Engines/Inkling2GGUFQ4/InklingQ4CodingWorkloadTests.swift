import Foundation
import Testing
import TinyTitanCore
import TinyTitanPluginAPI

@testable import Inkling2GGUFQ4

/// A Swift coding workload on the Inkling 2 Q4 install (TT-121): five
/// TinyTitan-style requests, rendered with Inkling's chat format, tokenized by
/// the install's own tokenizer and decoded greedily one after another through
/// one decoder, so the expert cache carries over between requests as it does
/// in the server. Opt-in with `TINYTITAN_INKLING2_CODING=1`;
/// `TINYTITAN_INKLING2_SLOTS` sets the slots per layer (76 is the
/// `--ram-budget 12G` plan). `TINYTITAN_INKLING2_WAIT_ALL=1` waits for every
/// background read at each visit (the old policy), `TINYTITAN_INKLING2_SPIN=0`
/// resolves on the event listener's queue, `TINYTITAN_INKLING2_TRACE_DEPTH`
/// adds the routers further ahead to `TINYTITAN_INKLING2_ROUTING_TRACE`.
struct InklingQ4CodingWorkloadTests {
    static let environment = ProcessInfo.processInfo.environment

    static let prompts = [
        "Write a Swift final class `ExpertSlotCache` that keeps N fixed-size slots in one "
            + "MTLBuffer, maps expert ids to slots with least-recently-used eviction, and loads a "
            + "missing expert with pread from a file descriptor. Include doc comments.",
        "Write a Swift function that parses a GGUF v3 header (magic, version, tensor count, "
            + "metadata count) from a [UInt8] with bounds checks, throwing a typed error enum.",
        "In Swift 6, write a Sendable struct `LoadPlan` with a static function that splits a RAM "
            + "budget in bytes into a context window and expert slots, plus swift-testing @Test "
            + "cases for it.",
        "Write a Package.swift for a SwiftPM library with a Metal-using Swift target, a C target "
            + "built at -O2 and a test target, all in Swift 6 language mode, and explain each part.",
        "Refactor this Swift loop to use DispatchQueue.concurrentPerform and explain why it is "
            + "thread-safe: `for i in 0..<n { out[i] = read(fd, offsets[i]) }`",
    ]

    @Test(
        .enabled(
            if: environment["TINYTITAN_INKLING2_Q4_INSTALL"] != nil
                && environment["TINYTITAN_INKLING2_CODING"] == "1"))
    func swiftCodingWorkload() async throws {
        let directory = URL(
            fileURLWithPath: try #require(Self.environment["TINYTITAN_INKLING2_Q4_INSTALL"]))
        let slots = Self.environment["TINYTITAN_INKLING2_SLOTS"].flatMap(Int.init) ?? 76
        let factory = Inkling2GGUFQ4Plugin.engineFactory(host: .standard)
        let tokenizerFolder = try #require(try factory.tokenizerFolder?(directory))
        let tokenizer = try await GFTokenizer.load(from: tokenizerFolder)
        let stopIDs = Set(tokenizer.stopTokenIDs.map(Int.init))
        let decoder = try InklingQ4Decoder(
            directory: directory, maxContext: 4_096, maxChunk: 256, experts: .slots(slots))
        let cache = try #require(decoder.expertCache)
        cache.waitPerExpert = Self.environment["TINYTITAN_INKLING2_WAIT_ALL"] != "1"
        decoder.routing?.spin = Self.environment["TINYTITAN_INKLING2_SPIN"] != "0"
        let depth = Self.environment["TINYTITAN_INKLING2_TRACE_DEPTH"].flatMap(Int.init)
        decoder.traceDepth = depth ?? 0
        let log = { (line: String) in FileHandle.standardError.write(Data((line + "\n").utf8)) }
        var decodeSeconds = 0.0
        var decoded = 0
        var prefillSeconds = 0.0
        // With TINYTITAN_INKLING2_ROUTING_TRACE, every visit's routed ids
        // (R) and lookahead prediction (P) go to that file, per prompt and
        // phase, for replaying cache policies offline.
        let tracePath = Self.environment["TINYTITAN_INKLING2_ROUTING_TRACE"]
        var trace: [String] = []
        func drain(_ phase: String) {
            guard tracePath != nil else { return }
            trace.append("# \(phase)")
            trace += (cache.routingLog ?? []).map {
                ($0.depth == 0 ? "R " : "P\($0.depth) ") + "\($0.layer) "
                    + $0.ids.map(String.init).joined(separator: ",")
            }
            cache.routingLog = []
        }
        if tracePath != nil { cache.routingLog = [] }
        for (index, prompt) in Self.prompts.enumerated() {
            let rendered = InklingChatTemplate.render(
                messages: [GFTokenizer.Message(role: .user, content: prompt)],
                tools: [], effort: nil, openTextTurn: true)
            let ids = tokenizer.encode(rendered, addBOS: false).map(Int.init)
            decoder.state.reset()
            let prefillStarted = Date()
            var next = try decoder.prefill(ids, position: 0).next
            let prefill = Date().timeIntervalSince(prefillStarted)
            drain("prefill \(index + 1)")
            let decodeStarted = Date()
            var produced = 0
            while produced < 256, !stopIDs.contains(next) {
                produced += 1
                next = try decoder.step(token: next, position: ids.count + produced - 1).next
            }
            let seconds = Date().timeIntervalSince(decodeStarted)
            drain("decode \(index + 1)")
            decodeSeconds += seconds
            decoded += produced
            prefillSeconds += prefill
            log(
                String(
                    format: "coding #%d: prompt %d tok, prefill %.2f s, %d tokens at %.2f tok/s",
                    index + 1, ids.count, prefill, produced, Double(produced) / seconds))
        }
        if let tracePath {
            try trace.joined(separator: "\n").write(
                toFile: tracePath, atomically: true, encoding: .utf8)
        }
        let s = cache.statistics
        #expect(decoded > 0)
        log(
            String(
                format:
                    "coding total (%d slots, wait %@): decode %.2f tok/s over %d tokens, "
                    + "prefill %.1f s; %llu hits, %llu misses, prefetch %llu/%llu used, "
                    + "prefetch wait %.1f s, blocking reads %.1f s",
                slots, cache.waitPerExpert ? "per expert" : "all", Double(decoded) / decodeSeconds,
                decoded, prefillSeconds, s.hits, s.misses, s.prefetchUsed, s.prefetchLoads,
                s.prefetchWaitSeconds, s.readSeconds))
    }
}
