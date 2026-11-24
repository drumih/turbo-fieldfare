import Foundation
import Testing

@testable import Inkling2GGUFQ4

/// The Inkling 2 Q4 variant reading its `.ssdai` install (TT-121 phase 2).
///
/// Oracle, model-gated on `TINYTITAN_INKLING2_Q4_INSTALL` (the install
/// directory): greedy agreement, under teacher forcing, with the llama.cpp
/// reference (`q4kxl-reference.json`, captured from `llama-server` on the same
/// Q4_K_XL file the install was converted from).
@Suite(.serialized) struct InklingQ4ParityTests {
    private static let environment = ProcessInfo.processInfo.environment
    private static var install: String? { environment["TINYTITAN_INKLING2_Q4_INSTALL"] }

    struct Reference: Decodable {
        struct Step: Decodable {
            let id: Int
        }
        struct Case: Decodable {
            let prompt: String
            let prompt_ids: [Int]
            let steps: [Step]
        }
        let cases: [Case]
    }

    static func reference() throws -> Reference {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/q4kxl-reference.json")
        return try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
    }

    static func log(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    @Test(.enabled(if: install != nil))
    func greedyTokensMatchLlamaCpp() throws {
        let decoder = try InklingQ4Decoder(
            directory: URL(fileURLWithPath: try #require(Self.install)), maxContext: 256,
            maxChunk: 64)
        var matched = 0
        var total = 0
        for item in try Self.reference().cases {
            decoder.state.reset()
            var result = try decoder.prefill(item.prompt_ids, position: 0)
            var position = item.prompt_ids.count
            for (index, expected) in item.steps.enumerated() {
                total += 1
                matched += result.next == expected.id ? 1 : 0
                if index + 1 < item.steps.count {
                    result = try decoder.step(token: expected.id, position: position)
                    position += 1
                }
            }
        }
        Self.log("Inkling Q4 install: greedy agreement with llama.cpp \(matched)/\(total)")
        #expect(total > 0)
        #expect(matched == total, "greedy agreement \(matched)/\(total)")
    }

    /// Softcapped logits of every teacher-forced step of every reference case,
    /// first through a chunked prefill of the prompt, then token by token.
    static func trace(
        prefill: ([Int]) throws -> [Float], step: (Int, Int) throws -> [Float],
        reset: () -> Void
    ) throws -> [[Float]] {
        var rows: [[Float]] = []
        for item in try reference().cases {
            reset()
            rows.append(try prefill(item.prompt_ids))
            var position = item.prompt_ids.count
            for expected in item.steps.dropLast() {
                rows.append(try step(expected.id, position))
                position += 1
            }
        }
        return rows
    }
}

/// Expert streaming (TT-121 phase 2): the experts live in per-layer slots read
/// from the install with the page cache bypassed, and a layer visit stops
/// after its router to load what it chose. Fewer slots than a prefill chunk
/// uses forces several rounds per visit and constant eviction; the logits must
/// still be bit-identical to the mapped install, because every expert's
/// matrix is the same bytes run by the same kernel at the same tile width.
extension InklingQ4ParityTests {
    static func mappedTrace(_ directory: URL) throws -> [[Float]] {
        try autoreleasepool {
            let decoder = try InklingQ4Decoder(
                directory: directory, maxContext: 256, maxChunk: 64, experts: .mapped)
            return try traceLogits(decoder)
        }
    }

    static func traceLogits(_ decoder: InklingQ4Decoder) throws -> [[Float]] {
        try trace(
            prefill: { try #require(decoder.prefill($0, position: 0, returnLogits: true).logits) },
            step: {
                try #require(decoder.step(token: $0, position: $1, returnLogits: true).logits)
            },
            reset: { decoder.state.reset() })
    }

    /// Slots per layer, and whether decode resolves them behind a shared
    /// event (one command buffer per token) or ends a buffer at each router.
    static var slotCounts: [(slots: Int, eventRouting: Bool)] {
        var counts = [(16, true), (16, false), (64, true)]
        if environment["TINYTITAN_INKLING2_Q4_FULL_SLOTS"] == "1" { counts.append((128, true)) }
        return counts
    }

    @Test(.enabled(if: install != nil))
    func streamedExpertsAreBitIdenticalToTheMappedInstall() throws {
        let directory = URL(fileURLWithPath: try #require(Self.install))
        let mapped = try Self.mappedTrace(directory)
        for (slots, eventRouting) in Self.slotCounts {
            let (streamed, stats): ([[Float]], InklingQ4ExpertCache.Statistics) =
                try autoreleasepool {
                    let decoder = try InklingQ4Decoder(
                        directory: directory, maxContext: 256, maxChunk: 64,
                        experts: .slots(slots))
                    decoder.eventRouting = eventRouting
                    let cache = try #require(decoder.expertCache)
                    return (try Self.traceLogits(decoder), cache.statistics)
                }
            var differing = 0
            for (ours, theirs) in zip(streamed, mapped)
            where ours.map(\.bitPattern) != theirs.map(\.bitPattern) {
                differing += 1
            }
            Self.log(
                "Inkling Q4, \(slots) slots/layer, event routing \(eventRouting): \(differing) of \(mapped.count) rows differ; "
                    + "\(stats.syncs) syncs, \(stats.rounds) rounds, \(stats.hits) hits, "
                    + "\(stats.misses) misses, \(stats.bytesRead >> 20) MiB read")
            #expect(streamed.count == mapped.count)
            #expect(differing == 0, "\(slots) slots, event routing \(eventRouting)")
            if slots < 128 {
                #expect(stats.misses > 0 && stats.syncs > 0)
            }
            if slots == 16 {
                // A 64-token chunk routes to more than 16 experts in a layer.
                #expect(stats.rounds > stats.syncs, "no visit needed a second round")
            }
        }
    }
}

/// Decode speed with streamed experts, wall clock per token after a short
/// prompt. Opt-in with `TINYTITAN_KERNEL_BENCH=1`; `TINYTITAN_INKLING2_SLOTS`
/// picks the slots per layer (0 = mapped).
extension InklingQ4ParityTests {
    @Test(.enabled(if: install != nil && environment["TINYTITAN_KERNEL_BENCH"] == "1"))
    func streamedDecodeThroughput() throws {
        let slots = Self.environment["TINYTITAN_INKLING2_SLOTS"].flatMap(Int.init) ?? 32
        let decoder = try InklingQ4Decoder(
            directory: URL(fileURLWithPath: try #require(Self.install)), maxContext: 512,
            maxChunk: 64, experts: slots == 0 ? .mapped : .slots(slots))
        decoder.lookaheadPrefetch = Self.environment["TINYTITAN_INKLING2_PREFETCH"] != "0"
        decoder.eventRouting = Self.environment["TINYTITAN_INKLING2_EVENT"] != "0"
        decoder.routing?.spin = Self.environment["TINYTITAN_INKLING2_SPIN"] != "0"
        let prompt = [371, 151451, 149879, 151194, 149884]
        var next = try decoder.prefill(prompt, position: 0).next
        let count = 64
        var samples: [Double] = []
        var gpu: Double = 0
        let readBefore = decoder.expertCache?.statistics.readSeconds ?? 0
        let handlerBefore = decoder.routing?.handlerSeconds ?? 0
        let waitBefore = decoder.expertCache?.statistics.prefetchWaitSeconds ?? 0
        for offset in 0..<count {
            let started = Date()
            next = try decoder.step(token: next, position: prompt.count + offset).next
            samples.append(Date().timeIntervalSince(started))
            gpu += decoder.lastGPUSeconds
        }
        let read = (decoder.expertCache?.statistics.readSeconds ?? 0) - readBefore
        let total = samples.reduce(0, +)
        #expect(total > 0)
        let stats = decoder.expertCache?.statistics
        Self.log(
            String(
                format:
                    "Inkling Q4 decode, %@: %.2f tok/s, median %.1f ms; resident experts %d MiB",
                slots == 0 ? "mapped" : "\(slots) slots", Double(count) / total,
                samples.sorted()[count / 2] * 1e3,
                (decoder.expertCache?.residentBytes ?? 0) >> 20)
                + (stats.map {
                    "; \($0.hits) hits, \($0.misses) misses in \($0.loadCalls) loads; lookahead \($0.predictedCorrect)/"
                        + "\($0.predicted) right, prefetch \($0.prefetchUsed)/\($0.prefetchLoads) used, "
                        + String(format: "waited %.1f ms", $0.prefetchWaitSeconds * 1e3)
                } ?? "")
                + String(
                    format: "; per token: GPU %.1f ms, expert reads %.1f ms, rest %.1f ms",
                    gpu / Double(count) * 1e3, read / Double(count) * 1e3,
                    (total - gpu - read) / Double(count) * 1e3)
                + String(
                    format: "; handlers %.1f ms/token (prefetch wait %.1f ms)",
                    ((decoder.routing?.handlerSeconds ?? 0) - handlerBefore) / Double(count) * 1e3,
                    ((stats?.prefetchWaitSeconds ?? 0) - waitBefore) / Double(count) * 1e3))
    }
}

/// Writes the decode routing trace (one line per layer visit: the stored layer
/// and the routed ids) to `TINYTITAN_INKLING2_ROUTING_TRACE`, for measuring
/// prefetch policies offline. Opt-in.
extension InklingQ4ParityTests {
    @Test(.enabled(if: install != nil && environment["TINYTITAN_INKLING2_ROUTING_TRACE"] != nil))
    func writeDecodeRoutingTrace() throws {
        let path = try #require(Self.environment["TINYTITAN_INKLING2_ROUTING_TRACE"])
        let decoder = try InklingQ4Decoder(
            directory: URL(fileURLWithPath: try #require(Self.install)), maxContext: 1_024,
            maxChunk: 64, experts: .slots(64))
        let cache = try #require(decoder.expertCache)
        var lines: [String] = []
        for item in try Self.reference().cases {
            decoder.state.reset()
            var next = try decoder.prefill(item.prompt_ids, position: 0).next
            cache.routingLog = []
            for offset in 0..<96 {
                next = try decoder.step(token: next, position: item.prompt_ids.count + offset).next
            }
            let log = cache.routingLog ?? []
            cache.routingLog = nil
            lines.append("# case")
            lines += log.map { "\($0.layer) " + $0.ids.map(String.init).joined(separator: ",") }
        }
        #expect(lines.count > 48)
        try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
    }
}
