import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@MainActor
@Suite struct TranscriptStreamingTests {
    /// Both streaming paths are measured and framed, so the progressive render
    /// can be compared against the raw appends it replaces rather than only
    /// against a threshold.
    enum Mode: String, CaseIterable, CustomStringConvertible {
        case progressive
        case raw

        var description: String { rawValue }

        var environment: [String: String] {
            ["TURBO_FIELDFARE_PROGRESSIVE_RENDER": self == .raw ? "0" : "1"]
        }
    }

    /// 20 tok/s at a 0.1 s tick is two tokens per tick; Gemma 4 averages close
    /// to four characters per token on English prose. No wall-clock sleeping:
    /// only the chunk size models the rate.
    static let charactersPerTick = 8

    /// Fixtures long enough for decile statistics to mean anything. Both are
    /// the shapes that an O(n^2) re-render would blow up on first.
    static let timingGated: Set<String> = ["large-code-block", "fifty-equations"]

    struct TickTiming {
        let headDecileMean: Double
        let tailDecileMean: Double
        var ratio: Double { headDecileMean > 0 ? tailDecileMean / headDecileMean : .infinity }
    }

    static func ticks(_ source: String) -> [String] {
        var prefixes: [String] = []
        var index = source.startIndex
        while index < source.endIndex {
            let next = source.index(
                index,
                offsetBy: charactersPerTick,
                limitedBy: source.endIndex) ?? source.endIndex
            index = next
            prefixes.append(String(source[source.startIndex..<next]))
        }
        return prefixes
    }

    static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds)
            + Double(duration.components.attoseconds) * 1e-18
    }

    static func timing(_ durations: [Duration]) -> TickTiming? {
        let decile = durations.count / 10
        guard decile > 0 else { return nil }
        let head = durations.prefix(decile).map(seconds).reduce(0, +) / Double(decile)
        let tail = durations.suffix(decile).map(seconds).reduce(0, +) / Double(decile)
        return TickTiming(headDecileMean: head, tailDecileMean: tail)
    }

    @Test(arguments: TranscriptCorpus.fixtures, Mode.allCases)
    func streamsFixtureWithoutSuperlinearTickCost(_ fixture: String, _ mode: Mode) throws {
        let source = try TranscriptCorpus.source(fixture)
        let prefixes = Self.ticks(source)
        try #require(!prefixes.isEmpty)

        var milestones: [Int: String] = [:]
        for (label, percent) in [("p10", 10), ("p50", 50), ("p90", 90)] {
            milestones[max(0, prefixes.count * percent / 100 - 1)] = label
        }
        if let index = prefixes.firstIndex(where: { $0.contains("$$") }) {
            milestones[index] = "first-display-math"
        }
        if let index = prefixes.firstIndex(where: { $0.contains("```") }) {
            milestones[index] = "first-fence"
        }

        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController(
            environment: mode.environment)
        let clock = ContinuousClock()
        var durations: [Duration] = []
        durations.reserveCapacity(prefixes.count)

        for (index, prefix) in prefixes.enumerated() {
            let elapsed = clock.measure {
                controller.synchronize(
                    storage: storage,
                    prompt: "Explain this",
                    response: prefix,
                    isTerminal: false)
            }
            durations.append(elapsed)
            if let label = milestones[index] {
                try TranscriptFrameRenderer.record(
                    storage,
                    named: "\(fixture).stream-\(label).\(mode).light.png",
                    dark: false)
            }
        }

        let finalize = clock.measure {
            controller.synchronize(
                storage: storage,
                prompt: "Explain this",
                response: source,
                isTerminal: true)
        }
        #expect(controller.response == source)
        #expect(controller.isFinalized)

        guard let timing = Self.timing(durations) else {
            #expect(prefixes.count < 10)
            return
        }
        print("""
            STREAM \(fixture) \(mode): ticks=\(prefixes.count) \
            head=\(String(format: "%.1f", timing.headDecileMean * 1e6))us \
            tail=\(String(format: "%.1f", timing.tailDecileMean * 1e6))us \
            ratio=\(String(format: "%.2f", timing.ratio)) \
            finalize=\(String(format: "%.1f", Self.seconds(finalize) * 1e3))ms
            """)
        if Self.timingGated.contains(fixture) {
            #expect(
                timing.ratio < 3,
                "\(fixture) \(mode) tail decile is \(timing.ratio)x the head decile")
        }
    }
}
