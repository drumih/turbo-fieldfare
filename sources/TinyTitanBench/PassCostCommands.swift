import Foundation
import Metal
import TinyTitan
import TinyTitanLib

enum PassCostError: Error, CustomStringConvertible {
    case usage
    case noDevice
    case diverged(position: Int, expected: Int32, actual: Int32)

    var description: String {
        switch self {
        case .usage:
            "usage: TinyTitanBench passcost <install> [roundsPerWidth=6] [chunked|dedicated|both] [promptFile]"
        case .noDevice:
            "no Metal device"
        case .diverged(let position, let expected, let actual):
            "scalar replay diverged at position \(position): expected \(expected), got \(actual)"
        }
    }
}

/// `TinyTitanBench passcost <install> [roundsPerWidth] [promptFile]`: gate 2 of
/// docs/plan-wide-verify.md -- what one W-row pass costs against a scalar token.
///
/// A W-row verify pass is the boundary token plus W-1 drafted tokens through the
/// chunked prefill path. This bench times exactly that, with the drafts the
/// model would accept: phase 1 decodes a greedy reference continuation scalar
/// (which is also the scalar baseline), phase 2 replays it and, at each round's
/// boundary, prefills the next W reference tokens as one pass, restores the
/// state captured before it, and advances by scalar decode.
///
/// Two choices keep the number honest:
///
/// - A round runs one width, and the pass precedes the scalar decode of its
///   tokens, so the pass never finds its own experts already cached. The cache
///   holds what a real speculative step would find: the recent tokens'.
/// - The scalar baseline comes from phase 1, which runs no passes, so a pass
///   cannot have warmed the experts a scalar token then reads.
///
/// The last row's greedy token is compared with the reference; a mismatch is
/// reported, not hidden -- it is the byte-identity question (task 3) showing up.
extension TinyTitanBench {
    /// Which pass is timed: the production chunked prefill (gate 2 as the
    /// plan defined it), or the dedicated W-row verify (`verifyGreedyRows`:
    /// the MTP pair schedule widened, decode cache pinned).
    /// `both` alternates the two round by round in one process, so they share
    /// one scalar baseline and whatever else the machine is doing.
    enum PassCostPath: String, Codable {
        case chunked
        case dedicated
        case both
    }

    struct PassCostRow: Codable {
        let path: PassCostPath
        let width: Int
        let position: Int
        let passMilliseconds: Double
        let lastRowMatches: Bool
        /// Dedicated path only: head time inside the pass, and how many rows'
        /// greedy predictions equal scalar decode's.
        var headMilliseconds: Double?
        var rowsMatching: Int?
    }

    struct PassCostReport: Codable {
        let install: String
        let paths: [PassCostPath]
        let widths: [Int]
        let promptTokens: Int
        let scalarMillisecondsMedian: Double
        let scalarMillisecondsMean: Double
        let scalarSamples: Int
        let rows: [PassCostRow]
        let fits: [PassCostFit]
    }

    /// Least squares over one path's passes: cost = base + perRow * width.
    struct PassCostFit: Codable {
        let path: PassCostPath
        let baseMilliseconds: Double
        let perRowMilliseconds: Double
    }

    static func runPassCost(
        install: String, roundsPerWidth: Int, path: PassCostPath, promptFile: String?
    ) throws {
        /// unchecked-invariant: written only inside the Task below and read
        /// only after `semaphore.wait()` returns; no concurrent access.
        final class Box: @unchecked Sendable { var error: Error? }
        let box = Box()
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                try await runPassCostAsync(
                    install: install, roundsPerWidth: max(1, roundsPerWidth),
                    path: path, promptFile: promptFile)
            } catch {
                box.error = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.error { throw error }
    }

    static let passCostWidths = [2, 4, 8, 16]

    /// Code the model is asked to restate: an edit-shaped prompt, so routing
    /// looks like the git-session workload rather than open prose.
    static func passCostPrompt(promptFile: String?) throws -> String {
        if let promptFile {
            return try String(contentsOfFile: promptFile, encoding: .utf8)
        }
        let source = "sources/TinyTitan/Runtime/Generation/StreamingMTP.swift"
        let body =
            (try? String(contentsOfFile: source, encoding: .utf8))
            .map {
                $0.split(separator: "\n", omittingEmptySubsequences: false)
                    .prefix(140).joined(separator: "\n")
            }
            ?? "func add(_ a: Int, _ b: Int) -> Int {\n    return a + b\n}"
        return "Below is a Swift source file.\n```swift\n\(body)\n```\n\n"
            + "Here is the same file with every doc comment shortened to one line:\n"
            + "```swift\n"
    }

    /// lint:allow-long one measurement script: load, reference decode, replay
    /// with interleaved passes, report. The phases share the token stream and
    /// the logits buffer, and splitting them would only thread both through.
    private static func runPassCostAsync(
        install: String, roundsPerWidth: Int, path: PassCostPath, promptFile: String?
    ) async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw PassCostError.noDevice
        }
        // The decode head production picks for a pure-greedy request.
        let session = try await ServerModelSession.load(
            modelDirectory: URL(fileURLWithPath: install),
            maxContext: 8192,
            forceLogitsHead: false,
            device: device)
        let runner = session.benchmarkRunner
        let prompt = session.benchmarkTokenizer.encode(
            try passCostPrompt(promptFile: promptFile), addBOS: true)
        let installURL = URL(fileURLWithPath: install)
        let vocab = try ArchConfig.resolved(
            forFamily: ManifestReader.peekFamily(directoryURL: installURL),
            directoryURL: installURL
        ).vocabSize
        guard
            let logits = device.makeBuffer(
                length: vocab * MemoryLayout<Float16>.stride,
                options: .storageModeShared)
        else { throw PassCostError.noDevice }
        let prefillConfig = PrefillRuntimeConfig.production(chunkTokens: 32)
        // Every round advances 16 tokens whatever its width, so the expert
        // cache turns over the same way between passes on both paths.
        let maxWidth = passCostWidths.max() ?? 16
        // The dedicated pass plans its rows' expert union as one lease against
        // the slot cache, so it is limited to what is guaranteed to fit.
        let paths: [PassCostPath] = path == .both ? [.chunked, .dedicated] : [path]
        let widths =
            paths.contains(.dedicated)
            ? passCostWidths.filter { $0 <= runner.wideVerifyMaximumRows } : passCostWidths
        guard !widths.isEmpty else {
            throw PrefillError.chunkedUnsupported(
                "dedicated verify needs a routed slot cache of at least 2 * topK")
        }
        let warmup = 32
        let rounds = widths.count * paths.count * roundsPerWidth
        let total = warmup + rounds * maxWidth + maxWidth

        func seedToken(_ result: PrefillResult) -> Int32 {
            switch result.seed {
            case .greedyToken(let token): return Int32(bitPattern: token)
            case .logitsWritten: return argmaxHalf(logits, count: vocab)
            }
        }
        func now() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
        func ms(_ start: UInt64) -> Double { Double(now() &- start) / 1e6 }

        // Phase 1: the greedy reference and the scalar baseline. reference[i]
        // is the token at position prompt.count + i.
        runner.reset()
        var reference = [
            seedToken(
                try await runner.prefillChunked(
                    tokens: prompt[...], startPosition: 0,
                    outputMode: .greedyIfAvailable, config: prefillConfig,
                    into: logits, onProgress: { _ in }))
        ]
        var scalar: [Double] = []
        for index in 0..<total {
            let start = now()
            try await runner.produce(
                token: reference[index], position: prompt.count + index,
                into: logits)
            if index >= warmup { scalar.append(ms(start)) }
            reference.append(Int32(bitPattern: runner.lastGreedyToken))
        }

        // Phase 2: replay to the first boundary, then one pass per round.
        runner.reset()
        _ = try await runner.prefillChunked(
            tokens: prompt[...], startPosition: 0,
            outputMode: .greedyIfAvailable, config: prefillConfig,
            into: logits, onProgress: { _ in })
        func advance(from index: Int, count: Int) async throws {
            for step in index..<(index + count) {
                try await runner.produce(
                    token: reference[step], position: prompt.count + step,
                    into: logits)
                let got = Int32(bitPattern: runner.lastGreedyToken)
                guard got == reference[step + 1] else {
                    throw PassCostError.diverged(
                        position: prompt.count + step + 1,
                        expected: reference[step + 1], actual: got)
                }
            }
        }
        try await advance(from: 0, count: warmup)
        var rows: [PassCostRow] = []
        var index = warmup
        for round in 0..<rounds {
            // Paths alternate every round and widths every `paths.count`
            // rounds, so each width's passes on the two paths are neighbours.
            let roundPath = paths[round % paths.count]
            let width = widths[(round / paths.count) % widths.count]
            let position = prompt.count + index
            let snapshot = try runner.captureInferenceState()
            let auxiliary = try runner.captureAuxiliaryState()
            try runner.prepareForContinuation(expectedPosition: position)
            let start = now()
            var row: PassCostRow
            switch roundPath {
            case .chunked, .both:
                let result = try await runner.prefillChunked(
                    tokens: reference[index..<(index + width)],
                    startPosition: position,
                    outputMode: .greedyIfAvailable, config: prefillConfig,
                    into: logits, onProgress: { _ in })
                row = PassCostRow(
                    path: .chunked, width: width, position: position,
                    passMilliseconds: ms(start),
                    lastRowMatches: seedToken(result) == reference[index + width])
            case .dedicated:
                let result = try await runner.verifyGreedyRows(
                    Array(reference[index..<(index + width)]), startPosition: position)
                row = PassCostRow(
                    path: .dedicated, width: width, position: position,
                    passMilliseconds: ms(start),
                    lastRowMatches: result.predictions.last == reference[index + width])
                row.headMilliseconds = Double(result.headNanos) / 1e6
                row.rowsMatching =
                    result.predictions.indices.filter {
                        result.predictions[$0] == reference[index + $0 + 1]
                    }.count
            }
            try runner.restoreInferenceState(snapshot, auxiliary: auxiliary)
            rows.append(row)
            try await advance(from: index, count: maxWidth)
            index += maxWidth
        }

        let report = summarizePassCost(
            install: install, paths: paths, widths: widths, promptTokens: prompt.count,
            scalar: scalar, rows: rows)
        printPassCost(report)
        try writePassCost(report)
    }

    static func summarizePassCost(
        install: String, paths: [PassCostPath], widths: [Int], promptTokens: Int,
        scalar: [Double], rows: [PassCostRow]
    ) -> PassCostReport {
        let fits = paths.map { path in
            let mine = rows.filter { $0.path == path }
            let n = Double(mine.count)
            let xs = mine.map { Double($0.width) }
            let ys = mine.map(\.passMilliseconds)
            let mx = xs.reduce(0, +) / max(n, 1)
            let my = ys.reduce(0, +) / max(n, 1)
            let sxy = zip(xs, ys).reduce(0) { $0 + ($1.0 - mx) * ($1.1 - my) }
            let sxx = xs.reduce(0) { $0 + ($1 - mx) * ($1 - mx) }
            let slope = sxx > 0 ? sxy / sxx : 0
            return PassCostFit(
                path: path, baseMilliseconds: my - slope * mx, perRowMilliseconds: slope)
        }
        return PassCostReport(
            install: install, paths: paths, widths: widths, promptTokens: promptTokens,
            scalarMillisecondsMedian: median(scalar),
            scalarMillisecondsMean: scalar.reduce(0, +) / Double(max(scalar.count, 1)),
            scalarSamples: scalar.count, rows: rows, fits: fits)
    }

    static func printPassCost(_ report: PassCostReport) {
        let scalar = report.scalarMillisecondsMedian
        print(
            String(
                format: "scalar token: median %.1f ms, mean %.1f ms (%d samples)",
                scalar, report.scalarMillisecondsMean, report.scalarSamples))
        for fit in report.fits {
            let path = fit.path
            print("path: \(path.rawValue)")
            print("width  passes  median ms  scalar tokens  last-row match  head ms  rows = scalar")
            for width in report.widths {
                let rows = report.rows.filter { $0.path == path && $0.width == width }
                let med = median(rows.map(\.passMilliseconds))
                let matched = rows.filter(\.lastRowMatches).count
                let head = rows.compactMap(\.headMilliseconds)
                let rowMatches = rows.compactMap(\.rowsMatching).reduce(0, +)
                let detail =
                    head.isEmpty
                    ? "      -  -"
                    : String(format: "%7.1f  %d/%d", median(head), rowMatches, width * rows.count)
                print(
                    String(
                        format: "%5d  %6d  %9.1f  %13.2f  %d/%d",
                        width, rows.count, med, med / scalar, matched, rows.count)
                        + "  " + detail)
            }
            let w8 = median(
                report.rows.filter { $0.path == path && $0.width == 8 }.map(\.passMilliseconds))
            let verdict = w8 / scalar <= 4.5 ? "PASS" : "STOP"
            print(
                String(
                    format:
                        "fit: %.0f ms + %.1f ms/row; gate 2 (W=8 <= 4.5 scalar tokens): %.2f -> %@",
                    fit.baseMilliseconds, fit.perRowMilliseconds, w8 / scalar, verdict))
        }
    }

    static func writePassCost(_ report: PassCostReport) throws {
        let directory = URL(fileURLWithPath: ".build/pass-cost", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "UTC")
        formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        let url = directory.appendingPathComponent(
            "passcost-\(formatter.string(from: Date())).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: url)
        print("wrote \(url.path)")
    }

    static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }

    static func argmaxHalf(_ buffer: MTLBuffer, count: Int) -> Int32 {
        let values = buffer.contents().assumingMemoryBound(to: Float16.self)
        var best = 0
        var bestValue = Float(values[0])
        for index in 1..<count where Float(values[index]) > bestValue {
            best = index
            bestValue = Float(values[index])
        }
        return Int32(best)
    }
}
