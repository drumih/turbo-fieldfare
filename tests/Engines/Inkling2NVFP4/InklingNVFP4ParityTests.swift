import Foundation
import Testing
import TinyTitanInkling2

@testable import Inkling2NVFP4

/// TT-125, model-gated:
/// - `TINYTITAN_INKLING2_NVFP4_INSTALL` + `TINYTITAN_INKLING2_Q4_INSTALL`: the Swift
///   repack of real NVFP4 experts equals the Q4 install's bytes for the same
///   experts (the two installs hold the same weights).
/// - `TINYTITAN_INKLING2_NVFP4_INSTALL` + `TINYTITAN_INKLING2_MODEL` (the raw
///   checkpoint): the plugin's decoder, teacher-forced on the llama.cpp
///   reference cases, chooses the same token as the checkpoint's own reference
///   implementation (`Engines/Inkling2`) at every step.
@Suite(.serialized)
struct InklingNVFP4ParityTests {
    static let environment = ProcessInfo.processInfo.environment
    static var nvfp4: URL? {
        environment["TINYTITAN_INKLING2_NVFP4_INSTALL"].map(URL.init(fileURLWithPath:))
    }
    static var q4: URL? {
        environment["TINYTITAN_INKLING2_Q4_INSTALL"].map(URL.init(fileURLWithPath:))
    }
    static var raw: URL? { environment["TINYTITAN_INKLING2_MODEL"].map(URL.init(fileURLWithPath:)) }

    static func log(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    @Test(.enabled(if: nvfp4 != nil && q4 != nil))
    func repackedExpertsEqualTheQ4Install() throws {
        let install = try InklingQ4Install(directory: try #require(Self.nvfp4))
        let q4Root = try #require(Self.q4)
        let q4Layout = try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: q4Root.appendingPathComponent("packed_experts/layout.json")))
                as? [String: Any])
        let q4Layers = try #require(q4Layout["layers"] as? [[String: Any]])
        var compared = 0
        for plan in install.slotPlans where plan.kind == .nvfp4 && [0, 9, 19].contains(plan.layer) {
            let q4Layer = try #require(q4Layers.first { ($0["layer"] as? Int) == plan.layer })
            let q4Experts = try #require(q4Layer["experts"] as? [[String: Any]])
            let q4File = q4Root.appendingPathComponent(
                "packed_experts/" + (q4Layer["file"] as? String ?? ""))
            for expert in [0, 64, 127] {
                let file = try FileHandle(
                    forReadingFrom: install.directory.appendingPathComponent(
                        "packed_experts/" + plan.file))
                defer { try? file.close() }
                try file.seek(toOffset: UInt64(expert * plan.fileStride))
                let planar = try #require(try file.read(upToCount: plan.fileStride))
                let q4Expert = q4Experts[expert]
                let q4Offset = try #require(q4Expert["offset"] as? Int)
                let q4Tensors = try #require(q4Expert["tensors"] as? [String: [String: Any]])
                let q4Handle = try FileHandle(forReadingFrom: q4File)
                defer { try? q4Handle.close() }
                for (name, matrix, interleaved) in [
                    ("gate_up", plan.gateUp, true), ("down", plan.down, false),
                ] {
                    var mine = [UInt8](repeating: 0, count: matrix.slotBytes)
                    planar.withUnsafeBytes { raw in
                        mine.withUnsafeMutableBytes { out in
                            guard let base = raw.baseAddress, let target = out.baseAddress else {
                                return
                            }
                            InklingNVFP4Repack.nvfp4(
                                values: base + matrix.valuesOffset,
                                scales: base + matrix.scalesOffset,
                                rows: matrix.rows, columns: matrix.columns,
                                interleaved: interleaved,
                                into: target)
                        }
                    }
                    let tensor = try #require(q4Tensors[name])
                    try q4Handle.seek(toOffset: UInt64(q4Offset + (tensor["offset"] as? Int ?? 0)))
                    let theirs = try #require(try q4Handle.read(upToCount: matrix.slotBytes))
                    #expect(Data(mine) == theirs, "layer \(plan.layer) expert \(expert) \(name)")
                    compared += 1
                }
            }
        }
        #expect(compared == 18)
    }

    struct Reference: Decodable {
        struct Step: Decodable { let id: Int }
        struct Case: Decodable {
            let prompt_ids: [Int]
            let steps: [Step]
        }
        let cases: [Case]
    }

    static func reference() throws -> Reference {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent(
                "TinyTitan/Inkling2GGUF/Inkling2GGUFReferences/q4kxl-reference.json")
        return try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
    }

    /// The greedy choice after every teacher-forced position of every case.
    static func oracleTokens(_ raw: URL, _ cases: [Reference.Case]) throws -> [Int] {
        try autoreleasepool {
            let model = try Inkling2TextModel(
                directory: raw, config: try Inkling2Config.load(directory: raw),
                store: try Inkling2ExpertStore(directory: raw))
            var chosen: [Int] = []
            for item in cases {
                var caches = model.makeCaches()
                var logits: [Float] = []
                for token in item.prompt_ids {
                    logits = try model.forward(token: token, caches: &caches)
                }
                for (index, step) in item.steps.enumerated() {
                    chosen.append(Inkling2TextModel.greedyToken(logits))
                    if index + 1 < item.steps.count {
                        logits = try model.forward(token: step.id, caches: &caches)
                    }
                }
            }
            return chosen
        }
    }

    /// The plugin's greedy choices, and at each its logit gap over the
    /// llama.cpp reference token (0 where they agree).
    static func pluginTokens(_ install: URL, _ cases: [Reference.Case]) throws
        -> (tokens: [Int], gaps: [Float])
    {
        try autoreleasepool {
            let decoder = try InklingQ4Decoder(
                directory: install, maxContext: 256, maxChunk: 64, experts: .slots(16))
            // A command buffer per visit: the GPU never waits on the CPU's expert
            // repack (the logits are the same bits either way).
            decoder.eventRouting = false
            var chosen: [Int] = []
            var gaps: [Float] = []
            for item in cases {
                decoder.state.reset()
                var result = try decoder.prefill(item.prompt_ids, position: 0, returnLogits: true)
                var position = item.prompt_ids.count
                for (index, step) in item.steps.enumerated() {
                    let logits = try #require(result.logits)
                    chosen.append(result.next)
                    gaps.append(logits[result.next] - logits[step.id])
                    if index + 1 < item.steps.count {
                        result = try decoder.step(
                            token: step.id, position: position, returnLogits: true)
                        position += 1
                    }
                }
            }
            return (chosen, gaps)
        }
    }

    /// Two references, measured 2026-11-24: llama.cpp on the Q4 GGUF (the graph
    /// this plugin runs) and the raw-checkpoint route (`Engines/Inkling2`). They
    /// disagree with each other on 7 of 96 positions, where the raw route picks
    /// rare high ids. Wherever both agree, the plugin must agree too, or the
    /// difference must be a near-tie (under 0.5 logit) between the plugin's
    /// BF16/MXFP8 weights and the GGUF's Q8_0: measured, one position at 0.007.
    @Test(.enabled(if: nvfp4 != nil && raw != nil))
    func greedyTokensAgreeWithBothReferences() throws {
        let cases = try Self.reference().cases
        let oracle = try Self.oracleTokens(try #require(Self.raw), cases)
        let (plugin, gaps) = try Self.pluginTokens(try #require(Self.nvfp4), cases)
        let llama = cases.flatMap { $0.steps.map(\.id) }
        let count = { (a: [Int], b: [Int]) in zip(a, b).filter { $0 == $1 }.count }
        Self.log(
            "Inkling NVFP4 plugin: greedy agreement with llama.cpp \(count(plugin, llama))/"
                + "\(llama.count), with the raw-checkpoint route \(count(plugin, oracle))/"
                + "\(oracle.count); the two references agree on \(count(oracle, llama))")
        #expect(plugin.count == llama.count && oracle.count == llama.count)
        for index in llama.indices
        where oracle[index] == llama[index] && plugin[index] != llama[index] {
            Self.log("  step \(index): plugin \(plugin[index]) against both, gap \(gaps[index])")
            #expect(gaps[index] < 0.5, "step \(index) is not a near-tie (\(gaps[index]))")
        }
    }
}

extension InklingNVFP4ParityTests {
    /// Where the plugin and llama.cpp (on the Q4 GGUF) choose differently, how
    /// far apart the two candidates are in the plugin's own logits: a near-tie
    /// is a precision difference between BF16/MXFP8 and the GGUF's Q8_0, a
    /// wide margin would be a defect.
    @Test(.enabled(if: nvfp4 != nil))
    func disagreementsWithLlamaCppAreNearTies() throws {
        let cases = try Self.reference().cases
        let decoder = try InklingQ4Decoder(
            directory: try #require(Self.nvfp4), maxContext: 256, maxChunk: 64,
            experts: .slots(16))
        decoder.eventRouting = false
        var step = 0
        var worst: Float = 0
        for item in cases {
            decoder.state.reset()
            var result = try decoder.prefill(item.prompt_ids, position: 0, returnLogits: true)
            var position = item.prompt_ids.count
            for (index, expected) in item.steps.enumerated() {
                let logits = try #require(result.logits)
                if result.next != expected.id {
                    let gap = logits[result.next] - logits[expected.id]
                    worst = max(worst, gap)
                    Self.log(
                        "  step \(step): plugin \(result.next) (\(logits[result.next])), llama.cpp "
                            + "\(expected.id) (\(logits[expected.id])), gap \(gap)")
                }
                step += 1
                if index + 1 < item.steps.count {
                    result = try decoder.step(
                        token: expected.id, position: position, returnLogits: true)
                    position += 1
                }
            }
        }
        Self.log("Inkling NVFP4 plugin: largest logit gap where it differs from llama.cpp \(worst)")
        #expect(worst < 0.5, "a disagreement with llama.cpp is not a near-tie (\(worst))")
    }
}

extension InklingNVFP4ParityTests {
    /// TT-135: the CPU cost of repacking one streamed expert, the part of a
    /// cache miss the NVFP4 plugin adds over the Q4 plugin. Diagnostic: enabled
    /// with `TINYTITAN_INKLING2_REPACK_TIMING=1`.
    @Test(.enabled(if: nvfp4 != nil && environment["TINYTITAN_INKLING2_REPACK_TIMING"] == "1"))
    func repackCostPerExpert() throws {
        let install = try InklingQ4Install(directory: try #require(Self.nvfp4))
        let plan = try #require(install.slotPlans.first { $0.layer == 0 })
        let file = try FileHandle(
            forReadingFrom: install.directory.appendingPathComponent("packed_experts/" + plan.file))
        defer { try? file.close() }
        let planar = try #require(try file.read(upToCount: plan.fileStride))
        var slot = [UInt8](repeating: 0, count: plan.slotStride)
        typealias Repack = (
            UnsafeRawPointer, UnsafeRawPointer, Int, Int, Bool, UnsafeMutableRawPointer
        ) -> Void
        let variants: [(String, Repack)] = [
            (
                "nibble reference",
                {
                    InklingNVFP4Repack.nvfp4Reference(
                        values: $0, scales: $1, rows: $2, columns: $3, interleaved: $4, into: $5)
                }
            ),
            (
                "word-wide",
                {
                    InklingNVFP4Repack.nvfp4(
                        values: $0, scales: $1, rows: $2, columns: $3, interleaved: $4, into: $5)
                }
            ),
        ]
        var totals = [Double](repeating: 0, count: variants.count)
        let rounds = 10
        // Interleaved rounds, so machine load hits both alike.
        for _ in 0..<rounds {
            for (index, variant) in variants.enumerated() {
                let started = Date()
                planar.withUnsafeBytes { raw in
                    slot.withUnsafeMutableBytes { out in
                        guard let base = raw.baseAddress, let target = out.baseAddress else {
                            return
                        }
                        for (matrix, interleaved) in [(plan.gateUp, true), (plan.down, false)] {
                            variant.1(
                                base + matrix.valuesOffset, base + matrix.scalesOffset,
                                matrix.rows, matrix.columns, interleaved,
                                target + matrix.slotOffset)
                        }
                    }
                }
                totals[index] += Date().timeIntervalSince(started)
            }
        }
        for (index, variant) in variants.enumerated() {
            Self.log(
                "Inkling NVFP4 repack (layer 0 expert), \(variant.0): "
                    + "\(totals[index] / Double(rounds) * 1000) ms per expert")
        }
        #expect(totals.allSatisfy { $0 > 0 })
    }
}

extension InklingNVFP4ParityTests {
    /// TT-135 diagnostic: greedy decode at the 12G plan's 59 slots, with the
    /// cache's counters, to show where a token's time goes. Enabled with
    /// `TINYTITAN_INKLING2_DECODE_TIMING=1`; an observation, not a benchmark.
    @Test(.enabled(if: nvfp4 != nil && environment["TINYTITAN_INKLING2_DECODE_TIMING"] == "1"))
    func decodeTimingAtTheTwelveGigabytePlan() throws {
        let decoder = try InklingQ4Decoder(
            directory: try #require(Self.nvfp4), maxContext: 4_096, maxChunk: 64,
            experts: .slots(59))
        let prompt = try Self.reference().cases[3].prompt_ids
        var result = try decoder.prefill(prompt, position: 0)
        var position = prompt.count
        let cache = try #require(decoder.expertCache)
        let before = cache.statistics
        let count = 48
        let started = Date()
        for _ in 0..<count {
            result = try decoder.step(token: result.next, position: position)
            position += 1
        }
        let seconds = Date().timeIntervalSince(started)
        let after = cache.statistics
        Self.log(
            "Inkling NVFP4 decode at 59 slots: \(Double(count) / seconds) tok/s over \(count) tokens; "
                + "misses \(after.misses - before.misses), blocking reads "
                + "\(after.readSeconds - before.readSeconds) s, prefetch wait "
                + "\(after.prefetchWaitSeconds - before.prefetchWaitSeconds) s")
        #expect(seconds > 0)
    }
}
