import CoreML
import Foundation
import Metal
import Testing
import TinyTitanCore

@testable import TinyTitan

/// A sidecar metadata geometry block, as `tools/export_ane_prefill.py` writes
/// it. The defaults are the 35B-A3B row; tests that use another shape pass it.
private func sidecarGeometry(
    family: ModelFamily = .qwen36, hidden: Int = 2048,
    numHeads: Int = 16, numKVHeads: Int = 2,
    headDim: Int = 256, chunkTokens: Int = 4096,
    layers: [Int] = [3]
) -> [String: Any] {
    [
        "family": family.rawValue, "hiddenSize": hidden, "numHeads": numHeads,
        "numKVHeads": numKVHeads, "headDim": headDim,
        "chunkTokens": chunkTokens, "fullAttentionLayers": layers,
    ]
}

/// A model's full-attention mask: 1 at `layers`, 2 (linear) elsewhere.
private func fullMask(layers: [Int], count: Int = 40) -> [UInt8] {
    var mask = [UInt8](repeating: 2, count: count)
    for layer in layers where layer >= 0 && layer < count { mask[layer] = 1 }
    return mask
}

/// A sparse-indexed (Qwen 3.8-shaped) sidecar over a toy geometry, with the
/// model's own indexer attached, so the fold can be exercised without Core ML,
/// model weights or a real indexer pass.
private func sparseANE(
    context: MetalContext, chunk: Int, layers: [Int],
    budget: Int, compressRatio: Int
)
    throws -> (URL, ANEPrefillAttention)
{
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("ane-test-\(UUID().uuidString)")
    let sidecar = dir.appendingPathComponent("ane_prefill")
    try FileManager.default.createDirectory(
        at: sidecar,
        withIntermediateDirectories: true)
    let geometry = sidecarGeometry(
        family: .qwen38flash, hidden: 16,
        numHeads: 1, numKVHeads: 1, headDim: 4,
        chunkTokens: chunk, layers: layers)
    let meta: [String: Any] = [
        "version": 1, "family": "qwen38flash", "chunkTokens": chunk,
        "histories": [0, chunk], "layers": layers, "geometry": geometry,
        "aneCompileVerified": true, "selectionFolded": true,
    ]
    try JSONSerialization.data(withJSONObject: meta)
        .write(to: sidecar.appendingPathComponent("ane_prefill.json"))
    let indexer = SparseIndexerConfig(
        numHeads: 1, numKVHeads: 1, headDim: 4,
        budget: budget,
        compressRatio: compressRatio)
    let ane = try ANEPrefillAttention(
        modelDirectory: dir, device: context.device,
        hiddenSize: 16, kvDim: 4, weightsSha256: nil,
        family: .qwen38flash,
        fullAttentionLayerMask: fullMask(
            layers: layers,
            count: 8),
        sparseIndexer: indexer, configChunkTokens: chunk)
    return (dir, ane)
}

/// A `QSASelection` whose compacted form is `keep(row)`, ascending. The byte
/// mask is filled to match for realism, but the fold reads the compacted form —
/// which is exactly what the GPU's attention gathers.
private func makeSelection(
    device: MTLDevice, rows: Int, maskStride: Int,
    indexStride: Int,
    keep: (Int) -> [Int]
) throws -> QSASelection {
    guard
        let mask = device.makeBuffer(
            length: max(1, rows * maskStride),
            options: .storageModeShared),
        let indices = device.makeBuffer(
            length: max(1, rows * indexStride * MemoryLayout<UInt32>.stride),
            options: .storageModeShared),
        let counts = device.makeBuffer(
            length: max(1, rows * MemoryLayout<UInt32>.stride),
            options: .storageModeShared)
    else {
        throw ModelError.residentBufferWrapFailed
    }
    let maskPtr = mask.contents().bindMemory(
        to: UInt8.self,
        capacity: max(1, rows * maskStride))
    let indexPtr = indices.contents().bindMemory(
        to: UInt32.self,
        capacity: max(1, rows * indexStride))
    let countPtr = counts.contents().bindMemory(
        to: UInt32.self,
        capacity: max(1, rows))
    for row in 0..<rows {
        let kept = keep(row)
        for (slot, key) in kept.enumerated() {
            guard slot < indexStride else { break }
            maskPtr[row * maskStride + key] = 1
            indexPtr[row * indexStride + slot] = UInt32(key)
        }
        countPtr[row] = UInt32(min(kept.count, indexStride))
    }
    return QSASelection(
        mask: mask, maskStride: maskStride, indices: indices,
        indexStride: indexStride, counts: counts)
}

@Suite struct ANEPrefillAttentionTests {
    // Default-on since the deferred-pin A/Bs qualified it on an idle machine:
    // 3.14x end to end at 4-bit, 1.91x at 8-bit. A model with no sidecar still
    // loads -- the runner degrades to the GPU unless the setting was named
    // explicitly.
    @Test func environmentSwitchDefaultsOnAndFailsClosed() throws {
        #expect(try RuntimePrefillANE.environmentValue([:]) == .on)
        #expect(!RuntimePrefillANE.wasRequestedExplicitly([:]))
        #expect(
            RuntimePrefillANE.wasRequestedExplicitly(
                ["TINYTITAN_PREFILL_ANE": "off"]))
        #expect(
            try RuntimePrefillANE.environmentValue(
                ["TINYTITAN_PREFILL_ANE": "off"]) == .off)
        #expect(
            try RuntimePrefillANE.environmentValue(
                ["TINYTITAN_PREFILL_ANE": "on"]) == .on)
        #expect(throws: PrefillError.self) {
            try RuntimePrefillANE.environmentValue(["TINYTITAN_PREFILL_ANE": "1"])
        }
        #expect(throws: PrefillError.self) {
            try RuntimePrefillANE.environmentValue(["TINYTITAN_PREFILL_ANE": ""])
        }
    }

    /// The mask mode is a verification control, not a tuning knob: `folded` is
    /// the shipped behaviour and `causal` feeds the mask the path would build
    /// without the fold, which is wrong for a sparse-indexed model. Nonsense is
    /// refused rather than silently treated as the default.
    @Test func theMaskModeControlDefaultsToFoldedAndRejectsNonsense() throws {
        #expect(try ANEPrefillAttention.MaskMode.environmentValue([:]) == .folded)
        #expect(
            try ANEPrefillAttention.MaskMode.environmentValue(
                ["TINYTITAN_ANE_MASK": "folded"]) == .folded)
        #expect(
            try ANEPrefillAttention.MaskMode.environmentValue(
                ["TINYTITAN_ANE_MASK": "causal"]) == .causal)
        #expect(throws: PrefillError.self) {
            try ANEPrefillAttention.MaskMode.environmentValue(
                ["TINYTITAN_ANE_MASK": "causal-only"])
        }
    }

    @Test func missingSidecarFailsClosedWithExportHint() throws {
        let ctx = try MetalContext()
        let empty = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        #expect(throws: PrefillError.self) {
            _ = try ANEPrefillAttention(
                modelDirectory: empty,
                device: ctx.device,
                hiddenSize: 2048, kvDim: 512,
                weightsSha256: nil,
                family: .qwen36,
                fullAttentionLayerMask: fullMask(layers: [3]),
                sparseIndexer: .none,
                configChunkTokens: 4096)
        }
    }

    /// The sidecar read was an uncapped `Data(contentsOf:)`. Bounded like every
    /// other metadata reader here, and refused by name so an oversized export
    /// is not mistaken for a corrupt one.
    @Test func anOversizedSidecarIsRefusedByTheBoundNotByTheDecoder() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-bound-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let metaURL = dir.appendingPathComponent("ane_prefill.json")
        var meta: [String: Any] = [
            "version": 1, "family": "qwen38flash", "chunkTokens": 4096,
            "histories": [0, 4096], "layers": [3, 7],
            "aneCompileVerified": true, "selectionFolded": true,
        ]
        // Padding the reader ignores and the bound does not.
        meta["pad"] = String(repeating: "x", count: 512)
        try JSONSerialization.data(withJSONObject: meta).write(to: metaURL)
        let size = try Data(contentsOf: metaURL).count
        #expect(size > 64, "the fixture is too small to cross a 64-byte bound")

        var described = ""
        do {
            _ = try ANEPrefillAttention.loadSidecarMetadata(at: metaURL, maxBytes: 64)
        } catch {
            described = String(describing: error)
        }
        #expect(
            described.contains("\(size)") && described.contains("over the 64-byte metadata bound"),
            "the refusal named neither the size nor the cap: \(described)")
        #expect(try ANEPrefillAttention.loadSidecarMetadata(at: metaURL).version == 1)
    }

    @Test func sidecarExportedFromDifferentWeightsIsRejected() throws {
        let ctx = try MetalContext()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        let sidecar = dir.appendingPathComponent("ane_prefill")
        try FileManager.default.createDirectory(
            at: sidecar, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let meta: [String: Any] = [
            "version": 1, "family": "qwen36", "chunkTokens": 4096,
            "histories": [0], "layers": [3],
            "geometry": sidecarGeometry(),
            "weightsSha256": String(repeating: "a", count: 64),
            "aneCompileVerified": true,
        ]
        try JSONSerialization.data(withJSONObject: meta)
            .write(to: sidecar.appendingPathComponent("ane_prefill.json"))
        // Matching digest loads; a different one must fail closed rather than
        // computing plausible-looking attention from the wrong weights.
        _ = try ANEPrefillAttention(
            modelDirectory: dir, device: ctx.device,
            hiddenSize: 2048, kvDim: 512,
            weightsSha256: String(repeating: "A", count: 64),
            family: .qwen36,
            fullAttentionLayerMask: fullMask(layers: [3]),
            sparseIndexer: .none,
            configChunkTokens: 4096)
        #expect(throws: PrefillError.self) {
            _ = try ANEPrefillAttention(
                modelDirectory: dir, device: ctx.device,
                hiddenSize: 2048, kvDim: 512,
                weightsSha256: String(repeating: "b", count: 64),
                family: .qwen36,
                fullAttentionLayerMask: fullMask(layers: [3]),
                sparseIndexer: .none,
                configChunkTokens: 4096)
        }
    }

    /// A sidecar is built for one geometry. Running one that belongs to another
    /// model -- another width, head split, family or layer set -- computes a
    /// *different* attention and produces fluent output nothing flags, so each
    /// disagreement must be refused and send the runner to the GPU path.
    @Test func sidecarForAnotherGeometryIsRejected() throws {
        let ctx = try MetalContext()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        let sidecar = dir.appendingPathComponent("ane_prefill")
        try FileManager.default.createDirectory(
            at: sidecar, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let metaURL = sidecar.appendingPathComponent("ane_prefill.json")
        func write(_ geometry: [String: Any], layers: [Int] = [3]) throws {
            let meta: [String: Any] = [
                "version": 1, "family": "qwen36", "chunkTokens": 4096,
                "histories": [0], "layers": layers, "geometry": geometry,
                "aneCompileVerified": true,
            ]
            try JSONSerialization.data(withJSONObject: meta).write(to: metaURL)
        }
        func load() throws {
            _ = try ANEPrefillAttention(
                modelDirectory: dir, device: ctx.device,
                hiddenSize: 2048, kvDim: 512,
                weightsSha256: nil,
                family: .qwen36,
                fullAttentionLayerMask: fullMask(layers: [3, 7]),
                sparseIndexer: .none,
                configChunkTokens: 4096)
        }
        // No geometry block at all (a sidecar from the qwen36-only exporter).
        let legacy: [String: Any] = [
            "version": 1, "family": "qwen36", "chunkTokens": 4096,
            "histories": [0], "layers": [3], "aneCompileVerified": true,
        ]
        try JSONSerialization.data(withJSONObject: legacy).write(to: metaURL)
        #expect(throws: PrefillError.self) { try load() }

        // Another hidden width (the 4B's 2560 against this 2048 model).
        try write(sidecarGeometry(hidden: 2560))
        #expect(throws: PrefillError.self) { try load() }

        // Another head split: 4 kv heads x 256 is the 4B's kvDim, not 512.
        try write(sidecarGeometry(numKVHeads: 4))
        #expect(throws: PrefillError.self) { try load() }

        // Another family's sidecar.
        try write(sidecarGeometry(family: .qwen35Dense))
        #expect(throws: PrefillError.self) { try load() }

        // A layer that is not full attention in this model.
        try write(sidecarGeometry(), layers: [4])
        #expect(throws: PrefillError.self) { try load() }

        // The matching geometry loads.
        try write(sidecarGeometry())
        try load()
    }

    /// Qwen 3.8: a sparse-indexed model is served by folding the indexer's key
    /// selection into the mask the sidecar is fed, so a sidecar with the
    /// matching geometry **loads** — but only one that records the contract.
    /// Loaded with a causal-only mask it would attend to keys the model drops
    /// past the 2,051 visible keys where dense attention stops being exact, and
    /// nothing downstream would flag it.
    @Test func aSparseIndexedModelLoadsOnlyAFoldedSidecar() throws {
        let ctx = try MetalContext()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        let sidecar = dir.appendingPathComponent("ane_prefill")
        try FileManager.default.createDirectory(
            at: sidecar, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let metaURL = sidecar.appendingPathComponent("ane_prefill.json")
        func write(_ folded: Bool?) throws {
            var meta: [String: Any] = [
                "version": 1, "family": "qwen38flash", "chunkTokens": 4096,
                "histories": [0], "layers": [3],
                "geometry": sidecarGeometry(family: .qwen38flash), "aneCompileVerified": true,
            ]
            if let folded { meta["selectionFolded"] = folded }
            try JSONSerialization.data(withJSONObject: meta).write(to: metaURL)
        }
        let indexer = SparseIndexerConfig(
            numHeads: 24, numKVHeads: 2,
            headDim: 256, budget: 2048,
            compressRatio: 4)
        func load() throws -> ANEPrefillAttention {
            try ANEPrefillAttention(
                modelDirectory: dir, device: ctx.device,
                hiddenSize: 2048, kvDim: 512, weightsSha256: nil,
                family: .qwen38flash, fullAttentionLayerMask: fullMask(layers: [3]),
                sparseIndexer: indexer, configChunkTokens: 4096)
        }
        #expect(QSAExactness(indexer).maximumExactVisibleKeys == 2_051)

        // No record at all (a sidecar from an exporter that did not know about
        // the fold): refused, naming the fix.
        try write(nil)
        #expect(throws: PrefillError.self) { _ = try load() }
        // Explicitly false: the same refusal.
        try write(false)
        #expect(throws: PrefillError.self) { _ = try load() }

        // The contract: loads, and says the fold is required.
        try write(true)
        let ane = try load()
        #expect(ane.requiresSelection)
        #expect(ane.exactVisibleKeys == 2_051)

        // The same sidecar on a model with no indexer needs no fold.
        let denseANE = try ANEPrefillAttention(
            modelDirectory: dir, device: ctx.device,
            hiddenSize: 2048, kvDim: 512, weightsSha256: nil,
            family: .qwen38flash, fullAttentionLayerMask: fullMask(layers: [3]),
            sparseIndexer: .none, configChunkTokens: 4096)
        #expect(!denseANE.requiresSelection)
    }

    /// The fold is the whole wiring: the sidecar's mask input is arbitrary, so
    /// a sparse model's selection is written into the same `-30000` the causal
    /// mask uses, and the graph is untouched. A wrong fold attends to keys the
    /// model drops and nothing downstream flags it, so the arithmetic is
    /// checked directly on a chunk small enough to read.
    @Test func theSelectionIsFoldedIntoTheAdditiveMask() throws {
        let ctx = try MetalContext()
        let (dir, ane) = try sparseANE(
            context: ctx, chunk: 8, layers: [3],
            budget: 4, compressRatio: 2)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ane.requiresSelection)
        #expect(ane.exactVisibleKeys == 5)

        let history = 8
        let chunk = 8
        let total = history + chunk
        let indexStride = 5
        var expected: [Set<Int>] = []
        let selection = try makeSelection(
            device: ctx.device, rows: chunk, maskStride: total,
            indexStride: indexStride
        ) { row in
            // The query's own key is always kept (the "ragged tail"), plus
            // key 0 standing in for a block the indexer ranked in; every
            // other key of the row is dropped.
            let visible = history + row + 1
            let kept: Set<Int> = [0, visible - 1]
            expected.append(kept)
            return Array(kept).sorted()
        }
        let array = try ane.selectionMask(
            history: history, tokenCount: chunk,
            selection: selection)
        #expect(array.shape == [1, 1, 8, 16])
        let values = array.dataPointer.bindMemory(
            to: Float16.self,
            capacity: chunk * total)
        for row in 0..<chunk {
            for column in 0..<total {
                let want: Float16 =
                    expected[row].contains(column)
                    ? 0 : ANEPrefillAttention.maskNegative
                #expect(
                    values[row * total + column] == want,
                    "row \(row) column \(column) should be \(want)")
            }
        }
    }

    /// A chunk's last pass is partial: the selection covers `tokenCount` rows
    /// and only `history + tokenCount` columns, and everything outside it must
    /// be masked rather than left as whatever a previous layer wrote. Padding
    /// rows are zero queries, whose all-masked row softmaxes uniformly — finite
    /// and discarded — so they must not carry a real row's selection.
    @Test func partialChunksMaskTheirPaddingAndUnselectedColumns() throws {
        let ctx = try MetalContext()
        let (dir, ane) = try sparseANE(
            context: ctx, chunk: 8, layers: [3],
            budget: 4, compressRatio: 2)
        defer { try? FileManager.default.removeItem(at: dir) }
        let history = 8
        let chunk = 8
        let total = history + chunk
        let tokenCount = 3
        let indexStride = 5
        let selection = try makeSelection(
            device: ctx.device, rows: tokenCount,
            maskStride: history + tokenCount, indexStride: indexStride
        ) { row in
            [0, history + row]
        }
        let array = try ane.selectionMask(
            history: history,
            tokenCount: tokenCount,
            selection: selection)
        let values = array.dataPointer.bindMemory(
            to: Float16.self,
            capacity: chunk * total)
        for row in 0..<chunk {
            for column in 0..<total {
                var want = ANEPrefillAttention.maskNegative
                if row < tokenCount,
                    column == 0 || column == history + row
                {
                    want = 0
                }
                #expect(
                    values[row * total + column] == want,
                    "row \(row) column \(column) should be \(want)")
            }
        }
    }

    /// Eligibility and shadow continuity, using a synthetic sidecar manifest
    /// so no Core ML package or model weights are involved.
    @Test func chunkEligibilityEnforcesAlignmentCoverageAndContinuity() throws {
        let ctx = try MetalContext()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        let sidecar = dir.appendingPathComponent("ane_prefill")
        try FileManager.default.createDirectory(
            at: sidecar, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let meta: [String: Any] = [
            "version": 1, "family": "qwen36", "chunkTokens": 4096,
            "histories": [0, 4096], "layers": [3, 7], "aneCompileVerified": true,
            "geometry": sidecarGeometry(layers: [3, 7]),
        ]
        try JSONSerialization.data(withJSONObject: meta)
            .write(to: sidecar.appendingPathComponent("ane_prefill.json"))
        let ane = try ANEPrefillAttention(
            modelDirectory: dir,
            device: ctx.device,
            hiddenSize: 2048, kvDim: 512,
            weightsSha256: nil,
            family: .qwen36,
            fullAttentionLayerMask: fullMask(layers: [3, 7]),
            sparseIndexer: .none,
            configChunkTokens: 4096)
        #expect(ane.maxPromptTokens == 8192)
        #expect(ane.coveredLayers == Set([3, 7]))

        // Config chunk mismatch, misaligned start, uncovered history: all out.
        #expect(
            !ane.eligibleChunk(
                startPosition: 0, tokenCount: 512,
                configChunkTokens: 1024))
        #expect(
            !ane.eligibleChunk(
                startPosition: 100, tokenCount: 4096,
                configChunkTokens: 4096))
        #expect(
            !ane.eligibleChunk(
                startPosition: 8192, tokenCount: 100,
                configChunkTokens: 4096))

        // A short single-chunk prompt stays on the GPU (padding waste).
        #expect(
            !ane.eligibleChunk(
                startPosition: 0, tokenCount: 512,
                configChunkTokens: 4096))
        // Fresh full-chunk prompt resets the shadow and is eligible.
        #expect(
            ane.eligibleChunk(
                startPosition: 0, tokenCount: 4096,
                configChunkTokens: 4096))
        // Without finishChunk, a follow-up chunk must fall back (continuity).
        #expect(
            !ane.eligibleChunk(
                startPosition: 4096, tokenCount: 100,
                configChunkTokens: 4096))
        ane.finishChunk(startPosition: 0, tokenCount: 4096)
        #expect(ane.shadowTokens == 4096)
        #expect(
            ane.eligibleChunk(
                startPosition: 4096, tokenCount: 100,
                configChunkTokens: 4096))
        // A partial final chunk clears the shadow: nothing may resume it.
        ane.finishChunk(startPosition: 4096, tokenCount: 100)
        #expect(ane.shadowTokens == 0)
    }

    /// Issue #7: an exporter that did not watch the ANE compile can write a
    /// sidecar the ANE refuses, which Core ML then runs on the CPU at ~38x the
    /// GPU prefill cost while exiting 0. The runtime must not trust it.
    @Test func sidecarWithoutVerifiedANECompilationIsRejected() throws {
        let ctx = try MetalContext()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        let sidecar = dir.appendingPathComponent("ane_prefill")
        try FileManager.default.createDirectory(
            at: sidecar, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let metaURL = sidecar.appendingPathComponent("ane_prefill.json")
        func write(_ meta: [String: Any]) throws {
            try JSONSerialization.data(withJSONObject: meta).write(to: metaURL)
        }
        let base: [String: Any] = [
            "version": 1, "family": "qwen36", "chunkTokens": 4096,
            "histories": [0], "layers": [3],
            "geometry": sidecarGeometry(),
        ]
        func load() throws {
            _ = try ANEPrefillAttention(
                modelDirectory: dir, device: ctx.device,
                hiddenSize: 2048, kvDim: 512,
                weightsSha256: nil,
                family: .qwen36,
                fullAttentionLayerMask: fullMask(layers: [3]),
                sparseIndexer: .none,
                configChunkTokens: 4096)
        }
        // Missing flag (any sidecar written before this check): refused.
        try write(base)
        #expect(throws: PrefillError.self) { try load() }
        // Explicitly false (an export that saw the ANE refuse): refused.
        var failed = base
        failed["aneCompileVerified"] = false
        try write(failed)
        #expect(throws: PrefillError.self) { try load() }
        // Verified: loads.
        var verified = base
        verified["aneCompileVerified"] = true
        try write(verified)
        try load()
    }

    /// A model may carry one sidecar per chunk width, because the width that
    /// wins depends on the prompt: 4,096 for long ones, a smaller chunk to
    /// reach the band under it at all.
    @Test func theConfiguredChunkSelectsTheSidecarDirectory() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let fallback = dir.appendingPathComponent("ane_prefill")
        let narrow = dir.appendingPathComponent("ane_prefill-1024")
        try FileManager.default.createDirectory(
            at: fallback,
            withIntermediateDirectories: true)

        // Nothing chunk-specific: the default directory is used.
        #expect(
            ANEPrefillAttention.sidecarDirectory(
                modelDirectory: dir, configChunkTokens: 1024
            ).lastPathComponent
                == "ane_prefill")

        // An empty directory is not a sidecar; the metadata file decides.
        try FileManager.default.createDirectory(
            at: narrow,
            withIntermediateDirectories: true)
        #expect(
            ANEPrefillAttention.sidecarDirectory(
                modelDirectory: dir, configChunkTokens: 1024
            ).lastPathComponent
                == "ane_prefill")

        try JSONSerialization.data(withJSONObject: ["version": 1])
            .write(to: narrow.appendingPathComponent("ane_prefill.json"))
        #expect(
            ANEPrefillAttention.sidecarDirectory(
                modelDirectory: dir, configChunkTokens: 1024
            ).lastPathComponent
                == "ane_prefill-1024")
        // A different configured chunk still falls back to the default.
        #expect(
            ANEPrefillAttention.sidecarDirectory(
                modelDirectory: dir, configChunkTokens: 4096
            ).lastPathComponent
                == "ane_prefill")
    }

    /// The graph's shapes are fixed by its chunk, so a sidecar built for
    /// another width cannot be fed — it is refused, not approximated.
    @Test func aSidecarBuiltForAnotherChunkIsRefused() throws {
        let ctx = try MetalContext()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        let sidecar = dir.appendingPathComponent("ane_prefill")
        try FileManager.default.createDirectory(
            at: sidecar,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let meta: [String: Any] = [
            "version": 1, "family": "qwen36", "chunkTokens": 1024,
            "histories": [0], "layers": [3],
            "geometry": sidecarGeometry(chunkTokens: 1024),
            "aneCompileVerified": true,
        ]
        try JSONSerialization.data(withJSONObject: meta)
            .write(to: sidecar.appendingPathComponent("ane_prefill.json"))
        do {
            _ = try ANEPrefillAttention(
                modelDirectory: dir, device: ctx.device,
                hiddenSize: 2048, kvDim: 512, weightsSha256: nil,
                family: .qwen36, fullAttentionLayerMask: fullMask(layers: [3]),
                sparseIndexer: .none, configChunkTokens: 4096)
            Issue.record("a 1,024-token sidecar loaded under a 4,096 chunk")
        } catch {
            // Names the fix, so the operator does not have to guess the width.
            #expect("\(error)".contains("--chunk 4096"))
        }
    }

    @Test func shadowAppendSkipsPartialChunksAndStoresFullOnes() throws {
        let ctx = try MetalContext()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ane-test-\(UUID().uuidString)")
        let sidecar = dir.appendingPathComponent("ane_prefill")
        try FileManager.default.createDirectory(
            at: sidecar, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let meta: [String: Any] = [
            "version": 1, "family": "qwen36", "chunkTokens": 8,
            "histories": [0, 8], "layers": [3], "aneCompileVerified": true,
            "geometry": sidecarGeometry(
                hidden: 16, numHeads: 1, numKVHeads: 1,
                headDim: 4, chunkTokens: 8, layers: [3]),
        ]
        try JSONSerialization.data(withJSONObject: meta)
            .write(to: sidecar.appendingPathComponent("ane_prefill.json"))
        let ane = try ANEPrefillAttention(
            modelDirectory: dir,
            device: ctx.device,
            hiddenSize: 16, kvDim: 4,
            weightsSha256: nil,
            family: .qwen36,
            fullAttentionLayerMask: fullMask(layers: [3], count: 8),
            sparseIndexer: .none,
            configChunkTokens: 8)
        let kPtr = ane.stagingK.contents().bindMemory(
            to: Float16.self,
            capacity: 8 * 4)
        for index in 0..<(8 * 4) { kPtr[index] = Float16(index) }
        // Partial chunk: never appended (it is always the last chunk).
        ane.appendShadow(layer: 3, startPosition: 0, tokenCount: 4)
        ane.finishChunk(startPosition: 0, tokenCount: 4)
        #expect(ane.shadowTokens == 0)
        // Full chunk: appended and visible after finishChunk.
        ane.appendShadow(layer: 3, startPosition: 0, tokenCount: 8)
        ane.finishChunk(startPosition: 0, tokenCount: 8)
        #expect(ane.shadowTokens == 8)
    }
}
