import Foundation
import Metal
import CoreML
import TinyTitanCore

// The ANE prefill sidecar's model lifecycle and per-layer masks: residency,
// compilation, loading, release, and the causal/selection masks the graph reads.
//
// Split out of `ANEPrefillAttention.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
extension ANEPrefillAttention {

    func model(layer: Int, history: Int) async throws -> MLModel {
        if let cached = residentModel,
            cached.layer == layer, cached.history == history
        {
            return cached.model
        }
        if let pending = preloaded, pending.layer == layer,
            pending.history == history
        {
            preloaded = nil
            // Drop the old arena only once the new model is in hand, then
            // adopt it — never two resident at once for longer than the
            // handover itself.
            let loaded = try await pending.task.value.model
            residentModel = (layer, history, loaded)
            traceResident(layer: layer, history: history)
            return loaded
        }
        // A preload for a different layer is now useless; await and discard it
        // rather than leaking an arena behind the resident one.
        if let stale = preloaded {
            preloaded = nil
            _ = try? await stale.task.value
        }
        residentModel = nil
        let compiled = try await compiledModelURL(layer: layer)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        configuration.functionName = "h\(history)"
        let loaded = try MLModel(
            contentsOf: compiled,
            configuration: configuration)
        residentModel = (layer, history, loaded)
        traceResident(layer: layer, history: history)
        return loaded
    }

    /// `TINYTITAN_ANE_MEMORY_TRACE=1`: the footprint with one E5RT arena
    /// resident, so it can be compared with the decode-start line and with a
    /// GPU-prefilled run — the arena's size is that difference.
    func traceResident(layer: Int, history: Int) {
        guard ProcessInfo.processInfo.environment["TINYTITAN_ANE_MEMORY_TRACE"] == "1" else {
            return
        }
        FileHandle.standardError.write(
            Data(
                String(
                    format:
                        "[ane-mem] resident layer=%d history=%d footprint=%.1f MiB\n",
                    layer, history, ProcessMemory.physFootprintMiB()
                ).utf8))
    }

    /// The on-disk compiled model for `layer`, compiling it from the package
    /// on first use and whenever the package is newer.
    func compiledModelURL(layer: Int) async throws -> URL {
        let package = packageDir.appendingPathComponent("layer_\(layer).mlpackage")
        let compiled = compiledDir.appendingPathComponent("layer_\(layer).mlmodelc")
        let fm = FileManager.default
        func modifiedDate(_ url: URL) -> Date {
            (try? fm.attributesOfItem(atPath: url.path)[.modificationDate]
                as? Date) ?? .distantPast
        }
        if !fm.fileExists(atPath: compiled.path)
            || modifiedDate(compiled) < modifiedDate(package)
        {
            guard fm.fileExists(atPath: package.path) else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill sidecar is missing \(package.lastPathComponent)")
            }
            let temporary = try await MLModel.compileModel(at: package)
            _ = try? fm.removeItem(at: compiled)
            try fm.moveItem(at: temporary, to: compiled)
        }
        return compiled
    }

    /// The compile-and-load half of `model(layer:history:)`, without touching
    /// `residentModel` — safe to run detached for a preload.
    func loadModel(layer: Int, history: Int) async throws -> MLModel {
        let compiled = try await compiledModelURL(layer: layer)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        configuration.functionName = "h\(history)"
        return try MLModel(contentsOf: compiled, configuration: configuration)
    }

    /// Drops everything prefill allocated. Called when prefill hands over to
    /// decode so nothing ANE-side competes with the expert cache for
    /// residency; cheap no-op when nothing is loaded.
    ///
    /// The Core ML model is not the only thing that has to go. Decode runs
    /// entirely on the GPU and never reads the shadow rows or the masks, but
    /// they stay allocated for the life of this object unless dropped here:
    /// ~33 MB of shadow per covered layer plus one mask per history window
    /// (33 MB at h0, 100 MB at h8192) is roughly half a gigabyte taken from
    /// the expert slot cache for the whole of decode. Releasing only the
    /// model left that behind and cost measured decode throughput -- the
    /// bounded cache is the entire reason a model larger than RAM runs at
    /// all, so prefill scratch must not outlive prefill.
    ///
    /// Masks are rebuilt on the next prefill (a few tens of ms of fills
    /// against a prefill measured in minutes); shadow rows are per-request by
    /// construction, so `shadowTokens` resets with them and a later chunk
    /// correctly falls back rather than attending to a freed history.
    func releaseModels() {
        // `TINYTITAN_ANE_MEMORY_TRACE=1`: what dropping the Core ML model does to
        // the process footprint, which is the TT-004 question — an E5RT arena
        // that is *not* returned keeps costing residency and bandwidth through
        // decode. Measured before the drop, after it, and after our own scratch
        // buffers are freed, so the model's share is separable from the masks'.
        let trace = ProcessInfo.processInfo.environment["TINYTITAN_ANE_MEMORY_TRACE"] == "1"
        let before = trace ? ProcessMemory.physFootprintMiB() : 0
        residentModel = nil
        preloaded?.task.cancel()
        preloaded = nil
        let afterModel = trace ? ProcessMemory.physFootprintMiB() : 0
        // Release the borrowing MLMultiArrays before the storage they point
        // at: they are built with `deallocator: nil`, so this dictionary owns
        // the memory.
        masks.removeAll()
        for pointer in maskStorage.values { pointer.deallocate() }
        maskStorage.removeAll()
        selectionMasks.removeAll()
        for pointer in selectionMaskStorage.values { pointer.deallocate() }
        selectionMaskStorage.removeAll()
        for pointer in selectionNegativeRow.values { pointer.deallocate() }
        selectionNegativeRow.removeAll()
        for pointer in shadowK.values { pointer.deallocate() }
        for pointer in shadowV.values { pointer.deallocate() }
        shadowK.removeAll()
        shadowV.removeAll()
        shadowTokens = 0
        if trace {
            FileHandle.standardError.write(
                Data(
                    String(
                        format:
                            "[ane-mem] release before=%.1f afterModelDrop=%.1f afterScratchFree=%.1f MiB "
                            + "(model held %.1f, scratch %.1f)\n",
                        before, afterModel, ProcessMemory.physFootprintMiB(),
                        before - afterModel, afterModel - ProcessMemory.physFootprintMiB()
                    ).utf8))
        }
    }

    /// Starts loading `layer`'s model for `history` in the background, if it
    /// is not already resident or in flight. Called right after a prediction
    /// returns, so the load runs while the caller encodes and executes the
    /// layer's MoE stage on the GPU.
    func preload(layer: Int, history: Int) {
        if let cached = residentModel,
            cached.layer == layer, cached.history == history
        {
            return
        }
        if let pending = preloaded,
            pending.layer == layer, pending.history == history
        {
            return
        }
        preloaded?.task.cancel()
        preloaded = (
            layer, history,
            Task { [self] in
                LoadedModelBox(
                    model: try await loadModel(
                        layer: layer,
                        history: history))
            }
        )
    }

    func mask(history: Int) throws -> MLMultiArray {
        if let cached = masks[history] { return cached }
        let total = history + chunkTokens
        let count = chunkTokens * total
        let storage = UnsafeMutableRawPointer.allocate(
            byteCount: count * MemoryLayout<Float16>.stride,
            alignment: 16_384)
        let values = storage.bindMemory(to: Float16.self, capacity: count)
        for row in 0..<chunkTokens {
            let base = row * total
            let allowed = history + row + 1
            for column in 0..<allowed { values[base + column] = 0 }
            for column in allowed..<total {
                values[base + column] = Self.maskNegative
            }
        }
        let array = try MLMultiArray(
            dataPointer: storage,
            shape: [1, 1, NSNumber(value: chunkTokens), NSNumber(value: total)],
            dataType: .float16,
            strides: [
                NSNumber(value: count), NSNumber(value: count),
                NSNumber(value: total), 1,
            ],
            deallocator: nil)
        maskStorage[history] = storage
        masks[history] = array
        return array
    }

    /// The additive mask for one chunk of one layer: `-30000` on every key the
    /// query must not read, `0` on every key it may.
    ///
    /// For a sparse-indexed model the causal mask alone is not the attention
    /// this model computes: past the dense-exact window the QSA indexer drops
    /// keys, and the GPU path reads exactly the kept ones. The sidecar's mask
    /// input is *arbitrary*, so folding the selection in here makes the ANE's
    /// softmax the GPU's gather — `exp(-30000)` underflows to zero in fp16
    /// exactly as an omitted key contributes nothing — and the graph does not
    /// change at all.
    ///
    /// The buffer is per history window and rewritten for every prediction:
    /// the selection is per layer *and* per prompt, so a cached fill cannot be
    /// reused across layers or requests. Refilling is one memcpy and at most
    /// `selectionWidth` stores per row — tens of milliseconds against a
    /// prefill measured in minutes.
    ///
    /// Rows past `tokenCount` are the chunk's padding (zero queries), which an
    /// all-masked row averages uniformly — finite, and discarded by `predict`.
    ///
    /// Internal rather than private: `ANEPrefillAttentionTests` checks the fold
    /// against a hand-built selection, and the fold is exactly the arithmetic a
    /// wrong mask would silently get wrong.
    func selectionMask(
        history: Int, tokenCount: Int,
        selection: QSASelection
    ) throws -> MLMultiArray {
        let halfBytes = MemoryLayout<Float16>.stride
        let total = history + chunkTokens
        let count = chunkTokens * total
        let array: MLMultiArray
        if let cached = selectionMasks[history] {
            array = cached
        } else {
            let storage = UnsafeMutableRawPointer.allocate(
                byteCount: count * halfBytes, alignment: 16_384)
            let negatives = UnsafeMutableRawPointer.allocate(
                byteCount: total * halfBytes, alignment: 16_384)
            let negativeValues = negatives.bindMemory(
                to: Float16.self,
                capacity: total)
            for column in 0..<total { negativeValues[column] = Self.maskNegative }
            array = try MLMultiArray(
                dataPointer: storage,
                shape: [1, 1, NSNumber(value: chunkTokens), NSNumber(value: total)],
                dataType: .float16,
                strides: [
                    NSNumber(value: count), NSNumber(value: count),
                    NSNumber(value: total), 1,
                ],
                deallocator: nil)
            selectionMaskStorage[history] = storage
            selectionNegativeRow[history] = negatives
            selectionMasks[history] = array
        }
        guard let storage = selectionMaskStorage[history],
            let negativeRow = selectionNegativeRow[history]?.bindMemory(
                to: Float16.self, capacity: total)
        else {
            throw ModelError.internalInconsistency(
                detail: "the ANE selection mask for history \(history) was not cached")
        }
        let values = storage.bindMemory(to: Float16.self, capacity: count)
        let indices = selection.indices.contents().bindMemory(
            to: UInt32.self, capacity: max(1, tokenCount * selection.indexStride))
        let counts = selection.counts.contents().bindMemory(
            to: UInt32.self, capacity: max(1, tokenCount))
        for row in 0..<chunkTokens {
            let base = row * total
            memcpy(
                UnsafeMutableRawPointer(values + base),
                UnsafeRawPointer(negativeRow), total * halfBytes)
            guard row < tokenCount else { continue }
            // The compacted ascending selection the GPU's attention gathers:
            // the same keys, so the same softmax.
            let written = min(Int(counts[row]), selection.indexStride)
            let rowIndices = indices + row * selection.indexStride
            for slot in 0..<written {
                let key = Int(rowIndices[slot])
                guard key < total else { continue }
                values[base + key] = 0
            }
        }
        return array
    }

    func wrap(
        _ buffer: MTLBuffer, rows: Int,
        columns: Int
    ) throws -> MLMultiArray {
        try MLMultiArray(
            dataPointer: buffer.contents(),
            shape: [NSNumber(value: rows), NSNumber(value: columns)],
            dataType: .float16,
            strides: [NSNumber(value: columns), 1],
            deallocator: nil)
    }

    func wrapShadow(
        _ storage: UnsafeMutableRawPointer,
        rows: Int
    ) throws -> MLMultiArray {
        try MLMultiArray(
            dataPointer: storage,
            shape: [NSNumber(value: rows), NSNumber(value: kvDim)],
            dataType: .float16,
            strides: [NSNumber(value: kvDim), 1],
            deallocator: nil)
    }

    /// Runs one layer's attention block. `stagingNormed` must already hold
    /// the chunk's post-norm hidden rows; results land in `stagingOut` /
    /// `stagingK` / `stagingV` (real `tokenCount` rows; padding discarded).
    ///
    /// - Parameter selection: this layer's QSA key selection for this chunk, or
    ///   nil where the model has no indexer or every visible key is kept. A
    ///   sparse-indexed model past its dense-exact window must supply one: the
    ///   causal mask would otherwise attend to keys the model drops, so a
    ///   missing selection there is refused rather than run.
    func predict(
        layer: Int, history: Int, tokenCount: Int,
        selection: QSASelection?
    ) async throws {
        let halfBytes = MemoryLayout<Float16>.stride
        if tokenCount < chunkTokens {
            // Padded rows must be zeros: zero queries attend uniformly and
            // produce finite garbage that is discarded, whereas stale staging
            // bytes could push fp16 out of range.
            let start = tokenCount * hiddenSize * halfBytes
            let length = (chunkTokens - tokenCount) * hiddenSize * halfBytes
            memset(stagingNormed.contents().advanced(by: start), 0, length)
        }
        let maskFeature: MLMultiArray
        if let selection, maskMode == .folded {
            maskFeature = try selectionMask(
                history: history,
                tokenCount: tokenCount,
                selection: selection)
        } else {
            if requiresSelection, maskMode == .causal, let exact = exactVisibleKeys,
                history + tokenCount > exact, !loggedCausalMask
            {
                loggedCausalMask = true
                // stderr for the same reason as the fallback notice: stdout is
                // the generated text.
                FileHandle.standardError.write(
                    Data(
                        ("TinyTitan ane-prefill: TINYTITAN_ANE_MASK=causal feeds the "
                            + "causal-only mask, which is WRONG for this sparse-indexed "
                            + "model past \(exact) visible keys; verification control "
                            + "only\n").utf8))
            }
            // No selection is only correct while every visible key is kept.
            // Past that the GPU path gathers the indexer's choice and the ANE
            // has to be fed the same one; a missing selection there is a caller
            // bug, not permission to attend densely.
            if requiresSelection, maskMode == .folded, let exact = exactVisibleKeys,
                history + tokenCount > exact
            {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill has no QSA selection for the chunk at "
                        + "\(history)+\(tokenCount) tokens, where this model's "
                        + "indexer drops keys past \(exact) visible ones; refusing to "
                        + "attend densely")
            }
            maskFeature = try mask(history: history)
        }
        var features: [String: MLMultiArray] = [
            "normed": try wrap(
                stagingNormed, rows: chunkTokens,
                columns: hiddenSize),
            "mask": maskFeature,
        ]
        if history > 0 {
            guard let kShadow = shadowK[layer], let vShadow = shadowV[layer] else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill shadow missing for layer \(layer) at history \(history)")
            }
            features["k_hist"] = try wrapShadow(kShadow, rows: history)
            features["v_hist"] = try wrapShadow(vShadow, rows: history)
        }
        let provider = try MLDictionaryFeatureProvider(
            dictionary: features.mapValues { MLFeatureValue(multiArray: $0) })
        let options = MLPredictionOptions()
        options.outputBackings = [
            "out": try wrap(stagingOut, rows: chunkTokens, columns: hiddenSize),
            "k_new": try wrap(stagingK, rows: chunkTokens, columns: kvDim),
            "v_new": try wrap(stagingV, rows: chunkTokens, columns: kvDim),
        ]
        let model = try await model(layer: layer, history: history)
        let result = try await model.prediction(from: provider, options: options)
        // Output backings are best-effort; copy back any output Core ML chose
        // to allocate elsewhere.
        try copyIfNotBacked(
            result, name: "out", buffer: stagingOut,
            elements: chunkTokens * hiddenSize)
        try copyIfNotBacked(
            result, name: "k_new", buffer: stagingK,
            elements: chunkTokens * kvDim)
        try copyIfNotBacked(
            result, name: "v_new", buffer: stagingV,
            elements: chunkTokens * kvDim)
    }

    func copyIfNotBacked(
        _ result: MLFeatureProvider, name: String,
        buffer: MTLBuffer, elements: Int
    ) throws {
        guard let array = result.featureValue(for: name)?.multiArrayValue else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill output '\(name)' missing from prediction")
        }
        array.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress,
                base != buffer.contents()
            else { return }
            memcpy(
                buffer.contents(), base,
                elements * MemoryLayout<Float16>.stride)
        }
    }
}
