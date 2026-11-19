import CoreML
import Foundation
import Metal
import TinyTitanCore

/// Switch for the ANE prefill attention path (Track A).
///
/// `on` routes every full-attention layer's prefill attention block through a
/// Core ML sidecar exported by `tools/export_ane_prefill.py`, leaving GDN
/// layers, the MoE, the KV cache format, and all of decode untouched. Output
/// is NOT byte-identical to the GPU path — the sidecar computes in fp16 with
/// a different reduction order (measured ~1% per-layer mean deviation against
/// an fp32 reference) — and it engages only once a prefill reaches one full
/// chunk, so shorter prompts never reach it.
///
/// **On by default** since v4.6: `environmentValue` returns `.on` when
/// `TINYTITAN_PREFILL_ANE` is unset, and a model with no sidecar falls back to the
/// GPU quietly, which is the normal case. `TINYTITAN_PREFILL_ANE=off` opts out.
/// The asymmetry is deliberate and lives in `wasRequestedExplicitly`: someone
/// who asked for `on` and has no sidecar is told, because they meant it; the
/// default must not fail a load over an optional experiment.
///
/// Anything that must be byte-reproducible has to pin this rather than follow
/// the default — `tools/golden-baseline.sh` exports `TINYTITAN_PREFILL_ANE=off` for
/// exactly that reason.
public enum RuntimePrefillANE: String, Codable, Sendable {
    case off
    case on

    /// Whether the caller named the setting, as opposed to taking the
    /// default.
    ///
    /// The two want different failure behaviour. Someone who wrote
    /// `TINYTITAN_PREFILL_ANE=on` and has no sidecar should be told so, with the
    /// export command; a default-on runtime meeting a model that has no
    /// sidecar should quietly use the GPU, because most models do not have
    /// one and failing to load would be absurd.
    public static func wasRequestedExplicitly(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        environment["TINYTITAN_PREFILL_ANE"] != nil
    }

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimePrefillANE {
        guard let raw = environment["TINYTITAN_PREFILL_ANE"] else { return .on }
        guard let value = RuntimePrefillANE(rawValue: raw) else {
            throw PrefillError.chunkedUnsupported(
                "unsupported TINYTITAN_PREFILL_ANE '\(raw)'; allowed: off, on")
        }
        return value
    }
}

/// Transfers a loaded `MLModel` out of a background load task.
///
/// unchecked-invariant: the box is created inside the loading task, handed to
/// exactly one awaiting consumer, and never mutated; the prefill loop that
/// consumes it is single-flight, so no two threads ever hold the same model.
struct LoadedModelBox: @unchecked Sendable {
    let model: MLModel
}

/// Runs the full-attention prefill block on the Neural Engine.
///
/// One multifunction `.mlpackage` per full-attention layer holds functions
/// `h0, h4096, ...` sharing a single fp16 weight set; the function name is
/// the KV-history length, which in this runtime is always the chunk-aligned
/// `startPosition`. The block consumes the post-input-norm hidden chunk and a
/// token-major fp16 K/V history, and produces the attention branch output
/// plus the chunk's cache-layout K/V — the same bytes the GPU path stages
/// before quantizing into the cache, so `copyPrefillKVToCache` is reused
/// verbatim and decode sees an ordinary cache.
///
/// Design constants mirror the exporter and are validated against its
/// manifest at load; a mismatch fails closed rather than computing nonsense.
///
/// unchecked-invariant: driven exclusively by the single-flight prefill loop
/// of one runner; buffers and lazy caches are never touched concurrently.
final class ANEPrefillAttention: @unchecked Sendable {
    /// -30000 underflows fp16 exp() exactly like -inf without putting
    /// infinity arithmetic into the ANE graph (whose fused SDPA op NaNs).
    static let maskNegative = Float16(-30000)

    /// Which mask the ANE path feeds the sidecar.
    ///
    /// `folded` is the shipped behaviour: a sparse-indexed model's key
    /// selection is folded into the additive mask, which is the only way the
    /// ANE computes the attention the model actually computes. `causal` is a
    /// **verification control, not a tuning knob** — it feeds the causal-only
    /// mask, which is *wrong* for such a model past its dense-exact window. It
    /// exists so an A/B can show the fold is load-bearing by measuring the
    /// causal arm diverge from the GPU path, rather than asserting it.
    enum MaskMode: String {
        case folded
        case causal

        static func environmentValue(
            _ environment: [String: String] = ProcessInfo.processInfo.environment
        ) throws -> MaskMode {
            guard let raw = environment["TINYTITAN_ANE_MASK"] else { return .folded }
            guard let mode = MaskMode(rawValue: raw) else {
                throw PrefillError.chunkedUnsupported(
                    "unsupported TINYTITAN_ANE_MASK '\(raw)'; allowed: folded, causal")
            }
            return mode
        }
    }

    let chunkTokens: Int
    let histories: Set<Int>
    let coveredLayers: Set<Int>
    let maxPromptTokens: Int
    /// True when this model's full-attention layers select keys with a QSA
    /// indexer, so every chunk past the dense-exact window must be fed a mask
    /// with that selection folded in rather than the causal mask alone.
    let requiresSelection: Bool
    /// The visible-key count below which dense attention *is* the model's
    /// selection, from the model's own indexer geometry. A chunk that needs a
    /// selection and does not get one is refused rather than attended densely.
    let exactVisibleKeys: Int?
    /// Which mask the sidecar is fed; `.causal` only as a verification control.
    let maskMode: MaskMode

    /// Shared-mode staging: the GPU blits `normed` in, Core ML writes the
    /// three outputs back via output backings, the GPU quantizes K/V into the
    /// cache from the same memory, and the shadow append memcpys from it.
    let stagingNormed: MTLBuffer
    let stagingOut: MTLBuffer
    let stagingGate: MTLBuffer
    let stagingK: MTLBuffer
    let stagingV: MTLBuffer

    let hiddenSize: Int
    let kvDim: Int
    /// `numHeads * headDim`: the width of the attention core's output when the
    /// graph's tail is split (see `tailSplit`).
    let qDim: Int
    /// The sidecar returns `attn`/`gate` and the runtime applies gate + wo on
    /// the GPU; false for a historical fused sidecar that returns `out`.
    let tailSplit: Bool
    let packageDir: URL
    let compiledDir: URL
    /// At most one loaded model at a time. Each loaded function pins an
    /// E5RT/ANE inference arena (the h4096 variant's score tensors alone are
    /// ~1 GB); keeping 20 of them resident during a long prefill pressured
    /// the 8 GiB expert slot cache out of RAM and collapsed the decode that
    /// followed to ~2 tok/s. One-at-a-time bounds the ANE footprint to a
    /// single context at ~0.5 s reload cost per layer-chunk.
    var residentModel: (layer: Int, history: Int, model: MLModel)?
    /// In-flight load of the *next* layer's model, started as soon as this
    /// layer's prediction returns so the ~0.5 s load overlaps the GPU's MoE
    /// stage instead of serializing in front of the next prediction. At most
    /// one is outstanding, which keeps the one-resident-arena rule intact:
    /// the preloaded model only becomes resident when `model(layer:history:)`
    /// adopts it, and that is the same moment the previous one is dropped.
    var preloaded: (layer: Int, history: Int, task: Task<LoadedModelBox, Error>)?
    var masks: [Int: MLMultiArray] = [:]
    var maskStorage: [Int: UnsafeMutableRawPointer] = [:]
    /// The folded masks, one per history window, beside the causal ones: for a
    /// sparse-indexed model the mask depends on the *layer* as well, so the
    /// buffer is rewritten per covered layer. One chunk of one layer is live
    /// at a time in the chunk loop, so this is bounded by the same 33–134 MB
    /// per window the causal masks cost, not by layer count.
    var selectionMasks: [Int: MLMultiArray] = [:]
    var selectionMaskStorage: [Int: UnsafeMutableRawPointer] = [:]
    /// One all-`-30000` row per history window, the fold's reset.
    var selectionNegativeRow: [Int: UnsafeMutableRawPointer] = [:]
    /// Token-major fp16 K/V rows per layer, at absolute prompt positions, so
    /// later chunks can attend to exact-precision history without
    /// re-dequantizing the cache. Allocated on the first append (single-chunk
    /// prompts never pay for it) and reused across requests.
    var shadowK: [Int: UnsafeMutableRawPointer] = [:]
    var shadowV: [Int: UnsafeMutableRawPointer] = [:]
    var shadowTokens = 0
    var loggedFallback = false
    var loggedCausalMask = false

    /// - Parameter weightsSha256: the model's own recorded `model_weights.bin`
    ///   digest, taken from its install receipt. A sidecar exported from
    ///   different weights computes plausible-looking but wrong attention, and
    ///   nothing downstream would catch it — so the binding is checked here
    ///   and fails closed. Nil skips the check (no receipt available) and says
    ///   so, rather than silently trusting.
    /// - Parameters family, fullAttentionLayerMask: the model the sidecar is
    ///   being loaded for. The sidecar's recorded geometry must match it; a
    ///   mismatch is refused rather than run, because a graph built for another
    ///   width or head count computes a different attention and says nothing.
    /// - Parameter configChunkTokens: the runtime's configured prefill chunk.
    ///   It selects *which* sidecar directory is loaded, and the one that is
    ///   found must be built for exactly this chunk — the gate below is a
    ///   contract with the graph's fixed shapes, so a nearer one is not usable.
    /// - Parameter maskMode: `.folded` everywhere except a verification run;
    ///   nil reads `TINYTITAN_ANE_MASK` (the tests pass it explicitly).
    init(
        modelDirectory: URL, device: MTLDevice,
        hiddenSize: Int, kvDim: Int, weightsSha256: String?,
        family: ModelFamily, fullAttentionLayerMask: [UInt8],
        sparseIndexer: SparseIndexerConfig,
        configChunkTokens: Int,
        maskMode: MaskMode? = nil
    ) throws {
        self.maskMode = try maskMode ?? MaskMode.environmentValue()
        let dir = Self.sidecarDirectory(
            modelDirectory: modelDirectory,
            configChunkTokens: configChunkTokens)
        let metaURL = dir.appendingPathComponent("ane_prefill.json")
        guard FileManager.default.fileExists(atPath: metaURL.path) else {
            throw PrefillError.chunkedUnsupported(
                "TINYTITAN_PREFILL_ANE=on but \(metaURL.path) is missing; run "
                    + "tools/export_ane_prefill.py --model \(modelDirectory.path) "
                    + "--chunk \(configChunkTokens) for this model first")
        }
        let meta = try Self.loadSidecarMetadata(at: metaURL)
        guard meta.version == Self.expectedVersion else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar version \(meta.version) != supported \(Self.expectedVersion); re-export"
            )
        }
        // The graph's shapes are fixed by its chunk, so only a sidecar built
        // for the configured chunk can be fed. This is also what makes a
        // multi-width model safe: the chunk-specific directory is preferred,
        // and a fallback to the default one is refused here when it does not
        // match.
        guard meta.chunkTokens == configChunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar in \(dir.lastPathComponent) is built for a "
                    + "\(meta.chunkTokens)-token chunk but the runtime's prefill "
                    + "chunk is \(configChunkTokens); export one for this width "
                    + "(tools/export_ane_prefill.py --chunk \(configChunkTokens)) "
                    + "or configure --prefill-chunk \(meta.chunkTokens)")
        }
        // A sparse-indexed model is served by folding its indexer's selection
        // into the mask the sidecar is fed (`fillSelectionMask`): the graph's
        // mask input is an *arbitrary* additive mask, so no part of the graph
        // changes and the same sidecar works either way. What has to be true is
        // that the runtime actually folds — dense attention matches the
        // selection only through `keptBlocks * compressRatio +
        // (compressRatio - 1)` visible keys (2,051 for the shipped Qwen 3.8
        // geometry), and the smallest chunk the ANE accepts is a full 4,096,
        // already past it. A sidecar that does not record the contract is
        // refused rather than trusted, because a causal-only mask attends to
        // keys the model drops, silently and with plausible output.
        let exactVisibleKeys =
            sparseIndexer.enabled
            ? QSAExactness(sparseIndexer).maximumExactVisibleKeys : nil
        if sparseIndexer.enabled {
            guard meta.selectionFolded == true else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill sidecar does not record a folded sparse "
                        + "selection (selectionFolded is missing or false); with a "
                        + "causal-only mask the sidecar would attend to keys this "
                        + "model's indexer drops past \(exactVisibleKeys ?? 0) visible "
                        + "keys. Re-export with tools/export_ane_prefill.py.")
            }
        }
        // One geometry per sidecar. The graph's weights, head split, rope and
        // GQA expansion are all built from these numbers, so a sidecar that
        // disagrees with the model is not "close enough": it computes a
        // different attention. Refusing sends the runner to the GPU path.
        guard let geometry = meta.geometry else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar records no geometry (it predates the "
                    + "generalized exporter); re-export it for this model")
        }
        guard geometry.family == family.rawValue,
            meta.family == family.rawValue
        else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar is for family '\(geometry.family)' but this "
                    + "model is '\(family.rawValue)'; re-export it for this model")
        }
        guard geometry.hiddenSize == hiddenSize,
            geometry.numKVHeads * geometry.headDim == kvDim,
            geometry.chunkTokens == meta.chunkTokens
        else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar geometry (hidden \(geometry.hiddenSize), "
                    + "\(geometry.numKVHeads)x\(geometry.headDim) kv, chunk "
                    + "\(geometry.chunkTokens)) does not match this model (hidden "
                    + "\(hiddenSize), kvDim \(kvDim), chunk \(meta.chunkTokens))")
        }
        for layer in meta.layers {
            guard layer >= 0, layer < fullAttentionLayerMask.count,
                fullAttentionLayerMask[layer] == 1
            else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill sidecar covers layer \(layer), which is not a "
                        + "full-attention layer of this model")
            }
        }
        // Issue #7: a sidecar whose variants the ANE refused to compile loads
        // fine and then runs the whole prefill on the CPU, ~38x slower than the
        // GPU path. Only an exporter that verified the compilation sets this.
        // Refusing here means the GPU path is used instead (or, with
        // TINYTITAN_PREFILL_ANE=on, the load fails with the reason).
        guard meta.aneCompileVerified == true else {
            throw PrefillError.chunkedUnsupported(
                "ANE prefill sidecar does not record a verified ANE compilation "
                    + "(aneCompileVerified is missing or false); the Neural Engine "
                    + "may have refused it and prefill would run on the CPU at ~38x "
                    + "the GPU cost. Re-export with tools/export_ane_prefill.py.")
        }
        if let weightsSha256, let exported = meta.weightsSha256 {
            guard exported.lowercased() == weightsSha256.lowercased() else {
                throw PrefillError.chunkedUnsupported(
                    "ANE prefill sidecar was exported from different weights "
                        + "(sidecar \(exported.prefix(12))..., model "
                        + "\(weightsSha256.prefix(12))...); re-export it for this model")
            }
        } else {
            // stderr for the same reason as the fallback notice below:
            // stdout is the generated text.
            FileHandle.standardError.write(
                Data(
                    ("TinyTitan ane-prefill: sidecar/weights binding unverified "
                        + "(no receipt digest available); a stale sidecar would not "
                        + "be detected\n").utf8))
        }
        self.chunkTokens = meta.chunkTokens
        self.histories = Set(meta.histories)
        self.coveredLayers = Set(meta.layers)
        self.maxPromptTokens = (meta.histories.max() ?? 0) + meta.chunkTokens
        self.requiresSelection = sparseIndexer.enabled
        self.exactVisibleKeys = exactVisibleKeys
        self.hiddenSize = hiddenSize
        self.kvDim = kvDim
        self.qDim = meta.geometry.map { $0.numHeads * $0.headDim } ?? hiddenSize
        self.tailSplit = meta.tail == "split"
        self.packageDir = dir
        self.compiledDir = dir.appendingPathComponent("compiled-v\(meta.version)")
        try FileManager.default.createDirectory(
            at: compiledDir, withIntermediateDirectories: true)

        let halfBytes = MemoryLayout<Float16>.stride
        func staging(_ elements: Int, _ label: String) throws -> MTLBuffer {
            guard
                let made = device.makeBuffer(
                    length: elements * halfBytes,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = label
            return made
        }
        self.stagingNormed = try staging(
            meta.chunkTokens * hiddenSize,
            "ane.staging.normed")
        self.stagingOut = try staging(
            meta.chunkTokens * max(hiddenSize, self.qDim),
            "ane.staging.out")
        self.stagingGate = try staging(
            meta.chunkTokens * self.qDim,
            "ane.staging.gate")
        self.stagingK = try staging(meta.chunkTokens * kvDim, "ane.staging.k")
        self.stagingV = try staging(meta.chunkTokens * kvDim, "ane.staging.v")
    }

    deinit {
        for pointer in maskStorage.values { pointer.deallocate() }
        for pointer in selectionMaskStorage.values { pointer.deallocate() }
        for pointer in selectionNegativeRow.values { pointer.deallocate() }
        for pointer in shadowK.values { pointer.deallocate() }
        for pointer in shadowV.values { pointer.deallocate() }
    }

    /// Whether this chunk can run on the ANE. Continuity matters: a chunk at
    /// a nonzero start needs the shadow rows of every earlier chunk, so a
    /// resumed or partially GPU-processed prefill falls back for the rest of
    /// the request instead of attending to a hole.
    func eligibleChunk(
        startPosition: Int, tokenCount: Int,
        configChunkTokens: Int
    ) -> Bool {
        // A short prompt is one partial chunk; padding it to 4,096 costs
        // ~2 s of ANE work against under a second on the GPU, so the ANE
        // serves only full chunks and the continuation chunks of long
        // prompts — the workload it wins by 26x.
        let fullOrContinuation = tokenCount == chunkTokens || startPosition > 0
        guard configChunkTokens == chunkTokens,
            fullOrContinuation,
            tokenCount <= chunkTokens,
            startPosition % chunkTokens == 0,
            histories.contains(startPosition)
        else {
            if !loggedFallback {
                loggedFallback = true
                // stderr, not stdout: stdout carries generated tokens, and a
                // diagnostic written there lands in the middle of the model's
                // output. Harmless while ANE prefill was opt-in and this
                // never fired; corrupting once it became the default, which
                // is how the golden baselines caught it.
                FileHandle.standardError.write(
                    Data(
                        ("TinyTitan ane-prefill fallback: chunk at \(startPosition) "
                            + "(+\(tokenCount)) outside sidecar coverage "
                            + "(chunk \(chunkTokens), max prompt \(maxPromptTokens)); "
                            + "using the GPU path\n").utf8))
            }
            return false
        }
        if startPosition == 0 {
            shadowTokens = 0
            return true
        }
        return shadowTokens == startPosition
    }

    /// Saves the chunk's K/V rows for later chunks. Partial chunks are always
    /// the last chunk of a prompt, so their rows can never be history and the
    /// shadow (33 MB per layer) is never allocated for single-chunk prompts.
    func appendShadow(layer: Int, startPosition: Int, tokenCount: Int) {
        guard tokenCount == chunkTokens else { return }
        let rowBytes = kvDim * MemoryLayout<Float16>.stride
        let capacityBytes = maxPromptTokens * rowBytes
        if shadowK[layer] == nil {
            shadowK[layer] = .allocate(byteCount: capacityBytes, alignment: 16_384)
            shadowV[layer] = .allocate(byteCount: capacityBytes, alignment: 16_384)
        }
        let offset = startPosition * rowBytes
        let length = tokenCount * rowBytes
        guard let shadowKBuffer = shadowK[layer], let shadowVBuffer = shadowV[layer] else {
            return
        }
        memcpy(shadowKBuffer.advanced(by: offset), stagingK.contents(), length)
        memcpy(shadowVBuffer.advanced(by: offset), stagingV.contents(), length)
    }

    /// Marks the chunk's shadow rows visible to the next chunk. Called once
    /// after every covered layer appended, so a thrown mid-chunk error leaves
    /// `shadowTokens` behind `startPosition` and the next attempt falls back
    /// to the GPU instead of attending to partial history.
    func finishChunk(startPosition: Int, tokenCount: Int) {
        shadowTokens =
            tokenCount == chunkTokens
            ? startPosition + tokenCount : 0
        if tokenCount < chunkTokens {
            // A partial chunk is the prompt's last: decode is next.
            releaseModels()
        }
    }
}
