import Foundation
import Metal
import Darwin

public struct ModelLoadStats: Sendable {
    public var manifestSha256Nanos: UInt64
    public var receiptValidationNanos: UInt64
    public var eagerSha256Nanos: UInt64

    public init(manifestSha256Nanos: UInt64 = 0,
                receiptValidationNanos: UInt64 = 0,
                eagerSha256Nanos: UInt64 = 0) {
        self.manifestSha256Nanos = manifestSha256Nanos
        self.receiptValidationNanos = receiptValidationNanos
        self.eagerSha256Nanos = eagerSha256Nanos
    }
}

/// Bounded routed-expert cache configuration.
public enum ExpertStreamingMode: Sendable {
    /// Read each expert into one of `slotCount` 2 MB-aligned cache slots.
    case pread(slotCount: Int)
}

/// Loaded `.gturbo/` model. Resident weights live behind one mmap'd
/// `MTLBuffer`; routed expert weights live behind per-layer streaming
/// backends opened lazily on first touch.
public struct Model {
    public let device: MTLDevice
    public let config: ArchConfig
    public let streamingMode: ExpertStreamingMode
    public let expertCachePolicy: ExpertCachePolicy
    public let integrityPolicy: ModelIntegrityPolicy

    let residentBuffer: ResidentBuffer
    let residentIndex: ResidentIndex
    let packedExpertsLayout: PackedExpertsLayout
    let manifest: Manifest
    let directoryURL: URL

    /// Lazy state. Held inside a reference box so `Model` can stay a struct
    /// while still letting accessors mutate layer state via a serial queue.
    let streamersBox: StreamersBox
    let streamersQueue: DispatchQueue

    final class StreamersBox: @unchecked Sendable {
        var streamers: [PreadExpertStreamer?]
        var layerVerified: [Bool]
        init(numLayers: Int) {
            self.streamers = Array(repeating: nil, count: numLayers)
            self.layerVerified = Array(repeating: false, count: numLayers)
        }
    }

    init(device: MTLDevice,
         config: ArchConfig,
         streamingMode: ExpertStreamingMode,
         expertCachePolicy: ExpertCachePolicy,
         integrityPolicy: ModelIntegrityPolicy,
         residentBuffer: ResidentBuffer,
         residentIndex: ResidentIndex,
         packedExpertsLayout: PackedExpertsLayout,
         manifest: Manifest,
         directoryURL: URL) {
        self.device = device
        self.config = config
        self.streamingMode = streamingMode
        self.expertCachePolicy = expertCachePolicy
        self.integrityPolicy = integrityPolicy
        self.residentBuffer = residentBuffer
        self.residentIndex = residentIndex
        self.packedExpertsLayout = packedExpertsLayout
        self.manifest = manifest
        self.directoryURL = directoryURL
        self.streamersBox = StreamersBox(numLayers: packedExpertsLayout.numLayers)
        self.streamersQueue = DispatchQueue(label: "turbo-fieldfare.expert-streamers")
    }

    // MARK: - Resident accessors

    public var embedding: TensorView {
        try! resident(name: "language_model.model.embed_tokens.weight")
    }

    /// Gemma 4 ties lm_head to the embedding. The transpose for the lm_head
    /// GEMV path is the kernel's job, not the loader's.
    public var lmHead: TensorView { embedding }

    public func qProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.q_proj.weight")
    }
    public func kProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.k_proj.weight")
    }
    public func vProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.v_proj.weight")
    }
    public func oProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.o_proj.weight")
    }
    /// Writer emits `.router.proj.weight` (no `.mlp.` segment).
    public func router(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.proj.weight")
    }
    /// Writer emits the shared-expert FFN as `.mlp.{gate,up,down}_proj.weight`
    /// without a `.shared_expert.` segment.
    public func sharedExpertGate(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.gate_proj.weight")
    }
    public func sharedExpertUp(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.up_proj.weight")
    }
    public func sharedExpertDown(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.down_proj.weight")
    }
    public func inputNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).input_layernorm.weight")
    }
    public func postAttnNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_attention_layernorm.weight")
    }
    public var finalNorm: TensorView {
        try! resident(name: "language_model.model.norm.weight")
    }

    // MARK: - Per-head attention norms (Q/K only)
    //
    // `q_norm` and `k_norm` are RMSNorm with learnable scale, applied per head
    // before RoPE. `v_norm` has **no learnable weight** (no-scale RMSNorm) and
    // is therefore not stored as a tensor — the runtime uses an
    // explicit no-scale variant rather than consuming a unit-weight buffer.

    public func qNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.q_norm.weight")
    }
    public func kNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.k_norm.weight")
    }

    // MARK: - Feed-forward norms
    //
    // The Gemma 4 sandwich wraps two parallel FFN branches:
    //   pre_feedforward_layernorm        -> dense MLP input
    //   pre_feedforward_layernorm_2      -> routed expert input
    //   post_feedforward_layernorm_1     -> dense MLP output
    //   post_feedforward_layernorm_2     -> routed expert output
    //   post_feedforward_layernorm       -> combined (h1+h2) output

    public func preFFN(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).pre_feedforward_layernorm.weight")
    }
    public func preFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).pre_feedforward_layernorm_2.weight")
    }
    public func postFFN1(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm_1.weight")
    }
    public func postFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm_2.weight")
    }
    public func postFFN(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm.weight")
    }

    // MARK: - Router auxiliaries
    //
    // `router.scale` is a per-feature multiplier on the router's input
    // (post-RMSNorm), fused with 1/sqrt(hidden_size). `per_expert_scale` is
    // applied to the top-k routing weights after softmax over top-k.

    public func routerScale(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.scale")
    }
    public func routerPerExpertScale(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.per_expert_scale")
    }

    /// Per-layer scalar gain applied to the entire residual stream at the end
    /// of the layer; shape `[1]`, BF16.
    public func layerScalar(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).layer_scalar")
    }

    /// Resolve a tensor name to a `TensorView` against the resident buffer.
    /// `fileOffset` (absolute) is converted to a buffer-relative offset by
    /// subtracting the resident region's file offset (which equals
    /// `header.indexSize`).
    func resident(name: String) throws -> TensorView {
        guard let entry = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        let residentFileOffset = residentIndex.header.indexSize
        let relativeOffset = entry.fileOffset - residentFileOffset
        let scaleRel: UInt64 = entry.scaleSize > 0
            ? entry.scaleOffset - residentFileOffset : 0
        let biasRel: UInt64 = entry.biasSize > 0
            ? entry.biasOffset - residentFileOffset : 0
        return TensorView(
            buffer: residentBuffer.buffer,
            offset: relativeOffset,
            length: entry.sizeBytes,
            scaleOffset: scaleRel, scaleLength: entry.scaleSize,
            biasOffset:  biasRel,  biasLength:  entry.biasSize,
            shape: entry.shape,
            dtype: entry.dtype)
    }

    // MARK: - Routed expert (lazy)

    /// First touch of layer L opens its backend + verifies SHA-256; subsequent
    /// touches reuse the open backend. The backend resolves the expert to an
    /// cache-slot `(MTLBuffer, offset)` pair.
    public func routedExpert(layer L: Int, expert E: Int) throws -> TensorView {
        try ensureLayerOpened(L)
        let backend = streamersQueue.sync { streamersBox.streamers[L]! }
        let r = try backend.loadExpert(layer: 0, expert: E)
        return TensorView(
            buffer: r.buffer,
            offset: r.offset,
            length: r.size,
            scaleOffset: 0, scaleLength: 0,
            biasOffset:  0, biasLength:  0,
            shape: (UInt32(L), UInt32(E), 0, 0),
            dtype: 0)
    }

    /// Open layer L's file + verify SHA, idempotent.
    func ensureLayerOpened(_ L: Int) throws {
        try streamersQueue.sync {
            try openLayerLocked(L)
        }
    }

    /// Best-effort overlap hook for prefill: starts the same lazy layer open on
    /// the model's streamer queue without waiting for the first expert fetch.
    public func beginOpeningRoutedExpertStreamer(layer L: Int) {
        nonisolated(unsafe) let model = self
        streamersQueue.async {
            try? model.openLayerLocked(L)
        }
    }

    private func openLayerLocked(_ L: Int) throws {
        if streamersBox.streamers[L] != nil {
            return
        }
        let basename = packedExpertsLayout.layers[L].file
        let url = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(basename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelError.missingFile(name: "packed_experts/\(basename)")
        }
        if !streamersBox.layerVerified[L] {
            let manifestRel = "packed_experts/\(basename)"
            guard let entry = manifest.files[manifestRel] else {
                throw ModelError.missingFile(name: manifestRel)
            }
            switch integrityPolicy {
            case .fullSha256:
                try Sha256Verifier.verifyFile(at: url, named: manifestRel,
                                              expectedHex: entry.sha256)
            case .sizeCheckTrustedReceipt:
                try Self.verifyTrustedReceiptFileSize(url: url,
                                                      relativePath: manifestRel,
                                                      expectedSize: entry.size)
            }
            streamersBox.layerVerified[L] = true
        }
        let streamSize = UInt64(packedExpertsLayout.expertsPerLayer)
            * packedExpertsLayout.expertStride
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: streamSize,
            expertsPerLayer: packedExpertsLayout.expertsPerLayer,
            expertStride: packedExpertsLayout.expertStride,
            expertOffsets: packedExpertsLayout.layers[L].experts.map(\.offset))
        let slotCount: Int
        switch streamingMode {
        case .pread(let configuredSlotCount):
            slotCount = configuredSlotCount
        }
        streamersBox.streamers[L] = try PreadExpertStreamer(
            layout: layout,
            device: device,
            slotCount: slotCount,
            cachePolicy: expertCachePolicy)
    }

    /// Test hook: how many layer files have been opened so far.
    public func openLayerFileCount() -> Int {
        streamersQueue.sync { streamersBox.streamers.compactMap { $0 }.count }
    }

}

extension Model {

    /// Open a `.gturbo/` directory and return a typed handle. Eagerly verifies
    /// SHA-256 of `model_weights.bin` and `packed_experts/layout.json`; layer
    /// files are verified lazily on first `routedExpert(...)` touch.
    public static func load(directoryURL: URL,
                            device: MTLDevice,
                            expecting: ArchConfig = .gemma4_26B_A4B,
                            streamingMode: ExpertStreamingMode = .pread(slotCount: 16),
                            expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
                            integrityPolicy: ModelIntegrityPolicy? = nil,
                            loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil) throws -> Model {
        var stats = ModelLoadStats()
        defer {
            loadStats?.pointee = stats
        }
        let resolvedIntegrityPolicy = integrityPolicy ?? .fullSha256

        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let manifestShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let manifestSha = try Sha256Verifier.hashFile(at: manifestURL)
        stats.manifestSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - manifestShaStart
        let manifestSize = try Self.fileSize(at: manifestURL,
                                             relativePath: "manifest.json")
        let receipt: VerifiedInstallReceipt?
        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let loadedReceipt = try VerifiedInstallReceiptReader.load(directoryURL: directoryURL)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                loadedReceipt,
                directoryURL: directoryURL,
                manifestSha256: manifestSha)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
            receipt = loadedReceipt
        } else {
            receipt = nil
        }

        let manifest = try ManifestReader.load(directoryURL: directoryURL,
                                               expecting: expecting)
        if let receipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try VerifiedInstallReceiptReader.validate(receipt,
                                                      directoryURL: directoryURL,
                                                      manifest: manifest,
                                                      manifestSha256: manifestSha,
                                                      manifestSize: manifestSize)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        // Verify the small, always-touched files before mapping model data.
        let weightsURL = directoryURL.appendingPathComponent("model_weights.bin")
        let layoutURL  = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent("layout.json")
        guard let weightsEntry = manifest.files["model_weights.bin"] else {
            throw ModelError.missingFile(name: "model_weights.bin")
        }
        guard let layoutEntry = manifest.files["packed_experts/layout.json"] else {
            throw ModelError.missingFile(name: "packed_experts/layout.json")
        }
        let eagerShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try Sha256Verifier.verifyFile(at: weightsURL, named: "model_weights.bin",
                                      expectedHex: weightsEntry.sha256)
        try Sha256Verifier.verifyFile(at: layoutURL, named: "packed_experts/layout.json",
                                      expectedHex: layoutEntry.sha256)
        stats.eagerSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - eagerShaStart

        let residentIndex = try ResidentIndexReader.load(fileURL: weightsURL)

        // The resident index must account for the complete weights file.
        let attrs = try FileManager.default.attributesOfItem(atPath: weightsURL.path)
        if let fileSize = attrs[.size] as? UInt64 {
            let expected = residentIndex.header.indexSize + residentIndex.header.residentSize
            if fileSize != expected {
                throw ModelError.indexCorrupt(detail: """
                    model_weights.bin size \(fileSize) != indexSize \
                    \(residentIndex.header.indexSize) + residentSize \
                    \(residentIndex.header.residentSize) = \(expected)
                    """)
            }
        }

        let residentBuffer = try ResidentBuffer(
            fileURL: weightsURL,
            fileOffset: residentIndex.header.indexSize,
            residentSize: residentIndex.header.residentSize,
            device: device)

        let layout = try PackedExpertsLayoutReader.load(directoryURL: directoryURL)
        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try validateTrustedReceiptLayerLayout(directoryURL: directoryURL,
                                                  manifest: manifest,
                                                  layout: layout)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        return Model(
            device: device,
            config: expecting,
            streamingMode: streamingMode,
            expertCachePolicy: expertCachePolicy,
            integrityPolicy: resolvedIntegrityPolicy,
            residentBuffer: residentBuffer,
            residentIndex: residentIndex,
            packedExpertsLayout: layout,
            manifest: manifest,
            directoryURL: directoryURL)
    }

    private static func verifyTrustedReceiptFileSize(url: URL,
                                                     relativePath: String,
                                                     expectedSize: UInt64) throws {
        let actualSize = try fileSize(at: url, relativePath: relativePath)
        guard actualSize == expectedSize else {
            throw ModelError.trustedReceiptInvalid(
                detail: "\(relativePath) size \(actualSize) != \(expectedSize)")
        }
    }

    private static func fileSize(at url: URL, relativePath: String) throws -> UInt64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let sizeValue = attrs[.size] else {
            throw ModelError.trustedReceiptInvalid(detail: "missing size for \(relativePath)")
        }
        let actualSize: UInt64
        if let number = sizeValue as? NSNumber {
            actualSize = number.uint64Value
        } else if let value = sizeValue as? UInt64 {
            actualSize = value
        } else if let value = sizeValue as? Int {
            actualSize = UInt64(value)
        } else {
            throw ModelError.trustedReceiptInvalid(detail: "invalid size for \(relativePath)")
        }
        return actualSize
    }

    private static func validateTrustedReceiptLayerLayout(directoryURL: URL,
                                                          manifest: Manifest,
                                                          layout: PackedExpertsLayout) throws {
        let pageSize = UInt64(getpagesize())
        guard layout.expertStride % pageSize == 0 else {
            throw ModelError.trustedReceiptInvalid(
                detail: "expertStride \(layout.expertStride) is not page-aligned")
        }
        guard layout.numLayers == manifest.numLayers,
              layout.expertsPerLayer == manifest.expertsPerLayer,
              layout.expertStride == manifest.expertStride else {
            throw ModelError.trustedReceiptInvalid(detail: "layout does not match manifest dimensions")
        }
        for layer in layout.layers {
            guard layer.layer >= 0 && layer.layer < manifest.numLayers else {
                throw ModelError.trustedReceiptInvalid(detail: "layout layer out of range")
            }
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw ModelError.trustedReceiptInvalid(detail: "manifest missing \(relativePath)")
            }
            let expectedSize = UInt64(layout.expertsPerLayer) * layout.expertStride
            guard manifestEntry.size == expectedSize else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) manifest size \(manifestEntry.size) != \(expectedSize)")
            }
            let url = directoryURL
                .appendingPathComponent("packed_experts")
                .appendingPathComponent(layer.file)
            let actualSize = try fileSize(at: url, relativePath: relativePath)
            guard actualSize == expectedSize else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(expectedSize)")
            }
            guard layer.experts.count == layout.expertsPerLayer else {
                throw ModelError.trustedReceiptInvalid(detail: "\(relativePath) expert count mismatch")
            }
            for expert in layer.experts {
                guard expert.size == layout.expertStride else {
                    throw ModelError.trustedReceiptInvalid(
                        detail: "\(relativePath) expert \(expert.expert) size mismatch")
                }
                guard expert.offset % pageSize == 0 else {
                    throw ModelError.trustedReceiptInvalid(
                        detail: "\(relativePath) expert \(expert.expert) offset is not page-aligned")
                }
                guard expert.offset <= actualSize,
                      expert.size <= actualSize - expert.offset else {
                    throw ModelError.trustedReceiptInvalid(
                        detail: "\(relativePath) expert \(expert.expert) range exceeds file size")
                }
            }
        }
    }
}
