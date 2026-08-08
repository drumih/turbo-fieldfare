import Foundation
import Metal
import Darwin
import TurboFieldfareFormat

/// One routed-expert layer file, opened and verified by `K3Model`, owned by
/// the caller (the expert-streaming engine). `fileDescriptor` is a dup the
/// caller must close; the model keeps its own handle for re-verification.
public struct K3ExpertLayerFile: Sendable {
    /// 0-based layer id.
    public let layer: Int
    public let path: String
    public let fileDescriptor: Int32
    public let expertsPerLayer: Int
    public let expertStride: UInt64
    /// Absolute byte offset of each expert blob inside the file, keyed by
    /// logical expert id (the v2 layout permits a physical-rank permutation).
    public let expertOffsets: [UInt64]

    public init(layer: Int, path: String, fileDescriptor: Int32,
                expertsPerLayer: Int, expertStride: UInt64,
                expertOffsets: [UInt64]) {
        self.layer = layer
        self.path = path
        self.fileDescriptor = fileDescriptor
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.expertOffsets = expertOffsets
    }
}

/// Loaded K3 `.gturbo` v2 bundle. Resident weights live behind one mmap'd
/// `MTLBuffer`; routed experts stay on SSD in per-layer packed files opened
/// lazily on first touch (mirrors `Model`).
///
/// The tensor registry covers the full K3 schema from
/// docs/K3_DATAFLOW.md §"Checkpoint tensor names": every name is required at
/// load, a missing tensor throws (refuse-defaults), and per-tensor dtype /
/// shape / byte sizes are asserted against the manifest arch + quant slots.
public struct K3Model {
    public let device: MTLDevice
    public let config: K3ArchConfig
    public let integrityPolicy: ModelIntegrityPolicy
    public var modelID: String { manifest.modelID }
    public var sourceSnapshotHash: String? { manifest.sourceSnapshotHash }

    let residentBuffer: ResidentBuffer
    let residentIndex: ResidentIndex
    let expertsLayout: GTurboPackedExpertsLayoutV2
    let manifest: GTurboManifestV2
    let directoryURL: URL
    let modelDirectory: GTurboModelDirectory

    /// Fully-resolved resident views, keyed by full checkpoint tensor name
    /// (e.g. `language_model.model.layers.4.self_attn.q_proj.weight`).
    /// Built once at load after schema validation, so lookups never throw for
    /// schema names.
    public let tensors: [String: TensorView]

    /// Per-MoE-layer expert offsets keyed by logical expert id, 0-based layer.
    let expertOffsetsByLayer: [Int: [UInt64]]
    /// 0-based MoE layer id -> packed layer metadata.
    let layoutLayerByID: [Int: GTurboLayerV2]

    /// Lazy per-layer expert file handles, inside a reference box so `K3Model`
    /// stays a struct (same pattern as `Model.StreamersBox`).
    let layerFilesBox: LayerFilesBox
    let layerFilesQueue: DispatchQueue

    final class LayerFilesBox: @unchecked Sendable {
        var descriptors: [Int: K3ExpertLayerFile] = [:]
        var verified: Set<Int> = []
        var fds: [Int32] = []
        deinit { for fd in fds { close(fd) } }
    }

    init(device: MTLDevice,
         config: K3ArchConfig,
         integrityPolicy: ModelIntegrityPolicy,
         residentBuffer: ResidentBuffer,
         residentIndex: ResidentIndex,
         expertsLayout: GTurboPackedExpertsLayoutV2,
         manifest: GTurboManifestV2,
         directoryURL: URL,
         modelDirectory: GTurboModelDirectory,
         tensors: [String: TensorView]) {
        self.device = device
        self.config = config
        self.integrityPolicy = integrityPolicy
        self.residentBuffer = residentBuffer
        self.residentIndex = residentIndex
        self.expertsLayout = expertsLayout
        self.manifest = manifest
        self.directoryURL = directoryURL
        self.modelDirectory = modelDirectory
        self.tensors = tensors
        var byLayer: [Int: GTurboLayerV2] = [:]
        var offsets: [Int: [UInt64]] = [:]
        for layer in expertsLayout.layers {
            byLayer[layer.layer] = layer
            var byExpert = [UInt64](repeating: 0, count: expertsLayout.expertsPerLayer)
            for (position, expert) in layer.experts.enumerated() {
                let logical = expert.expert ?? position
                byExpert[logical] = expert.offset
            }
            offsets[layer.layer] = byExpert
        }
        self.layoutLayerByID = byLayer
        self.expertOffsetsByLayer = offsets
        self.layerFilesBox = LayerFilesBox()
        self.layerFilesQueue = DispatchQueue(label: "turbo-fieldfare.k3-layer-files")
    }

    // MARK: - Resident accessors

    /// Resolve any full checkpoint tensor name. Throws `tensorNotFound` for
    /// names outside the validated schema (ad-hoc lookups).
    public func tensor(_ name: String) throws -> TensorView {
        guard let view = tensors[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        return view
    }

    /// int8 g64 [vocab, hidden] embedding table.
    public var embedding: TensorView {
        tensors["language_model.model.embed_tokens.weight"]!
    }
    /// int8 g64 [vocab, hidden] untied output head.
    /// Lives at `language_model.lm_head.weight` (no `.model.` segment).
    public var lmHead: TensorView {
        tensors["language_model.lm_head.weight"]!
    }
    public var finalNorm: TensorView {
        tensors["language_model.model.norm.weight"]!
    }
    public var outputAttnResProj: TensorView {
        tensors["language_model.model.output_attn_res_proj.weight"]!
    }
    public var outputAttnResNorm: TensorView {
        tensors["language_model.model.output_attn_res_norm.weight"]!
    }

    static func layerPrefix(_ layer0: Int) -> String {
        "language_model.model.layers.\(layer0)"
    }

    public func inputNorm(layer L: Int) -> TensorView {
        tensors["\(Self.layerPrefix(L)).input_layernorm.weight"]!
    }
    public func postAttnNorm(layer L: Int) -> TensorView {
        tensors["\(Self.layerPrefix(L)).post_attention_layernorm.weight"]!
    }
    public func attnResProj(layer L: Int) -> TensorView {
        tensors["\(Self.layerPrefix(L)).self_attention_res_proj.weight"]!
    }
    public func attnResNorm(layer L: Int) -> TensorView {
        tensors["\(Self.layerPrefix(L)).self_attention_res_norm.weight"]!
    }
    public func mlpResProj(layer L: Int) -> TensorView {
        tensors["\(Self.layerPrefix(L)).mlp_res_proj.weight"]!
    }
    public func mlpResNorm(layer L: Int) -> TensorView {
        tensors["\(Self.layerPrefix(L)).mlp_res_norm.weight"]!
    }

    // KDA projections (KDA layers only).
    public func kdaQProj(layer L: Int) -> TensorView { kdaTensor(L, "q_proj.weight") }
    public func kdaKProj(layer L: Int) -> TensorView { kdaTensor(L, "k_proj.weight") }
    public func kdaVProj(layer L: Int) -> TensorView { kdaTensor(L, "v_proj.weight") }
    public func kdaGProj(layer L: Int) -> TensorView { kdaTensor(L, "g_proj.weight") }
    public func kdaOProj(layer L: Int) -> TensorView { kdaTensor(L, "o_proj.weight") }
    public func kdaFAProj(layer L: Int) -> TensorView { kdaTensor(L, "f_a_proj.weight") }
    public func kdaFBProj(layer L: Int) -> TensorView { kdaTensor(L, "f_b_proj.weight") }
    public func kdaBProj(layer L: Int) -> TensorView { kdaTensor(L, "b_proj.weight") }
    public func kdaONorm(layer L: Int) -> TensorView { kdaTensor(L, "o_norm.weight") }
    public func kdaALog(layer L: Int) -> TensorView { kdaTensor(L, "A_log") }
    public func kdaDTBias(layer L: Int) -> TensorView { kdaTensor(L, "dt_bias") }
    public func kdaConvWeight(layer L: Int, _ qkv: String) -> TensorView {
        kdaTensor(L, "\(qkv)_conv1d.weight")
    }
    public func kdaConvBias(layer L: Int, _ qkv: String) -> TensorView {
        kdaTensor(L, "\(qkv)_conv1d.bias")
    }
    private func kdaTensor(_ L: Int, _ suffix: String) -> TensorView {
        tensors["\(Self.layerPrefix(L)).self_attn.\(suffix)"]!
    }

    // MLA projections (MLA layers only).
    public func mlaQAProj(layer L: Int) -> TensorView { mlaTensor(L, "q_a_proj.weight") }
    public func mlaQANorm(layer L: Int) -> TensorView { mlaTensor(L, "q_a_layernorm.weight") }
    public func mlaQBProj(layer L: Int) -> TensorView { mlaTensor(L, "q_b_proj.weight") }
    public func mlaKVAProj(layer L: Int) -> TensorView {
        mlaTensor(L, "kv_a_proj_with_mqa.weight")
    }
    public func mlaKVANorm(layer L: Int) -> TensorView { mlaTensor(L, "kv_a_layernorm.weight") }
    public func mlaKVBProj(layer L: Int) -> TensorView { mlaTensor(L, "kv_b_proj.weight") }
    public func mlaGProj(layer L: Int) -> TensorView { mlaTensor(L, "g_proj.weight") }
    public func mlaOProj(layer L: Int) -> TensorView { mlaTensor(L, "o_proj.weight") }
    private func mlaTensor(_ L: Int, _ suffix: String) -> TensorView {
        tensors["\(Self.layerPrefix(L)).self_attn.\(suffix)"]!
    }

    // MoE resident tensors (non-dense layers only; routed experts stream).
    public func routerGate(layer L: Int) -> TensorView {
        moeTensor(L, "gate.weight")
    }
    public func routerCorrectionBias(layer L: Int) -> TensorView {
        moeTensor(L, "gate.e_score_correction_bias")
    }
    public func routedDownProj(layer L: Int) -> TensorView {
        moeTensor(L, "routed_expert_down_proj.weight")
    }
    public func routedUpProj(layer L: Int) -> TensorView {
        moeTensor(L, "routed_expert_up_proj.weight")
    }
    public func routedExpertNorm(layer L: Int) -> TensorView {
        moeTensor(L, "routed_expert_norm.weight")
    }
    public func sharedGateProj(layer L: Int) -> TensorView {
        moeTensor(L, "shared_experts.gate_proj.weight")
    }
    public func sharedUpProj(layer L: Int) -> TensorView {
        moeTensor(L, "shared_experts.up_proj.weight")
    }
    public func sharedDownProj(layer L: Int) -> TensorView {
        moeTensor(L, "shared_experts.down_proj.weight")
    }
    private func moeTensor(_ L: Int, _ suffix: String) -> TensorView {
        tensors["\(Self.layerPrefix(L)).block_sparse_moe.\(suffix)"]!
    }

    // Dense MLP (layer 0 only).
    public func denseGateProj(layer L: Int) -> TensorView { denseTensor(L, "gate_proj.weight") }
    public func denseUpProj(layer L: Int) -> TensorView { denseTensor(L, "up_proj.weight") }
    public func denseDownProj(layer L: Int) -> TensorView { denseTensor(L, "down_proj.weight") }
    private func denseTensor(_ L: Int, _ suffix: String) -> TensorView {
        tensors["\(Self.layerPrefix(L)).mlp.\(suffix)"]!
    }

    // MARK: - Routed expert layer files (lazy)

    /// Open + verify layer L's packed-expert file on first touch, then hand
    /// the caller a dup'd descriptor. Mirrors `Model.ensureLayerOpened`:
    /// SHA-256 under `.fullSha256`, size-only under `.sizeCheckTrustedReceipt`.
    public func expertLayerFile(_ layer0: Int) throws -> K3ExpertLayerFile {
        guard config.isMoE(layer0: layer0) else {
            throw ModelError.indexCorrupt(
                detail: "layer \(layer0) is dense; it has no packed-expert file")
        }
        guard let layoutLayer = layoutLayerByID[layer0],
              let offsets = expertOffsetsByLayer[layer0] else {
            throw ModelError.indexCorrupt(
                detail: "packed-expert layout has no entry for MoE layer \(layer0)")
        }
        return try layerFilesQueue.sync {
            if let cached = layerFilesBox.descriptors[layer0] {
                let dupFD = fcntl(cached.fileDescriptor, F_DUPFD_CLOEXEC, 0)
                guard dupFD >= 0 else {
                    throw ModelError.posixFailed(call: "fcntl(F_DUPFD_CLOEXEC)", errno: errno)
                }
                return K3ExpertLayerFile(
                    layer: layer0, path: cached.path, fileDescriptor: dupFD,
                    expertsPerLayer: cached.expertsPerLayer,
                    expertStride: cached.expertStride,
                    expertOffsets: cached.expertOffsets)
            }
            let manifestRel = "packed_experts/\(layoutLayer.file)"
            guard let entry = manifest.files[manifestRel] else {
                throw ModelError.missingFile(name: manifestRel)
            }
            let fd = try modelDirectory.openFile(manifestRel)
            do {
                let actualSize = try modelDirectory.fileSize(
                    fileDescriptor: fd, relativePath: manifestRel)
                guard actualSize == entry.size else {
                    throw ModelError.tensorSizeMismatch(
                        name: manifestRel, expected: entry.size, actual: actualSize)
                }
                switch integrityPolicy {
                case .fullSha256:
                    try Sha256Verifier.verifyFile(fileDescriptor: fd,
                                                  named: manifestRel,
                                                  expectedHex: entry.sha256)
                case .sizeCheckTrustedReceipt:
                    break
                }
            } catch {
                close(fd)
                throw error
            }
            layerFilesBox.fds.append(fd)
            let path = directoryURL
                .appendingPathComponent("packed_experts")
                .appendingPathComponent(layoutLayer.file)
                .path
            let descriptor = K3ExpertLayerFile(
                layer: layer0, path: path, fileDescriptor: fd,
                expertsPerLayer: expertsLayout.expertsPerLayer,
                expertStride: expertsLayout.expertStride,
                expertOffsets: offsets)
            layerFilesBox.descriptors[layer0] = descriptor
            return descriptor
        }
    }

    /// Test hook: how many layer files have been opened so far.
    public func openLayerFileCount() -> Int {
        layerFilesQueue.sync { layerFilesBox.descriptors.count }
    }
}

extension K3Model {

    /// K3's layout records six tensor slices for every routed expert. The
    /// pinned 93-layer / 896-expert bundle serializes to about 129 MiB, so
    /// the generic 16 MiB metadata ceiling used by v1 is not applicable.
    /// Keep this bounded: metadata is still read before it is authenticated
    /// against the manifest hash.
    static let packedExpertsLayoutMaxBytes: UInt64 = 256 * 1024 * 1024

    /// Open a K3 `.gturbo` v2 bundle and return a typed handle. Eagerly
    /// verifies SHA-256 of `model_weights.bin` and `packed_experts/layout.json`
    /// (same policy as `Model.load`); layer files are verified lazily on
    /// first `expertLayerFile(...)` touch.
    public static func load(bundleURL: URL,
                            device: MTLDevice,
                            expecting: K3ArchConfig = .kimiK3,
                            integrityPolicy: ModelIntegrityPolicy? = nil,
                            loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil)
        throws -> K3Model {
        var stats = ModelLoadStats()
        defer { loadStats?.pointee = stats }
        let resolvedIntegrityPolicy = integrityPolicy ?? .fullSha256
        let modelDirectory = try GTurboModelDirectory(rootURL: bundleURL)
        let manifestFD: Int32
        do { manifestFD = try modelDirectory.openFile("manifest.json") }
        catch ModelError.missingFile { throw ModelError.partialInstall(path: bundleURL.path) }
        defer { close(manifestFD) }
        let manifestData = try modelDirectory.readMetadata(
            fileDescriptor: manifestFD, relativePath: "manifest.json",
            maxBytes: ManifestReader.defaultMaxBytes)
        let manifestSize = UInt64(manifestData.count)
        let manifestShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let manifestSha = Sha256Verifier.hashData(manifestData)
        stats.manifestSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - manifestShaStart

        let receipt: VerifiedInstallReceipt?
        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let receiptFD: Int32
            do {
                receiptFD = try modelDirectory.openFile(VerifiedInstallReceiptReader.fileName)
            } catch ModelError.missingFile {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(VerifiedInstallReceiptReader.fileName) is missing")
            }
            defer { close(receiptFD) }
            let receiptData = try modelDirectory.readMetadata(
                fileDescriptor: receiptFD,
                relativePath: VerifiedInstallReceiptReader.fileName,
                maxBytes: VerifiedInstallReceiptReader.defaultMaxBytes)
            let loadedReceipt = try VerifiedInstallReceiptReader.decode(data: receiptData)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                loadedReceipt, directoryURL: bundleURL, manifestSha256: manifestSha)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
            receipt = loadedReceipt
        } else {
            receipt = nil
        }

        let manifest: GTurboManifestV2
        do {
            manifest = try GTurboManifestCodecV2.decode(manifestData)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
        let runtimeQuant = try K3ArchConfig.validate(
            manifest: manifest, expecting: expecting)

        if let receipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try validateTrustedReceipt(receipt,
                                       directoryURL: bundleURL,
                                       manifest: manifest,
                                       manifestSha256: manifestSha,
                                       manifestSize: manifestSize)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        // Verify the small, always-touched files before mapping model data.
        let weightsURL = bundleURL.appendingPathComponent("model_weights.bin")
        guard let weightsEntry = manifest.files["model_weights.bin"] else {
            throw ModelError.missingFile(name: "model_weights.bin")
        }
        guard let layoutEntry = manifest.files["packed_experts/layout.json"] else {
            throw ModelError.missingFile(name: "packed_experts/layout.json")
        }
        let weightsFD = try modelDirectory.openFile("model_weights.bin")
        defer { close(weightsFD) }
        let layoutFD = try modelDirectory.openFile("packed_experts/layout.json")
        defer { close(layoutFD) }
        let layoutData = try modelDirectory.readMetadata(
            fileDescriptor: layoutFD, relativePath: "packed_experts/layout.json",
            maxBytes: packedExpertsLayoutMaxBytes)
        guard UInt64(layoutData.count) == layoutEntry.size else {
            throw ModelError.tensorSizeMismatch(
                name: "packed_experts/layout.json",
                expected: layoutEntry.size,
                actual: UInt64(layoutData.count))
        }
        let weightsSize = try modelDirectory.fileSize(
            fileDescriptor: weightsFD, relativePath: "model_weights.bin")
        guard weightsSize == weightsEntry.size else {
            throw ModelError.tensorSizeMismatch(
                name: "model_weights.bin",
                expected: weightsEntry.size,
                actual: weightsSize)
        }
        let eagerShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try Sha256Verifier.verifyFile(fileDescriptor: weightsFD,
                                      named: "model_weights.bin",
                                      expectedHex: weightsEntry.sha256)
        guard Sha256Verifier.hashData(layoutData).lowercased()
                == layoutEntry.sha256.lowercased() else {
            throw ModelError.checksumMismatch(file: "packed_experts/layout.json")
        }
        stats.eagerSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - eagerShaStart

        let layout: GTurboPackedExpertsLayoutV2
        do {
            layout = try GTurboPackedExpertsLayoutCodecV2.decode(layoutData)
            try GTurboV2StructuralValidator.crossValidate(manifest: manifest, layout: layout)
        } catch {
            throw ModelError.indexCorrupt(detail: "layout.json: \(error)")
        }
        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            for layer in layout.layers {
                let relativePath = "packed_experts/\(layer.file)"
                guard let manifestEntry = manifest.files[relativePath] else {
                    throw ModelError.trustedReceiptInvalid(
                        detail: "manifest missing \(relativePath)")
                }
                let actualSize = try modelDirectory.fileSize(relativePath)
                guard actualSize == manifestEntry.size else {
                    throw ModelError.trustedReceiptInvalid(
                        detail: "\(relativePath) size \(actualSize) != \(manifestEntry.size)")
                }
            }
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        let residentIndex = try ResidentIndexReader.load(
            fileDescriptor: weightsFD, displayPath: "model_weights.bin")
        try validateRuntimeSchema(residentIndex: residentIndex,
                                  layout: layout,
                                  config: expecting,
                                  quant: runtimeQuant)

        // The resident index must account for the complete weights file.
        let (expectedSize, overflow) = residentIndex.header.indexSize
            .addingReportingOverflow(residentIndex.header.residentSize)
        if overflow || weightsSize != expectedSize {
            throw ModelError.indexCorrupt(detail: """
                model_weights.bin size \(weightsSize) != indexSize \
                \(residentIndex.header.indexSize) + residentSize \
                \(residentIndex.header.residentSize) = \(expectedSize)
                """)
        }

        let residentBuffer = try ResidentBuffer(
            fileURL: weightsURL,
            fileOffset: residentIndex.header.indexSize,
            residentSize: residentIndex.header.residentSize,
            device: device,
            fileDescriptor: weightsFD)

        let tensors = try buildTensorRegistry(residentIndex: residentIndex,
                                              residentBuffer: residentBuffer,
                                              config: expecting,
                                              quant: runtimeQuant)
        return K3Model(
            device: device,
            config: expecting,
            integrityPolicy: resolvedIntegrityPolicy,
            residentBuffer: residentBuffer,
            residentIndex: residentIndex,
            expertsLayout: layout,
            manifest: manifest,
            directoryURL: bundleURL,
            modelDirectory: modelDirectory,
            tensors: tensors)
    }

    /// `VerifiedInstallReceiptReader.validate` against the v2 manifest (the
    /// house validator is v1-typed; the file-set / size / sha rules are
    /// identical).
    private static func validateTrustedReceipt(_ receipt: VerifiedInstallReceipt,
                                               directoryURL: URL,
                                               manifest: GTurboManifestV2,
                                               manifestSha256: String,
                                               manifestSize: UInt64) throws {
        try VerifiedInstallReceiptReader.validateManifestBinding(
            receipt, directoryURL: directoryURL, manifestSha256: manifestSha256)
        var expectedFiles = Set(manifest.files.keys)
        expectedFiles.insert("manifest.json")
        let receiptFiles = Set(receipt.files.keys)
        guard receiptFiles == expectedFiles else {
            let missing = expectedFiles.subtracting(receiptFiles).sorted()
            let extra = receiptFiles.subtracting(expectedFiles).sorted()
            throw ModelError.trustedReceiptInvalid(
                detail: "receipt file set mismatch missing=\(missing) extra=\(extra)")
        }
        guard let manifestReceiptEntry = receipt.files["manifest.json"] else {
            throw ModelError.trustedReceiptInvalid(detail: "receipt missing manifest.json")
        }
        guard manifestReceiptEntry.size == manifestSize else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest.json size mismatch")
        }
        guard manifestReceiptEntry.sha256.lowercased() == manifestSha256.lowercased() else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest.json SHA mismatch")
        }
        for (rel, manifestEntry) in manifest.files {
            guard let receiptEntry = receipt.files[rel] else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt missing \(rel)")
            }
            guard receiptEntry.size == manifestEntry.size else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt size mismatch for \(rel)")
            }
            guard receiptEntry.sha256.lowercased() == manifestEntry.sha256.lowercased() else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt SHA mismatch for \(rel)")
            }
        }
    }

    // MARK: - Runtime schema

    /// One required resident tensor: dtype + logical dims derived from the
    /// (already arch-validated) config, so toy fixtures validate against their
    /// own dimensions.
    enum SchemaKind {
        case affine(rows: Int, columns: Int, bits: Int)
        case bf16Vector(Int)
        case fp32Vector(Int)
        case fp32Matrix(rows: Int, columns: Int)
    }

    struct SchemaEntry {
        let name: String
        let kind: SchemaKind
    }

    /// The full K3 resident schema (docs/K3_DATAFLOW.md §"Checkpoint tensor
    /// names"), generated from the config so the same code validates the
    /// canonical 93-layer model and tiny test bundles.
    static func schemaEntries(config c: K3ArchConfig,
                              quant: GTurboManifestQuantV2
                                  = KimiK3FormatProfile.quantInt4) -> [SchemaEntry] {
        var entries: [SchemaEntry] = []
        let hidden = c.hiddenSize
        let attentionBits = quant.attention.weightBits
        let embeddingBits = quant.embedding.weightBits
        let sharedBits = quant.sharedExpert.weightBits
        let latentBits = quant.latentProjection.weightBits
        let denseBits = quant.denseMLP.weightBits

        entries.append(SchemaEntry(
            name: "language_model.model.embed_tokens.weight",
            kind: .affine(rows: c.vocabSize, columns: hidden, bits: embeddingBits)))
        entries.append(SchemaEntry(
            name: "language_model.lm_head.weight",
            kind: .affine(rows: c.vocabSize, columns: hidden, bits: embeddingBits)))
        entries.append(SchemaEntry(
            name: "language_model.model.norm.weight", kind: .bf16Vector(hidden)))
        entries.append(SchemaEntry(
            name: "language_model.model.output_attn_res_proj.weight",
            kind: .bf16Vector(hidden)))
        entries.append(SchemaEntry(
            name: "language_model.model.output_attn_res_norm.weight",
            kind: .bf16Vector(hidden)))

        let kdaP = c.kdaChannels
        for layer in 0..<c.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            for suffix in ["input_layernorm.weight",
                           "post_attention_layernorm.weight",
                           "self_attention_res_proj.weight",
                           "self_attention_res_norm.weight",
                           "mlp_res_proj.weight",
                           "mlp_res_norm.weight"] {
                entries.append(SchemaEntry(name: "\(prefix).\(suffix)",
                                           kind: .bf16Vector(hidden)))
            }
            if c.isKDA(layer0: layer) {
                let attn = "\(prefix).self_attn"
                for stem in ["q_proj.weight", "k_proj.weight", "v_proj.weight",
                             "g_proj.weight"] {
                    entries.append(SchemaEntry(
                        name: "\(attn).\(stem)",
                        kind: .affine(rows: kdaP, columns: hidden, bits: attentionBits)))
                }
                entries.append(SchemaEntry(
                    name: "\(attn).o_proj.weight",
                    kind: .affine(rows: hidden, columns: kdaP, bits: attentionBits)))
                entries.append(SchemaEntry(
                    name: "\(attn).f_a_proj.weight",
                    kind: .affine(rows: c.kdaDecayLowRankSize, columns: hidden,
                                  bits: attentionBits)))
                entries.append(SchemaEntry(
                    name: "\(attn).f_b_proj.weight",
                    kind: .affine(rows: kdaP, columns: c.kdaDecayLowRankSize,
                                  bits: attentionBits)))
                entries.append(SchemaEntry(
                    name: "\(attn).b_proj.weight",
                    kind: .affine(rows: c.kdaNumHeads, columns: hidden,
                                  bits: attentionBits)))
                for qkv in ["q", "k", "v"] {
                    entries.append(SchemaEntry(
                        name: "\(attn).\(qkv)_conv1d.weight",
                        kind: .fp32Matrix(rows: kdaP, columns: c.kdaConvWidth)))
                    entries.append(SchemaEntry(
                        name: "\(attn).\(qkv)_conv1d.bias",
                        kind: .fp32Vector(kdaP)))
                }
                entries.append(SchemaEntry(name: "\(attn).A_log",
                                           kind: .fp32Vector(c.kdaNumHeads)))
                entries.append(SchemaEntry(name: "\(attn).dt_bias",
                                           kind: .fp32Vector(kdaP)))
                entries.append(SchemaEntry(name: "\(attn).o_norm.weight",
                                           kind: .fp32Vector(c.kdaHeadDim)))
            } else {
                let attn = "\(prefix).self_attn"
                entries.append(SchemaEntry(
                    name: "\(attn).q_a_proj.weight",
                    kind: .affine(rows: c.mlaQLoraRank, columns: hidden,
                                  bits: attentionBits)))
                entries.append(SchemaEntry(
                    name: "\(attn).q_a_layernorm.weight",
                    kind: .fp32Vector(c.mlaQLoraRank)))
                entries.append(SchemaEntry(
                    name: "\(attn).q_b_proj.weight",
                    kind: .affine(rows: c.mlaQElements, columns: c.mlaQLoraRank,
                                  bits: attentionBits)))
                entries.append(SchemaEntry(
                    name: "\(attn).kv_a_proj_with_mqa.weight",
                    kind: .affine(rows: c.mlaCacheRowElements, columns: hidden,
                                  bits: attentionBits)))
                entries.append(SchemaEntry(
                    name: "\(attn).kv_a_layernorm.weight",
                    kind: .fp32Vector(c.mlaKVLoraRank)))
                entries.append(SchemaEntry(
                    name: "\(attn).kv_b_proj.weight",
                    kind: .affine(rows: c.mlaKVBRows, columns: c.mlaKVLoraRank,
                                  bits: attentionBits)))
                entries.append(SchemaEntry(
                    name: "\(attn).g_proj.weight",
                    kind: .affine(rows: c.mlaOutputElements, columns: hidden,
                                  bits: attentionBits)))
                entries.append(SchemaEntry(
                    name: "\(attn).o_proj.weight",
                    kind: .affine(rows: hidden, columns: c.mlaOutputElements,
                                  bits: attentionBits)))
            }
            if c.isMoE(layer0: layer) {
                let moe = "\(prefix).block_sparse_moe"
                entries.append(SchemaEntry(
                    name: "\(moe).gate.weight",
                    kind: .fp32Matrix(rows: c.moeNumExperts, columns: hidden)))
                entries.append(SchemaEntry(
                    name: "\(moe).gate.e_score_correction_bias",
                    kind: .fp32Vector(c.moeNumExperts)))
                entries.append(SchemaEntry(
                    name: "\(moe).routed_expert_down_proj.weight",
                    kind: .affine(rows: c.moeLatentBottleneckSize, columns: hidden,
                                  bits: latentBits)))
                entries.append(SchemaEntry(
                    name: "\(moe).routed_expert_up_proj.weight",
                    kind: .affine(rows: hidden, columns: c.moeLatentBottleneckSize,
                                  bits: latentBits)))
                entries.append(SchemaEntry(
                    name: "\(moe).routed_expert_norm.weight",
                    kind: .bf16Vector(c.moeLatentBottleneckSize)))
                entries.append(SchemaEntry(
                    name: "\(moe).shared_experts.gate_proj.weight",
                    kind: .affine(rows: c.moeSharedExpertIntermediateSize,
                                  columns: hidden, bits: sharedBits)))
                entries.append(SchemaEntry(
                    name: "\(moe).shared_experts.up_proj.weight",
                    kind: .affine(rows: c.moeSharedExpertIntermediateSize,
                                  columns: hidden, bits: sharedBits)))
                entries.append(SchemaEntry(
                    name: "\(moe).shared_experts.down_proj.weight",
                    kind: .affine(rows: hidden,
                                  columns: c.moeSharedExpertIntermediateSize,
                                  bits: sharedBits)))
            } else {
                let mlp = "\(prefix).mlp"
                entries.append(SchemaEntry(
                    name: "\(mlp).gate_proj.weight",
                    kind: .affine(rows: c.denseMLPIntermediateSize, columns: hidden,
                                  bits: denseBits)))
                entries.append(SchemaEntry(
                    name: "\(mlp).up_proj.weight",
                    kind: .affine(rows: c.denseMLPIntermediateSize, columns: hidden,
                                  bits: denseBits)))
                entries.append(SchemaEntry(
                    name: "\(mlp).down_proj.weight",
                    kind: .affine(rows: hidden, columns: c.denseMLPIntermediateSize,
                                  bits: denseBits)))
            }
        }
        return entries
    }

    /// Assert every schema tensor exists in the resident index with the exact
    /// dtype / shape / byte sizes the kernels are built against, and that the
    /// packed-expert blobs match the canonical MXFP4 subtensor layout.
    /// Missing or mismatched entries throw `ModelError.indexCorrupt`.
    static func validateRuntimeSchema(residentIndex: ResidentIndex,
                                      layout: GTurboPackedExpertsLayoutV2,
                                      config: K3ArchConfig,
                                      quant: GTurboManifestQuantV2
                                          = KimiK3FormatProfile.quantInt4) throws {
        let groupSize = Quantization.groupSize

        func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else {
                throw ModelError.indexCorrupt(detail: "\(field) byte count overflows UInt64")
            }
            return value
        }

        func entry(_ name: String) throws -> ResidentIndexEntry {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            return entry
        }

        func validate(_ item: SchemaEntry) throws {
            let e = try entry(item.name)
            switch item.kind {
            case .affine(let rows, let columns, let bits):
                guard columns % groupSize == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "\(item.name) columns \(columns) not a multiple of \(groupSize)")
                }
                guard let r = UInt32(exactly: rows), let c = UInt32(exactly: columns),
                      r > 0, c > 0 else {
                    throw ModelError.indexCorrupt(detail: "\(item.name) has invalid dimensions")
                }
                let elements = try checkedMultiply(UInt64(rows), UInt64(columns),
                                                   field: item.name)
                let bitCount = try checkedMultiply(elements, UInt64(bits), field: item.name)
                guard bitCount % 8 == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "\(item.name) packed byte count is fractional")
                }
                let groups = UInt64(columns / groupSize)
                let auxBytes = try checkedMultiply(
                    try checkedMultiply(UInt64(rows), groups, field: item.name),
                    UInt64(MemoryLayout<UInt16>.size), field: item.name)
                let primaryAlignment: UInt64 = bits == 4
                    ? UInt64(MemoryLayout<UInt16>.alignment) : 1
                guard e.dtype == GTurboFormatV1.DType.u32.rawValue,
                      e.shape.0 == r, e.shape.1 == c, e.shape.2 == 0, e.shape.3 == 0,
                      e.sizeBytes == bitCount / 8,
                      e.scaleSize == auxBytes, e.biasSize == auxBytes,
                      e.fileOffset % primaryAlignment == 0,
                      e.scaleOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0,
                      e.biasOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "\(item.name) affine metadata mismatch")
                }
            case .bf16Vector(let count):
                try validateVector(e, name: item.name, count: count,
                                   dtype: GTurboFormatV1.DType.bf16.rawValue,
                                   elementSize: MemoryLayout<UInt16>.size)
            case .fp32Vector(let count):
                try validateVector(e, name: item.name, count: count,
                                   dtype: GTurboFormatV1.DType.fp32.rawValue,
                                   elementSize: MemoryLayout<Float>.size)
            case .fp32Matrix(let rows, let columns):
                guard let r = UInt32(exactly: rows), let c = UInt32(exactly: columns),
                      r > 0, c > 0 else {
                    throw ModelError.indexCorrupt(detail: "\(item.name) has invalid dimensions")
                }
                let bytes = try checkedMultiply(
                    try checkedMultiply(UInt64(rows), UInt64(columns), field: item.name),
                    UInt64(MemoryLayout<Float>.size), field: item.name)
                guard e.dtype == GTurboFormatV1.DType.fp32.rawValue,
                      e.shape.0 == r, e.shape.1 == c, e.shape.2 == 0, e.shape.3 == 0,
                      e.sizeBytes == bytes,
                      e.scaleOffset == 0, e.scaleSize == 0,
                      e.biasOffset == 0, e.biasSize == 0,
                      e.fileOffset % UInt64(MemoryLayout<Float>.alignment) == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "\(item.name) does not match the required FP32 schema")
                }
            }
        }

        func validateVector(_ e: ResidentIndexEntry, name: String, count: Int,
                            dtype: UInt8, elementSize: Int) throws {
            guard let logicalCount = UInt32(exactly: count), logicalCount > 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            let expectedBytes = try checkedMultiply(UInt64(logicalCount),
                                                    UInt64(elementSize), field: name)
            guard e.dtype == dtype,
                  e.shape.0 == logicalCount,
                  e.shape.1 == 0, e.shape.2 == 0, e.shape.3 == 0,
                  e.sizeBytes == expectedBytes,
                  e.scaleOffset == 0, e.scaleSize == 0,
                  e.biasOffset == 0, e.biasSize == 0,
                  e.fileOffset % UInt64(elementSize) == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "\(name) does not match the required vector schema")
            }
        }

        for item in schemaEntries(config: config, quant: quant) {
            try validate(item)
        }

        // Packed expert blobs: canonical six-subtensor MXFP4 layout, identical
        // across experts (the streaming engine and the MoE kernels assume it).
        let canonical = K3ExpertSubtensorOffsets.canonical(
            dLatent: UInt32(config.moeLatentBottleneckSize),
            intermediate: UInt32(config.moeExpertIntermediateSize))
        let latent = UInt32(config.moeLatentBottleneckSize)
        let inter = UInt32(config.moeExpertIntermediateSize)
        let expectedSubTensors: [String: GTurboSubTensorV2] = [
            "w1_packed": GTurboSubTensorV2(
                offset: UInt64(canonical.w1PackedOff),
                size: UInt64(inter) * UInt64(latent) / 2,
                dtype: GTurboFormatV2.mxfp4PackedDType,
                shape: [inter, latent], bits: GTurboFormatV2.mxfp4PackedBits),
            "w1_scales": GTurboSubTensorV2(
                offset: UInt64(canonical.w1ScalesOff),
                size: UInt64(inter) * UInt64(latent) / 32,
                dtype: GTurboFormatV2.mxfp4ScaleDType,
                shape: [inter, latent / 32], bits: GTurboFormatV2.mxfp4ScaleBits),
            "w2_packed": GTurboSubTensorV2(
                offset: UInt64(canonical.w2PackedOff),
                size: UInt64(latent) * UInt64(inter) / 2,
                dtype: GTurboFormatV2.mxfp4PackedDType,
                shape: [latent, inter], bits: GTurboFormatV2.mxfp4PackedBits),
            "w2_scales": GTurboSubTensorV2(
                offset: UInt64(canonical.w2ScalesOff),
                size: UInt64(latent) * UInt64(inter) / 32,
                dtype: GTurboFormatV2.mxfp4ScaleDType,
                shape: [latent, inter / 32], bits: GTurboFormatV2.mxfp4ScaleBits),
            "w3_packed": GTurboSubTensorV2(
                offset: UInt64(canonical.w3PackedOff),
                size: UInt64(inter) * UInt64(latent) / 2,
                dtype: GTurboFormatV2.mxfp4PackedDType,
                shape: [inter, latent], bits: GTurboFormatV2.mxfp4PackedBits),
            "w3_scales": GTurboSubTensorV2(
                offset: UInt64(canonical.w3ScalesOff),
                size: UInt64(inter) * UInt64(latent) / 32,
                dtype: GTurboFormatV2.mxfp4ScaleDType,
                shape: [inter, latent / 32], bits: GTurboFormatV2.mxfp4ScaleBits),
        ]
        for layer in layout.layers {
            for (position, expert) in layer.experts.enumerated() {
                let logical = expert.expert ?? position
                for (name, expected) in expectedSubTensors {
                    guard let actual = expert.tensors[name], actual == expected else {
                        throw ModelError.indexCorrupt(
                            detail: "layer \(layer.layer) expert \(logical) subtensor "
                                + "\(name) does not match the canonical MXFP4 layout")
                    }
                }
            }
        }
    }

    /// Build the eager name → TensorView registry (offset resolution mirrors
    /// `Model.resident(name:)`). Called after `validateRuntimeSchema`, so a
    /// missing entry here is an internal inconsistency, not a schema error.
    private static func buildTensorRegistry(residentIndex: ResidentIndex,
                                            residentBuffer: ResidentBuffer,
                                            config: K3ArchConfig,
                                            quant: GTurboManifestQuantV2) throws
        -> [String: TensorView] {
        let residentFileOffset = residentIndex.header.indexSize
        func relative(_ absolute: UInt64, size: UInt64, name: String, field: String) throws
            -> UInt64 {
            if size == 0 {
                guard absolute == 0 else {
                    throw ModelError.indexCorrupt(
                        detail: "\(name).\(field) has an absent nonzero offset")
                }
                return 0
            }
            guard absolute >= residentFileOffset else {
                throw ModelError.indexCorrupt(
                    detail: "\(name).\(field) precedes the resident payload")
            }
            let rel = absolute - residentFileOffset
            guard rel <= residentIndex.header.residentSize,
                  size <= residentIndex.header.residentSize - rel else {
                throw ModelError.indexCorrupt(
                    detail: "\(name).\(field) exceeds the resident payload")
            }
            return rel
        }
        var registry: [String: TensorView] = [:]
        for item in schemaEntries(config: config, quant: quant) {
            guard let entry = residentIndex.entries[item.name] else {
                throw ModelError.tensorNotFound(name: item.name)
            }
            let weights = try relative(entry.fileOffset, size: entry.sizeBytes,
                                       name: item.name, field: "weights")
            let scales = try relative(entry.scaleOffset, size: entry.scaleSize,
                                      name: item.name, field: "scales")
            let biases = try relative(entry.biasOffset, size: entry.biasSize,
                                      name: item.name, field: "biases")
            registry[item.name] = TensorView(
                buffer: residentBuffer.buffer,
                offset: weights,
                length: entry.sizeBytes,
                scaleOffset: scales, scaleLength: entry.scaleSize,
                biasOffset: biases, biasLength: entry.biasSize,
                shape: entry.shape,
                dtype: entry.dtype)
        }
        return registry
    }
}
