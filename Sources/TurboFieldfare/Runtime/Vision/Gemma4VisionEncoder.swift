import Foundation
import Metal

public enum Gemma4VisionEncodingProgress: Sendable, Equatable {
    case preprocessing
    case encodingPatchEmbeddings
    case encodingLayer(completed: Int, total: Int)
    case projecting
}

public enum Gemma4VisionEncoderError: Error, CustomStringConvertible, Equatable {
    case invalidSidecar(String)
    case outputBufferTooSmall(required: Int, actual: Int)
    case outputBufferDeviceMismatch
    case gpuCommandFailed(String)
    case allocationFailed(label: String, bytes: Int)
    case invalidProjection

    public var description: String {
        switch self {
        case .invalidSidecar(let detail):
            return "Gemma 4 vision sidecar does not match the pinned architecture: \(detail)"
        case .outputBufferTooSmall(let required, let actual):
            return "projected feature output needs \(required) bytes; buffer has \(actual)"
        case .outputBufferDeviceMismatch:
            return "projected feature output belongs to a different Metal device"
        case .gpuCommandFailed(let detail):
            return "Gemma 4 vision GPU command failed: \(detail)"
        case .allocationFailed(let label, let bytes):
            return "could not allocate \(bytes) bytes for vision \(label)"
        case .invalidProjection:
            return "Gemma 4 vision projection contained non-finite or empty features"
        }
    }
}

public struct Gemma4VisionFeatures: @unchecked Sendable {
    public static let hiddenSize = 2_816

    public let buffer: MTLBuffer
    public let tokenCount: Int
    public let contentHash: String
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let resizedWidth: Int
    public let resizedHeight: Int
    public let sidecarSnapshotHash: String

    public var byteCount: Int {
        tokenCount * Self.hiddenSize * MemoryLayout<Float16>.stride
    }
}

/// Native, bounded-memory Gemma 4 image encoder.
///
/// The language model remains untouched. A single sidecar layer is read into
/// one shared Metal slot, consumed, and overwritten only after its command
/// buffer completes. Activations and one largest-matrix BF16→FP16 conversion
/// buffer are request-local and released when `encode` returns. Only the
/// projected `[softTokenCount, 2816]` buffer escapes to the caller.
public actor Gemma4VisionEncoder {
    public static let maximumSoftTokens = 280
    public static let visionHiddenSize = 1_152
    public static let intermediateSize = 4_304
    public static let layerCount = 27
    public static let headCount = 16
    public static let headSize = 72
    public static let positionTableSize = 10_240
    public static let mlpTileRows = 64

    private static let patchProjection = "vision_tower.patch_embedder.input_proj.weight"
    private static let positionTable = "vision_tower.patch_embedder.position_embedding_table"
    private static let standardBias = "vision_tower.std_bias"
    private static let standardScale = "vision_tower.std_scale"
    private static let projector = "embed_vision.embedding_projection.weight"

    private let context: MetalContext
    private let sidecar: VisionSidecar
    private let kernels: Gemma4VisionKernels
    private let projectorFallback: PrefillInt4QMM
    private let layerNames: [[LayerTensor: String]]
    private let maximumLayerBytes: Int
    private let maximumDenseWeightBytes: Int

    public init(modelDirectory: URL,
                context: MetalContext,
                integrityPolicy: VisionSidecarIntegrityPolicy = .fullSHA256) throws {
        let openedSidecar = try VisionSidecar(
            modelDirectory: modelDirectory,
            integrityPolicy: integrityPolicy)
        let names = try Self.resolveAndValidateTensorLayout(openedSidecar)
        self.context = context
        self.sidecar = openedSidecar
        self.kernels = try Gemma4VisionKernels(context: context)
        self.projectorFallback = try PrefillInt4QMM(context: context)
        self.layerNames = names
        self.maximumLayerBytes = try names.map { layer in
            try Self.packedBytes(for: Array(layer.values), sidecar: openedSidecar)
        }.max() ?? 0
        self.maximumDenseWeightBytes = try openedSidecar.tensor(
            named: names[0][.gate]!).sizeBytes
    }

    public var sidecarSnapshotHash: String {
        sidecar.manifest.sourceSnapshotHash
    }

    public func encode(
        imageAt imageURL: URL,
        into callerOutput: MTLBuffer? = nil,
        preprocessing: Gemma4ImagePreprocessingOptions = .gemma4Default,
        onProgress: (@Sendable (Gemma4VisionEncodingProgress) -> Void)? = nil
    ) async throws -> Gemma4VisionFeatures {
        try Task.checkCancellation()
        onProgress?(.preprocessing)
        let prepared = try Gemma4ImagePreprocessor.prepare(
            imageAt: imageURL,
            device: context.device,
            options: preprocessing)
        try Task.checkCancellation()

        // Actor isolation serializes requests so the explicit ~90 MiB bound
        // cannot be multiplied by concurrent encodes on this instance.

        let projectedBytes = prepared.softTokenCount
            * Gemma4VisionFeatures.hiddenSize
            * MemoryLayout<Float16>.stride
        let output: MTLBuffer
        if let callerOutput {
            guard callerOutput.device.registryID == context.device.registryID else {
                throw Gemma4VisionEncoderError.outputBufferDeviceMismatch
            }
            guard callerOutput.length >= projectedBytes else {
                throw Gemma4VisionEncoderError.outputBufferTooSmall(
                    required: projectedBytes,
                    actual: callerOutput.length)
            }
            output = callerOutput
        } else {
            output = try allocate(label: "projected output",
                                  bytes: projectedBytes,
                                  options: .storageModeShared)
        }

        let patchCount = prepared.patchCount
        let hiddenElements = patchCount * Self.visionHiddenSize
        let hiddenBytes = hiddenElements * MemoryLayout<Float16>.stride
        let hidden = try allocate(label: "hidden", bytes: hiddenBytes)
        let normalized = try allocate(label: "normalized", bytes: hiddenBytes)
        let query = try allocate(label: "query", bytes: hiddenBytes)
        let key = try allocate(label: "key", bytes: hiddenBytes)
        let value = try allocate(label: "value", bytes: hiddenBytes)
        let attention = try allocate(label: "attention", bytes: hiddenBytes)
        let weightSlot = try allocate(label: "streamed layer weights",
                                      bytes: max(maximumLayerBytes, 2_000_000),
                                      options: .storageModeShared)
        let convertedWeight = try allocate(label: "converted dense weight",
                                           bytes: maximumDenseWeightBytes)
        let mlpGate = try allocate(label: "MLP gate tile",
                                   bytes: Self.mlpTileRows * Self.intermediateSize * 2)
        let mlpUp = try allocate(label: "MLP up tile",
                                 bytes: Self.mlpTileRows * Self.intermediateSize * 2)
        let mlpDown = try allocate(label: "MLP down tile",
                                   bytes: Self.mlpTileRows * Self.visionHiddenSize * 2)

        onProgress?(.encodingPatchEmbeddings)
        let patchWeights = try pack(
            tensorNames: [Self.patchProjection],
            into: weightSlot)
        let positionRows = try loadPositionRows(
            patchColumns: prepared.patchColumns,
            patchRows: prepared.patchRows)
        var commandBuffer = try makeCommandBuffer(label: "Gemma4 vision patch embedding")
        kernels.encodeDenseBF16(
            commandBuffer: commandBuffer,
            input: prepared.patchValues,
            weight: weightSlot,
            weightOffset: patchWeights[Self.patchProjection]!,
            convertedWeight: convertedWeight,
            output: hidden,
            rows: patchCount,
            outputColumns: Self.visionHiddenSize,
            inputColumns: Gemma4PreparedImage.patchVectorSize)
        kernels.encodeSanitize(commandBuffer: commandBuffer,
                               buffer: hidden,
                               count: hiddenElements)
        kernels.encodeAddPositions(
            commandBuffer: commandBuffer,
            hidden: hidden,
            positions: positionRows,
            patchCount: patchCount,
            patchColumns: prepared.patchColumns,
            hiddenSize: Self.visionHiddenSize)
        try finish(commandBuffer)

        for layer in 0..<Self.layerCount {
            try Task.checkCancellation()
            let names = layerNames[layer]
            let offsets = try pack(tensorNames: Array(names.values), into: weightSlot)
            commandBuffer = try makeCommandBuffer(label: "Gemma4 vision layer \(layer)")

            kernels.encodeRMSRows(
                commandBuffer: commandBuffer,
                input: hidden,
                weight: weightSlot,
                weightOffset: offsets[names[.inputNorm]!]!,
                output: normalized,
                rows: patchCount,
                width: Self.visionHiddenSize)
            encodeDense(commandBuffer, normalized, weightSlot, offsets[names[.query]!]!,
                        convertedWeight, query, patchCount, Self.visionHiddenSize, Self.visionHiddenSize)
            kernels.encodeSanitize(commandBuffer: commandBuffer, buffer: query, count: hiddenElements)
            encodeDense(commandBuffer, normalized, weightSlot, offsets[names[.key]!]!,
                        convertedWeight, key, patchCount, Self.visionHiddenSize, Self.visionHiddenSize)
            kernels.encodeSanitize(commandBuffer: commandBuffer, buffer: key, count: hiddenElements)
            encodeDense(commandBuffer, normalized, weightSlot, offsets[names[.value]!]!,
                        convertedWeight, value, patchCount, Self.visionHiddenSize, Self.visionHiddenSize)
            kernels.encodeSanitize(commandBuffer: commandBuffer, buffer: value, count: hiddenElements)
            kernels.encodeRMSHeads(
                commandBuffer: commandBuffer,
                input: query,
                weight: weightSlot,
                weightOffset: offsets[names[.queryNorm]!]!,
                output: query,
                tokens: patchCount,
                heads: Self.headCount,
                headSize: Self.headSize)
            kernels.encodeRMSHeads(
                commandBuffer: commandBuffer,
                input: key,
                weight: weightSlot,
                weightOffset: offsets[names[.keyNorm]!]!,
                output: key,
                tokens: patchCount,
                heads: Self.headCount,
                headSize: Self.headSize)
            kernels.encodeRMSHeads(
                commandBuffer: commandBuffer,
                input: value,
                weight: nil,
                output: value,
                tokens: patchCount,
                heads: Self.headCount,
                headSize: Self.headSize)
            kernels.encodeRoPE(
                commandBuffer: commandBuffer,
                query: query,
                key: key,
                tokens: patchCount,
                patchColumns: prepared.patchColumns,
                heads: Self.headCount,
                headSize: Self.headSize)
            kernels.encodeAttention(
                commandBuffer: commandBuffer,
                query: query,
                key: key,
                value: value,
                output: attention,
                tokens: patchCount,
                heads: Self.headCount,
                headSize: Self.headSize)
            encodeDense(commandBuffer, attention, weightSlot, offsets[names[.output]!]!,
                        convertedWeight, query, patchCount, Self.visionHiddenSize, Self.visionHiddenSize)
            kernels.encodeSanitize(commandBuffer: commandBuffer, buffer: query, count: hiddenElements)
            kernels.encodeRMSRows(
                commandBuffer: commandBuffer,
                input: query,
                weight: weightSlot,
                weightOffset: offsets[names[.postAttentionNorm]!]!,
                output: attention,
                rows: patchCount,
                width: Self.visionHiddenSize)
            kernels.encodeResidualAdd(
                commandBuffer: commandBuffer,
                hidden: hidden,
                branch: attention,
                count: hiddenElements)
            kernels.encodeSanitize(commandBuffer: commandBuffer, buffer: hidden, count: hiddenElements)

            kernels.encodeRMSRows(
                commandBuffer: commandBuffer,
                input: hidden,
                weight: weightSlot,
                weightOffset: offsets[names[.preFFNNorm]!]!,
                output: normalized,
                rows: patchCount,
                width: Self.visionHiddenSize)
            var tileStart = 0
            while tileStart < patchCount {
                let tileRows = min(Self.mlpTileRows, patchCount - tileStart)
                let inputOffset = tileStart * Self.visionHiddenSize * 2
                kernels.encodeDenseBF16(
                    commandBuffer: commandBuffer,
                    input: normalized,
                    inputOffset: inputOffset,
                    weight: weightSlot,
                    weightOffset: offsets[names[.gate]!]!,
                    convertedWeight: convertedWeight,
                    output: mlpGate,
                    rows: tileRows,
                    outputColumns: Self.intermediateSize,
                    inputColumns: Self.visionHiddenSize)
                kernels.encodeSanitize(commandBuffer: commandBuffer,
                                       buffer: mlpGate,
                                       count: tileRows * Self.intermediateSize)
                kernels.encodeDenseBF16(
                    commandBuffer: commandBuffer,
                    input: normalized,
                    inputOffset: inputOffset,
                    weight: weightSlot,
                    weightOffset: offsets[names[.up]!]!,
                    convertedWeight: convertedWeight,
                    output: mlpUp,
                    rows: tileRows,
                    outputColumns: Self.intermediateSize,
                    inputColumns: Self.visionHiddenSize)
                kernels.encodeSanitize(commandBuffer: commandBuffer,
                                       buffer: mlpUp,
                                       count: tileRows * Self.intermediateSize)
                kernels.encodeGELUMultiply(
                    commandBuffer: commandBuffer,
                    gate: mlpGate,
                    up: mlpUp,
                    count: tileRows * Self.intermediateSize)
                kernels.encodeDenseBF16(
                    commandBuffer: commandBuffer,
                    input: mlpGate,
                    weight: weightSlot,
                    weightOffset: offsets[names[.down]!]!,
                    convertedWeight: convertedWeight,
                    output: mlpDown,
                    rows: tileRows,
                    outputColumns: Self.visionHiddenSize,
                    inputColumns: Self.intermediateSize)
                kernels.encodeSanitize(commandBuffer: commandBuffer,
                                       buffer: mlpDown,
                                       count: tileRows * Self.visionHiddenSize)
                kernels.encodeRMSRows(
                    commandBuffer: commandBuffer,
                    input: mlpDown,
                    weight: weightSlot,
                    weightOffset: offsets[names[.postFFNNorm]!]!,
                    output: query,
                    rows: tileRows,
                    width: Self.visionHiddenSize)
                kernels.encodeResidualAdd(
                    commandBuffer: commandBuffer,
                    hidden: hidden,
                    hiddenOffset: inputOffset,
                    branch: query,
                    count: tileRows * Self.visionHiddenSize)
                kernels.encodeSanitize(commandBuffer: commandBuffer,
                                       buffer: hidden,
                                       offset: inputOffset,
                                       count: tileRows * Self.visionHiddenSize)
                tileStart += tileRows
            }
            try finish(commandBuffer)
            onProgress?(.encodingLayer(completed: layer + 1, total: Self.layerCount))
        }

        try Task.checkCancellation()
        onProgress?(.projecting)
        let finalNames = [Self.standardBias, Self.standardScale, Self.projector]
        let finalOffsets = try pack(
            tensorNames: finalNames,
            into: weightSlot,
            includeAffineFor: [Self.projector])
        let pooledBytes = prepared.softTokenCount * Self.visionHiddenSize * 2
        precondition(normalized.length >= pooledBytes)
        precondition(query.length >= pooledBytes)
        commandBuffer = try makeCommandBuffer(label: "Gemma4 vision pooling and projection")
        kernels.encodePool(
            commandBuffer: commandBuffer,
            hidden: hidden,
            standardBias: weightSlot,
            biasOffset: finalOffsets[Self.standardBias]!,
            standardScale: weightSlot,
            scaleOffset: finalOffsets[Self.standardScale]!,
            output: normalized,
            patchColumns: prepared.patchColumns,
            patchRows: prepared.patchRows,
            hiddenSize: Self.visionHiddenSize)
        kernels.encodeRMSRows(
            commandBuffer: commandBuffer,
            input: normalized,
            weight: nil,
            output: query,
            rows: prepared.softTokenCount,
            width: Self.visionHiddenSize)
        let projectorLoaded = try sidecar.readTensor(
            named: Self.projector,
            into: weightSlot,
            at: finalOffsets[Self.projector]!,
            includeAffineCompanions: true)
        guard let scaleOffset = projectorLoaded.scaleOffset,
              let biasOffset = projectorLoaded.biasOffset else {
            throw Gemma4VisionEncoderError.invalidSidecar("projector affine companions are absent")
        }
        projectorFallback.encode(
            commandBuffer: commandBuffer,
            weights: weightSlot,
            weightsOffset: projectorLoaded.weightOffset,
            scales: weightSlot,
            scalesOffset: scaleOffset,
            biases: weightSlot,
            biasesOffset: biasOffset,
            x: query,
            y: output,
            t: prepared.softTokenCount,
            n: Gemma4VisionFeatures.hiddenSize,
            k: Self.visionHiddenSize)
        try finish(commandBuffer)

        let projectedElementCount = prepared.softTokenCount
            * Gemma4VisionFeatures.hiddenSize
        let projected = output.contents().bindMemory(
            to: Float16.self,
            capacity: projectedElementCount)
        var hasNonZeroFiniteValue = false
        for index in 0..<projectedElementCount {
            let value = projected[index]
            guard value.isFinite else {
                throw Gemma4VisionEncoderError.invalidProjection
            }
            hasNonZeroFiniteValue = hasNonZeroFiniteValue || value != 0
        }
        guard hasNonZeroFiniteValue else {
            throw Gemma4VisionEncoderError.invalidProjection
        }

        return Gemma4VisionFeatures(
            buffer: output,
            tokenCount: prepared.softTokenCount,
            contentHash: prepared.contentHash,
            sourceWidth: prepared.sourceWidth,
            sourceHeight: prepared.sourceHeight,
            resizedWidth: prepared.resizedWidth,
            resizedHeight: prepared.resizedHeight,
            sidecarSnapshotHash: sidecar.manifest.sourceSnapshotHash)
    }

    private func encodeDense(_ commandBuffer: MTLCommandBuffer,
                             _ input: MTLBuffer,
                             _ weights: MTLBuffer,
                             _ weightOffset: Int,
                             _ convertedWeight: MTLBuffer,
                             _ output: MTLBuffer,
                             _ rows: Int,
                             _ outputColumns: Int,
                             _ inputColumns: Int) {
        kernels.encodeDenseBF16(
            commandBuffer: commandBuffer,
            input: input,
            weight: weights,
            weightOffset: weightOffset,
            convertedWeight: convertedWeight,
            output: output,
            rows: rows,
            outputColumns: outputColumns,
            inputColumns: inputColumns)
    }

    private func loadPositionRows(patchColumns: Int, patchRows: Int) throws -> MTLBuffer {
        let rowBytes = Self.visionHiddenSize * MemoryLayout<UInt16>.stride
        let result = try allocate(
            label: "position rows",
            bytes: (patchColumns + patchRows) * rowBytes,
            options: .storageModeShared)
        try sidecar.readTensorSlice(
            named: Self.positionTable,
            sourceByteOffset: 0,
            count: patchColumns * rowBytes,
            into: result,
            at: 0)
        // The source is [2, 10240, 1152]. Repacking only the used rows means
        // the Metal add kernel intentionally indexes y immediately after the
        // selected x rows rather than assuming the source table stride.
        try sidecar.readTensorSlice(
            named: Self.positionTable,
            sourceByteOffset: Self.positionTableSize * rowBytes,
            count: patchRows * rowBytes,
            into: result,
            at: patchColumns * rowBytes)
        return result
    }

    private func pack(tensorNames: [String],
                      into buffer: MTLBuffer,
                      includeAffineFor affineNames: Set<String> = []) throws -> [String: Int] {
        var offsets: [String: Int] = [:]
        var cursor = 0
        for name in tensorNames.sorted() {
            cursor = VisionSidecar.align(cursor, to: 16)
            let loaded = try sidecar.readTensor(
                named: name,
                into: buffer,
                at: cursor,
                includeAffineCompanions: affineNames.contains(name))
            offsets[name] = cursor
            cursor += loaded.totalBytes
        }
        return offsets
    }

    private func allocate(label: String,
                          bytes: Int,
                          options: MTLResourceOptions = .storageModePrivate) throws -> MTLBuffer {
        guard bytes > 0,
              let buffer = context.device.makeBuffer(length: bytes, options: options) else {
            throw Gemma4VisionEncoderError.allocationFailed(label: label, bytes: bytes)
        }
        buffer.label = "Gemma4 vision \(label)"
        return buffer
    }

    private func makeCommandBuffer(label: String) throws -> MTLCommandBuffer {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw Gemma4VisionEncoderError.gpuCommandFailed("could not create command buffer")
        }
        commandBuffer.label = label
        return commandBuffer
    }

    private func finish(_ commandBuffer: MTLCommandBuffer) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw Gemma4VisionEncoderError.gpuCommandFailed(error.localizedDescription)
        }
        try Task.checkCancellation()
    }
}

private extension Gemma4VisionEncoder {
    enum LayerTensor: CaseIterable, Hashable {
        case inputNorm, postAttentionNorm, preFFNNorm, postFFNNorm
        case query, key, value, output, queryNorm, keyNorm
        case gate, up, down

        var suffix: String {
            switch self {
            case .inputNorm: ".input_layernorm.weight"
            case .postAttentionNorm: ".post_attention_layernorm.weight"
            case .preFFNNorm: ".pre_feedforward_layernorm.weight"
            case .postFFNNorm: ".post_feedforward_layernorm.weight"
            case .query: ".self_attn.q_proj.linear.weight"
            case .key: ".self_attn.k_proj.linear.weight"
            case .value: ".self_attn.v_proj.linear.weight"
            case .output: ".self_attn.o_proj.linear.weight"
            case .queryNorm: ".self_attn.q_norm.weight"
            case .keyNorm: ".self_attn.k_norm.weight"
            case .gate: ".mlp.gate_proj.linear.weight"
            case .up: ".mlp.up_proj.linear.weight"
            case .down: ".mlp.down_proj.linear.weight"
            }
        }

        var expectedDimensions: [Int] {
            switch self {
            case .inputNorm, .postAttentionNorm, .preFFNNorm, .postFFNNorm:
                [Gemma4VisionEncoder.visionHiddenSize]
            case .query, .key, .value, .output:
                [Gemma4VisionEncoder.visionHiddenSize, Gemma4VisionEncoder.visionHiddenSize]
            case .queryNorm, .keyNorm:
                [Gemma4VisionEncoder.headSize]
            case .gate, .up:
                [Gemma4VisionEncoder.intermediateSize, Gemma4VisionEncoder.visionHiddenSize]
            case .down:
                [Gemma4VisionEncoder.visionHiddenSize, Gemma4VisionEncoder.intermediateSize]
            }
        }
    }

    static func resolveAndValidateTensorLayout(_ sidecar: VisionSidecar) throws
        -> [[LayerTensor: String]] {
        guard sidecar.manifest.entryCount == 356,
              sidecar.manifest.sourceTensorCount == 358 else {
            throw Gemma4VisionEncoderError.invalidSidecar(
                "expected 356 resident entries / 358 source tensors, got \(sidecar.manifest.entryCount) / \(sidecar.manifest.sourceTensorCount)")
        }
        try require(sidecar, patchProjection, dtype: 1,
                    dimensions: [visionHiddenSize, Gemma4PreparedImage.patchVectorSize])
        try require(sidecar, positionTable, dtype: 1,
                    dimensions: [2, positionTableSize, visionHiddenSize])
        try require(sidecar, standardBias, dtype: 1, dimensions: [visionHiddenSize])
        try require(sidecar, standardScale, dtype: 1, dimensions: [visionHiddenSize])
        let projection = try require(sidecar, projector, dtype: 0,
                                     dimensions: [Gemma4VisionFeatures.hiddenSize, visionHiddenSize])
        guard projection.scaleSizeBytes == Gemma4VisionFeatures.hiddenSize
                * (visionHiddenSize / Quantization.groupSize) * 2,
              projection.biasSizeBytes == projection.scaleSizeBytes else {
            throw Gemma4VisionEncoderError.invalidSidecar("projector affine companion sizes are invalid")
        }

        var result: [[LayerTensor: String]] = []
        result.reserveCapacity(layerCount)
        let allNames = sidecar.tensorNames
        for layer in 0..<layerCount {
            let layerMarker = ".layers.\(layer)."
            var resolved: [LayerTensor: String] = [:]
            for role in LayerTensor.allCases {
                let matches = allNames.filter {
                    $0.contains(layerMarker) && $0.hasSuffix(role.suffix)
                }
                guard matches.count == 1, let name = matches.first else {
                    throw Gemma4VisionEncoderError.invalidSidecar(
                        "layer \(layer) role \(role) resolved \(matches.count) tensors")
                }
                try require(sidecar, name, dtype: 1, dimensions: role.expectedDimensions)
                resolved[role] = name
            }
            result.append(resolved)
        }
        return result
    }

    @discardableResult
    static func require(_ sidecar: VisionSidecar,
                        _ name: String,
                        dtype: UInt8,
                        dimensions: [Int]) throws -> VisionTensorDescriptor {
        let tensor = try sidecar.tensor(named: name)
        guard tensor.dtype == dtype else {
            throw Gemma4VisionEncoderError.invalidSidecar(
                "\(name) dtype \(tensor.dtype), expected \(dtype)")
        }
        guard tensor.dimensions == dimensions else {
            throw Gemma4VisionEncoderError.invalidSidecar(
                "\(name) shape \(tensor.dimensions), expected \(dimensions)")
        }
        let expectedBytes = dtype == 0
            ? dimensions.reduce(1, *) / 2 // logical int4 shape
            : dimensions.reduce(1, *) * 2
        guard tensor.sizeBytes == expectedBytes else {
            throw Gemma4VisionEncoderError.invalidSidecar(
                "\(name) has \(tensor.sizeBytes) bytes, expected \(expectedBytes)")
        }
        return tensor
    }

    static func packedBytes(for names: [String], sidecar: VisionSidecar) throws -> Int {
        var cursor = 0
        for name in names.sorted() {
            cursor = VisionSidecar.align(cursor, to: 16)
            cursor += try sidecar.tensor(named: name).sizeBytes
        }
        return cursor
    }
}
