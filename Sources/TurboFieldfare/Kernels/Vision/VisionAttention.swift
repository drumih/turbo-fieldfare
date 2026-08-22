import Foundation
import Metal

public final class VisionAttention {
    public enum Variant: String, Sendable {
        case native72
        case native72Q16
        case mppTensorOps72
        case mlxPadded80
        case mlxSteel80
    }

    public static let headDimension = 72
    public static let paddedHeadDimension = 80
    private let pipeline: MTLComputePipelineState
    private let mlxPipeline: MTLComputePipelineState?
    private let padPipeline: MTLComputePipelineState?
    private let unpadPipeline: MTLComputePipelineState?
    private let mlxQ: MTLBuffer?
    private let mlxK: MTLBuffer?
    private let mlxV: MTLBuffer?
    private let mlxOutput: MTLBuffer?
    private let mppPadded: Bool
    private let queryTile: Int
    public let variant: Variant

    /// True when the projections write the padded head-major layout directly and
    /// attention writes unpadded output, so no pad or unpad dispatch runs.
    ///
    /// Decided once, here: the constructor is told via `allowFusedPaddedLayout`
    /// whether the linear path can write the padded head-major store. The
    /// runtime used to AND this flag with that capability at encode time, which
    /// gave the decision two owners — a legal flag combination then cleared and
    /// blit-filled fused buffers no kernel read while the requested MPP kernel
    /// silently fell back to the native one.
    public let usesFusedPaddedLayout: Bool

    /// Softmax phase mode for the MPP kernel. `.parallel` spreads max, exp and
    /// sum across all 128 threads and is not bit-identical; `.parallelExp` keeps
    /// the tile sum sequential per query and is bit-identical to `.serial`.
    public enum SoftmaxPhase: UInt32, Sendable {
        case serial = 0
        case parallel = 1
        case parallelExp = 2
    }

    public let softmaxPhase: SoftmaxPhase
    public var usesParallelSoftmax: Bool { softmaxPhase == .parallel }

    /// Destination buffers for the QKV projections under the fused layout.
    public var fusedQ: MTLBuffer? { usesFusedPaddedLayout ? mlxQ : nil }
    public var fusedK: MTLBuffer? { usesFusedPaddedLayout ? mlxK : nil }
    public var fusedV: MTLBuffer? { usesFusedPaddedLayout ? mlxV : nil }

    public var additionalScratchBytes: Int {
        [mlxQ, mlxK, mlxV, mlxOutput]
            .compactMap(\.self)
            .reduce(0) { $0 + $1.length }
    }

    /// Zeroes the padded lanes the projections never write. Called once per image;
    /// nothing writes lanes 72 to 79 afterwards, so they stay zero across layers.
    public func encodeClearPadding(commandBuffer: MTLCommandBuffer) {
        guard usesFusedPaddedLayout,
              let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        for buffer in [mlxQ, mlxK, mlxV].compactMap(\.self) {
            blit.fill(buffer: buffer, range: 0..<buffer.length, value: 0)
        }
        blit.endEncoding()
    }

    /// Query rows per threadgroup in the *native* kernel, which is not always
    /// the tile the accelerated path dispatches with. Dispatching the MPP
    /// variant's 16 against a kernel built for 8 left the upper half of every
    /// attention output unwritten — silently, because the fallback only runs
    /// when the caller cannot supply the padded layout.
    private let nativeQueryTile: Int

    public init(context: MetalContext,
                environment: [String: String] = ProcessInfo.processInfo.environment,
                allowFusedPaddedLayout: Bool = true) throws {
        if let path = environment["TURBO_FIELDFARE_VISION_MLX_METALLIB"] {
            variant = .mlxSteel80
            queryTile = 32
            pipeline = try context.pipeline("vision_attention_online_72")
            nativeQueryTile = 8
            padPipeline = try context.pipeline("vision_pad_heads_72_to_80")
            unpadPipeline = try context.pipeline("vision_unpad_heads_80_to_72")
            let library = try context.device.makeLibrary(URL: URL(fileURLWithPath: path))
            let constants = MTLFunctionConstantValues()
            var disabled = false
            for index in [200, 201, 300, 301, 302] {
                constants.setConstantValue(
                    &disabled, type: .bool, index: index)
            }
            let function = try library.makeFunction(
                name: "steel_attention_bfloat16_bq32_bk32_bd80_wm4_wn1_maskbfloat16",
                constantValues: constants)
            mlxPipeline = try context.device.makeComputePipelineState(function: function)
            let paddedBytes = 2_520 * 16 * 80 * MemoryLayout<UInt16>.stride
            mlxQ = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
            mlxK = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
            mlxV = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
            mlxOutput = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
            mppPadded = false
            usesFusedPaddedLayout = false
            softmaxPhase = .serial
        } else if environment["TURBO_FIELDFARE_VISION_ATTENTION_PAD80"] == "1" {
            variant = .mlxPadded80
            queryTile = 8
            pipeline = try context.pipeline("vision_attention_online_80")
            nativeQueryTile = 8
            mlxPipeline = nil
            padPipeline = nil
            unpadPipeline = nil
            mlxQ = nil
            mlxK = nil
            mlxV = nil
            mlxOutput = nil
            mppPadded = false
            usesFusedPaddedLayout = false
            softmaxPhase = .serial
        } else if environment["TURBO_FIELDFARE_VISION_ATTENTION_MPP"] == "1" {
            variant = .mppTensorOps72
            queryTile = 16
            let library = try MetalContext.privateLibrary(
                device: context.device, module: "tensorops",
                includeVisionTensorOps: true)
            let pv80 = environment[
                "TURBO_FIELDFARE_VISION_ATTENTION_PV80"] == "1"
            guard let function = library.makeFunction(
                name: pv80
                    ? "mpp_vision_attention_bf16_pv80"
                    : "mpp_vision_attention_bf16") else {
                throw MetalError.missingFunction("mpp_vision_attention_bf16")
            }
            pipeline = try context.pipeline("vision_attention_online_72")
            nativeQueryTile = 8
            mlxPipeline = try context.device.makeComputePipelineState(function: function)
            // When the linear path cannot write the padded head-major store the
            // MPP kernel still runs, through the pad/unpad roundtrip, instead
            // of silently degrading to the native kernel while `variant`
            // reported otherwise.
            let fused = allowFusedPaddedLayout
                && environment["TURBO_FIELDFARE_VISION_PAD_ROUNDTRIP"] != "1"
            usesFusedPaddedLayout = fused
            padPipeline = fused
                ? nil : try context.pipeline("vision_pad_heads_72_to_80")
            unpadPipeline = fused
                ? nil : try context.pipeline("vision_unpad_heads_80_to_72")
            // The fused layout is addressed by padded GEMM rows, not patch rows.
            let maximumRows = fused
                ? Self.paddedRowCount(VisionConfig().maximumPatches)
                : VisionConfig().maximumPatches
            let paddedBytes = maximumRows * 16 * Self.paddedHeadDimension
                * MemoryLayout<UInt16>.stride
            mlxQ = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
            mlxK = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
            mlxV = try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
            mlxOutput = fused
                ? nil
                : try Self.makePrivateBuffer(device: context.device, bytes: paddedBytes)
            mppPadded = true
            // parallelExp is the default: bit-identical to the serial phase and
            // measurably faster. SERIAL_SOFTMAX=1 is the rollback.
            if environment["TURBO_FIELDFARE_VISION_ATTENTION_PARALLEL_SOFTMAX"] == "1" {
                softmaxPhase = .parallel
            } else if environment[
                "TURBO_FIELDFARE_VISION_ATTENTION_SERIAL_SOFTMAX"] == "1" {
                softmaxPhase = .serial
            } else {
                softmaxPhase = .parallelExp
            }
        } else if environment["TURBO_FIELDFARE_VISION_ATTENTION_Q8"] == "1" {
            variant = .native72
            queryTile = 8
            pipeline = try context.pipeline("vision_attention_online_72")
            nativeQueryTile = 8
            mlxPipeline = nil
            padPipeline = nil
            unpadPipeline = nil
            mlxQ = nil
            mlxK = nil
            mlxV = nil
            mlxOutput = nil
            mppPadded = false
            usesFusedPaddedLayout = false
            softmaxPhase = .serial
        } else {
            variant = .native72Q16
            queryTile = 16
            pipeline = try context.pipeline("vision_attention_online_72_q16")
            nativeQueryTile = 16
            mlxPipeline = nil
            padPipeline = nil
            unpadPipeline = nil
            mlxQ = nil
            mlxK = nil
            mlxV = nil
            mlxOutput = nil
            mppPadded = false
            usesFusedPaddedLayout = false
            softmaxPhase = .serial
        }
    }

    package static func paddedRowCount(_ rows: Int) -> Int {
        ((rows + 63) / 64) * 64
    }

    /// Phases of the padded attention path. Non-padded variants report `.core` only.
    public enum Phase: String, Sendable {
        case pad
        case core
        case unpad
    }

    public func encode(commandBuffer: MTLCommandBuffer,
                       q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
                       output: MTLBuffer,
                       sequenceLength: Int, numHeads: Int = 16,
                       inputRowStride: Int = 0) {
        encodePhased(q: q, k: k, v: v, output: output,
                     sequenceLength: sequenceLength, numHeads: numHeads,
                     inputRowStride: inputRowStride) { _, body in
            body(commandBuffer)
        }
    }

    /// Encodes attention one phase at a time, letting the caller choose the command
    /// buffer per phase. `encode` routes every phase to a single buffer; the stage
    /// profiler gives each phase its own buffer to attribute pad/core/unpad time.
    /// Under the fused layout only `.core` runs.
    public func encodePhased(
        q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, output: MTLBuffer,
        sequenceLength: Int, numHeads: Int = 16,
        inputRowStride: Int = 0,
        run: (Phase, (MTLCommandBuffer) -> Void) -> Void
    ) {
        precondition(sequenceLength > 0 && numHeads > 0)
        // The stride is the caller's confirmation that Q/K/V really are in the
        // padded head-major store this object committed to at construction
        // (via `allowFusedPaddedLayout`, which carries the linear path's
        // capability). Taking this branch on `usesFusedPaddedLayout` alone
        // would read a row-major call as 80-stride head-major — a precondition
        // failure, and silent garbage under -Ounchecked — so stride 0 still
        // routes to the native kernel as a last-resort fallback.
        if usesFusedPaddedLayout, inputRowStride > 0, let mlxPipeline {
            precondition(
                inputRowStride >= sequenceLength,
                "the fused layout reads head-major padded Q/K/V and needs their row stride")
            run(.core) { commandBuffer in
                encodeMPPTensorOps(
                    commandBuffer: commandBuffer,
                    sequenceLength: sequenceLength, numHeads: numHeads,
                    pipeline: mlxPipeline,
                    paddedQ: q, paddedK: k, paddedV: v,
                    paddedOutput: output,
                    inputRowStride: inputRowStride,
                    unpaddedOutputHeadDim: Self.headDimension)
            }
            return
        }
        if let mlxPipeline, let padPipeline, let unpadPipeline,
           let mlxQ, let mlxK, let mlxV, let mlxOutput {
            run(.pad) { commandBuffer in
                encodePad(commandBuffer: commandBuffer, q: q, k: k, v: v,
                          sequenceLength: sequenceLength, numHeads: numHeads,
                          padPipeline: padPipeline,
                          paddedQ: mlxQ, paddedK: mlxK, paddedV: mlxV)
            }
            run(.core) { commandBuffer in
                if mppPadded {
                    encodeMPPTensorOps(
                        commandBuffer: commandBuffer,
                        sequenceLength: sequenceLength, numHeads: numHeads,
                        pipeline: mlxPipeline,
                        paddedQ: mlxQ, paddedK: mlxK, paddedV: mlxV,
                        paddedOutput: mlxOutput,
                        inputRowStride: 0, unpaddedOutputHeadDim: 0)
                } else {
                    encodeMLXSteel(
                        commandBuffer: commandBuffer,
                        sequenceLength: sequenceLength, numHeads: numHeads,
                        pipeline: mlxPipeline,
                        paddedQ: mlxQ, paddedK: mlxK, paddedV: mlxV,
                        paddedOutput: mlxOutput)
                }
            }
            run(.unpad) { commandBuffer in
                encodeUnpad(commandBuffer: commandBuffer, output: output,
                            sequenceLength: sequenceLength, numHeads: numHeads,
                            unpadPipeline: unpadPipeline, paddedOutput: mlxOutput)
            }
            return
        }
        run(.core) { commandBuffer in
            encodeNative(commandBuffer: commandBuffer, q: q, k: k, v: v,
                         output: output,
                         sequenceLength: sequenceLength, numHeads: numHeads)
        }
    }

    private func encodeNative(commandBuffer: MTLCommandBuffer,
                              q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
                              output: MTLBuffer,
                              sequenceLength: Int, numHeads: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var length = UInt32(sequenceLength)
        var heads = UInt32(numHeads)
        encoder.setBytes(&length, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: (sequenceLength + nativeQueryTile - 1) / nativeQueryTile,
                    height: numHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodePad(
        commandBuffer: MTLCommandBuffer,
        q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
        sequenceLength: Int, numHeads: Int,
        padPipeline: MTLComputePipelineState,
        paddedQ: MTLBuffer, paddedK: MTLBuffer, paddedV: MTLBuffer
    ) {
        guard let pad = commandBuffer.makeComputeCommandEncoder() else { return }
        pad.setComputePipelineState(padPipeline)
        pad.setBuffer(q, offset: 0, index: 0)
        pad.setBuffer(k, offset: 0, index: 1)
        pad.setBuffer(v, offset: 0, index: 2)
        pad.setBuffer(paddedQ, offset: 0, index: 3)
        pad.setBuffer(paddedK, offset: 0, index: 4)
        pad.setBuffer(paddedV, offset: 0, index: 5)
        var rows = UInt32(sequenceLength), heads = UInt32(numHeads)
        pad.setBytes(&rows, length: 4, index: 6)
        pad.setBytes(&heads, length: 4, index: 7)
        pad.dispatchThreads(
            MTLSize(width: sequenceLength * numHeads * 80, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        pad.endEncoding()
    }

    private func encodeUnpad(
        commandBuffer: MTLCommandBuffer,
        output: MTLBuffer,
        sequenceLength: Int, numHeads: Int,
        unpadPipeline: MTLComputePipelineState,
        paddedOutput: MTLBuffer
    ) {
        guard let unpad = commandBuffer.makeComputeCommandEncoder() else { return }
        var rows = UInt32(sequenceLength), heads = UInt32(numHeads)
        unpad.setComputePipelineState(unpadPipeline)
        unpad.setBuffer(paddedOutput, offset: 0, index: 0)
        unpad.setBuffer(output, offset: 0, index: 1)
        unpad.setBytes(&rows, length: 4, index: 2)
        unpad.setBytes(&heads, length: 4, index: 3)
        unpad.dispatchThreads(
            MTLSize(width: sequenceLength * numHeads * 72, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        unpad.endEncoding()
    }

    private func encodeMLXSteel(
        commandBuffer: MTLCommandBuffer,
        sequenceLength: Int, numHeads: Int,
        pipeline: MTLComputePipelineState,
        paddedQ: MTLBuffer, paddedK: MTLBuffer, paddedV: MTLBuffer,
        paddedOutput: MTLBuffer
    ) {
        var parameters = MLXAttentionParameters(
            sequenceLength: sequenceLength, heads: numHeads)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(paddedQ, offset: 0, index: 0)
        encoder.setBuffer(paddedK, offset: 0, index: 1)
        encoder.setBuffer(paddedV, offset: 0, index: 2)
        encoder.setBuffer(paddedOutput, offset: 0, index: 3)
        encoder.setBytes(&parameters,
                         length: MemoryLayout<MLXAttentionParameters>.stride,
                         index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: (sequenceLength + 31) / 32,
                    height: numHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeMPPTensorOps(
        commandBuffer: MTLCommandBuffer,
        sequenceLength: Int, numHeads: Int,
        pipeline: MTLComputePipelineState,
        paddedQ: MTLBuffer, paddedK: MTLBuffer, paddedV: MTLBuffer,
        paddedOutput: MTLBuffer,
        inputRowStride: Int,
        unpaddedOutputHeadDim: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var rows = UInt32(sequenceLength), heads = UInt32(numHeads)
        var layout = VisionAttentionLayout(
            inputRowStride: UInt32(inputRowStride),
            unpaddedOutputHeadDim: UInt32(unpaddedOutputHeadDim),
            parallelSoftmax: softmaxPhase.rawValue)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(paddedQ, offset: 0, index: 0)
        encoder.setBuffer(paddedK, offset: 0, index: 1)
        encoder.setBuffer(paddedV, offset: 0, index: 2)
        encoder.setBuffer(paddedOutput, offset: 0, index: 3)
        encoder.setBytes(&rows, length: 4, index: 4)
        encoder.setBytes(&heads, length: 4, index: 5)
        encoder.setBytes(&layout,
                         length: MemoryLayout<VisionAttentionLayout>.stride,
                         index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (sequenceLength + 15) / 16,
                    height: numHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func makePrivateBuffer(device: MTLDevice, bytes: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        return buffer
    }

}

private struct VisionAttentionLayout {
    var inputRowStride: UInt32
    var unpaddedOutputHeadDim: UInt32
    var parallelSoftmax: UInt32
}

private struct MLXAttentionParameters {
    var batch: Int32 = 1
    var heads: Int32
    var dimension: Int32
    var queryLength: Int32
    var keyLength: Int32
    var gqaFactor: Int32 = 1
    var scale: Float = 1
    var queryBlocks: Int32
    var keyBlocks: Int32
    var alignedQueryBlocks: Int32
    var alignedKeyBlocks: Int32
    var queryRemainder: Int32
    var keyRemainder: Int32
    var queryOffset: Int32 = 0
    var queryStrideBatch: Int64
    var queryStrideHead: Int64
    var queryStrideSequence: Int64
    var keyStrideBatch: Int64
    var keyStrideHead: Int64
    var keyStrideSequence: Int64
    var valueStrideBatch: Int64
    var valueStrideHead: Int64
    var valueStrideSequence: Int64
    var outputStrideBatch: Int64
    var outputStrideHead: Int64
    var outputStrideSequence: Int64

    init(sequenceLength: Int, heads: Int) {
        self.heads = Int32(heads)
        self.dimension = 80
        queryLength = Int32(sequenceLength)
        keyLength = Int32(sequenceLength)
        queryBlocks = Int32((sequenceLength + 31) / 32)
        keyBlocks = queryBlocks
        alignedQueryBlocks = Int32(sequenceLength / 32)
        alignedKeyBlocks = alignedQueryBlocks
        queryRemainder = Int32(sequenceLength % 32)
        keyRemainder = queryRemainder
        let batchStride = Int64(heads * sequenceLength * 80)
        let headStride = Int64(sequenceLength * 80)
        let sequenceStride: Int64 = 80
        queryStrideBatch = batchStride
        queryStrideHead = headStride
        queryStrideSequence = sequenceStride
        keyStrideBatch = batchStride
        keyStrideHead = headStride
        keyStrideSequence = sequenceStride
        valueStrideBatch = batchStride
        valueStrideHead = headStride
        valueStrideSequence = sequenceStride
        outputStrideBatch = batchStride
        outputStrideHead = headStride
        outputStrideSequence = sequenceStride
    }
}
