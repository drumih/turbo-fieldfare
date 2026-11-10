import Foundation
import Metal

final class PrefillSharedExpert {
    private let shared: SharedExpertRuntime
    private let context: MetalContext
    private let batchedGEMM: MTLComputePipelineState?
    private let geluMul: MTLComputePipelineState?
    private var batchScratch: (gate: MTLBuffer, up: MTLBuffer, act: MTLBuffer, elements: Int)?

    /// Below this many tokens the per-token GEMV path stays: a 64-row matrix
    /// tile would be mostly padding.
    static let minimumBatchedTokens = 32

    init(context: MetalContext, weightBits: Int = 8) throws {
        self.shared = try SharedExpertRuntime(context: context, weightBits: weightBits)
        self.context = context
        let gemmFunctionName: String? = switch weightBits {
        case 4: "mpp_prefill_affine_threadgroup_f16"
        case 8: "mpp_prefill_affine_int8_threadgroup_f16"
        default: nil
        }
        let batchedEnabled = gemmFunctionName != nil
            && ProcessInfo.processInfo.environment["TURBO_FIELDFARE_PREFILL_SHARED_BATCHED"] != "0"
        var batchedGEMM: MTLComputePipelineState?
        var geluMul: MTLComputePipelineState?
        if batchedEnabled {
            do {
                let library = try MetalContext.privateLibrary(device: context.device, module: "tensorops")
                guard let gemmFunctionName, let function = library.makeFunction(name: gemmFunctionName) else {
                    throw SharedExpertInt8Error.dimensionMismatch("\(gemmFunctionName ?? "?") symbol unavailable")
                }
                batchedGEMM = try context.device.makeComputePipelineState(function: function)
                geluMul = try context.pipeline("prefill_gelu_mul_rows")
            } catch {
                batchedGEMM = nil
                geluMul = nil
                FileHandle.standardError.write(Data(
                    "PrefillSharedExpert: batched path unavailable: \(error)\n".utf8))
            }
        }
        self.batchedGEMM = batchedGEMM
        self.geluMul = geluMul
    }

    var batchedPathAvailable: Bool { batchedGEMM != nil }

    private func encodeInt8GEMM(_ cb: MTLCommandBuffer,
                                pipeline: MTLComputePipelineState,
                                projection: SharedExpertInt8Proj,
                                x: MTLBuffer, xOffset: Int,
                                y: MTLBuffer, yOffset: Int,
                                m: Int) {
        guard let encoder = cb.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(projection.weights, offset: projection.weightsOffset, index: 0)
        encoder.setBuffer(projection.scales, offset: projection.scalesOffset, index: 1)
        encoder.setBuffer(projection.biases, offset: projection.biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = projection.rows
        var kValue = projection.cols
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(projection.rows) + MPPPrefillInt4QMM.tileN - 1) / MPPPrefillInt4QMM.tileN,
                    height: (m + MPPPrefillInt4QMM.tileM - 1) / MPPPrefillInt4QMM.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func ensureBatchScratch(elements: Int) throws -> (gate: MTLBuffer, up: MTLBuffer, act: MTLBuffer) {
        if let batchScratch, batchScratch.elements >= elements {
            return (batchScratch.gate, batchScratch.up, batchScratch.act)
        }
        let length = elements * MemoryLayout<Float16>.stride
        guard let gate = context.device.makeBuffer(length: length, options: .storageModePrivate),
              let up = context.device.makeBuffer(length: length, options: .storageModePrivate),
              let act = context.device.makeBuffer(length: length, options: .storageModePrivate) else {
            throw SharedExpertInt8Error.scratchTooSmall("batched shared expert scratch of \(length) bytes")
        }
        batchScratch = (gate, up, act, elements)
        return (gate, up, act)
    }

    func encodeBlock(commandBuffer cb: MTLCommandBuffer,
                            x: MTLBuffer,
                            xOffset: Int = 0,
                            y: MTLBuffer,
                            yOffset: Int = 0,
                            gate: SharedExpertInt8Proj,
                            up: SharedExpertInt8Proj,
                            down: SharedExpertInt8Proj,
                            scratchGate: MTLBuffer,
                            scratchGateOffset: Int = 0,
                            scratchUp: MTLBuffer,
                            scratchUpOffset: Int = 0,
                            scratchAct: MTLBuffer,
                            scratchActOffset: Int = 0,
                            queryCount: Int,
                            d: Int,
                            intermediate: Int,
                            xStrideElements: Int,
                            yStrideElements: Int) throws {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(d > 0, "d must be positive")
        precondition(intermediate > 0, "intermediate must be positive")
        precondition(xStrideElements >= d, "x stride is too small")
        precondition(yStrideElements >= d, "y stride is too small")
        guard gate.rows == UInt32(intermediate), gate.cols == UInt32(d),
              up.rows == UInt32(intermediate), up.cols == UInt32(d),
              down.rows == UInt32(d), down.cols == UInt32(intermediate) else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "expected gate/up=(\(intermediate),\(d)) down=(\(d),\(intermediate))")
        }

        let halfBytes = MemoryLayout<Float16>.stride
        if let batchedGEMM, let geluMul,
           queryCount >= Self.minimumBatchedTokens,
           xStrideElements == d, yStrideElements == d,
           d.isMultiple(of: Quantization.groupSize),
           intermediate.isMultiple(of: Quantization.groupSize) {
            let elements = queryCount * intermediate
            let scratch = try ensureBatchScratch(elements: elements)
            encodeInt8GEMM(cb, pipeline: batchedGEMM, projection: gate,
                           x: x, xOffset: xOffset, y: scratch.gate, yOffset: 0, m: queryCount)
            encodeInt8GEMM(cb, pipeline: batchedGEMM, projection: up,
                           x: x, xOffset: xOffset, y: scratch.up, yOffset: 0, m: queryCount)
            guard let encoder = cb.makeComputeCommandEncoder() else { return }
            encoder.setComputePipelineState(geluMul)
            encoder.setBuffer(scratch.gate, offset: 0, index: 0)
            encoder.setBuffer(scratch.up, offset: 0, index: 1)
            encoder.setBuffer(scratch.act, offset: 0, index: 2)
            var count = UInt32(elements)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.dispatchThreads(MTLSize(width: elements, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            encoder.endEncoding()
            encodeInt8GEMM(cb, pipeline: batchedGEMM, projection: down,
                           x: scratch.act, xOffset: 0, y: y, yOffset: yOffset, m: queryCount)
            return
        }
        for row in 0..<queryCount {
            try shared.encode(commandBuffer: cb,
                              x: x,
                              xOffset: xOffset + row * xStrideElements * halfBytes,
                              gate: gate,
                              up: up,
                              down: down,
                              y: y,
                              yOffset: yOffset + row * yStrideElements * halfBytes,
                              scratchGate: scratchGate,
                              scratchGateOffset: scratchGateOffset,
                              scratchUp: scratchUp,
                              scratchUpOffset: scratchUpOffset,
                              scratchAct: scratchAct,
                              scratchActOffset: scratchActOffset)
        }
    }
}
