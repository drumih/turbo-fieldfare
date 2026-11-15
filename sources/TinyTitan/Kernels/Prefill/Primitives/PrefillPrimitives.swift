import Foundation
import Metal
import TinyTitanCore

final class PrefillEmbedLookupInt4 {
    private let pso: MTLComputePipelineState

    init(context: MetalContext, weightBits: Int = 4) throws {
        precondition([4, 8].contains(weightBits))
        self.pso = try context.pipeline(
            "prefill_embed_lookup_affine_block",
            constants: [
                MetalFunctionConstant(
                    index: 78,
                    value: .uint32(UInt32(weightBits)))
            ])
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        table: MTLBuffer, tableOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        tokens: MTLBuffer, tokensOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        t: UInt32,
        d: UInt32,
        outScale: Float,
        vocab: UInt32
    ) throws {
        precondition(
            d % UInt32(Quantization.groupSize) == 0,
            "D must be a multiple of \(Quantization.groupSize)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(table, offset: tableOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(tokens, offset: tokensOffset, index: 3)
        enc.setBuffer(out, offset: outOffset, index: 4)
        var tVar = t
        var dVar = d
        var scaleVar = outScale
        var vocabVar = vocab
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size, index: 7)
        enc.setBytes(&vocabVar, length: MemoryLayout<UInt32>.size, index: 8)
        enc.dispatchThreads(
            MTLSize(width: Int(d), height: Int(t), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
}

final class PrefillRMSNorm {
    private let psoBF16W: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoBF16W = try context.pipeline("prefill_rmsnorm_bf16w_block")
    }

    func encodeBF16W(
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        weight: MTLBuffer, weightOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        t: UInt32,
        d: UInt32,
        eps: Float
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(psoBF16W)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var tVar = t
        var dVar = d
        var epsVar = eps
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 5)
        let threads = min(Int(psoBF16W.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(t), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

final class PrefillInt4QMM {
    private let pso: MTLComputePipelineState

    init(context: MetalContext, weightBits: Int = 4) throws {
        precondition([4, 8].contains(weightBits))
        self.pso = try context.pipeline(
            "prefill_dequant_affine_qmm_f16_block",
            constants: [
                MetalFunctionConstant(
                    index: 78,
                    value: .uint32(UInt32(weightBits)))
            ])
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        t: Int,
        n: Int,
        k: Int
    ) throws {
        precondition(
            k % Quantization.groupSize == 0,
            "K must be a multiple of \(Quantization.groupSize)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        var tVar = UInt32(t)
        var nVar = UInt32(n)
        var kVar = UInt32(k)
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&kVar, length: MemoryLayout<UInt32>.size, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: (n + 7) / 8, height: (t + 7) / 8, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
    }
}
