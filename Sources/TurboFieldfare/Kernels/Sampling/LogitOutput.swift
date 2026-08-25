import Foundation
import Metal

/// Swift wrapper for `logit_softcap_softmax`.
///
///   z'[i] = softcap * tanh(z[i] / softcap)
///   p[i]  = exp(z'[i] - max_j z'[j]) / sum_j exp(z'[j] - max_j z'[j])
///
/// The reference path uses one threadgroup; the opt-in tiled path computes
/// per-tile statistics, merges them, then normalizes in parallel. Both use
/// FP16 storage and FP32 accumulation. The softcap value lives in the kernel
/// signature so callers and tests can disable it with a very large number.
final class LogitSoftcapSoftmax {
    private static let tileSize = 1024
    private static let maxVocabulary = 262_144

    private let referencePSO: MTLComputePipelineState
    private let tilePSO: MTLComputePipelineState
    private let reducePSO: MTLComputePipelineState
    private let normalizePSO: MTLComputePipelineState
    private let tileMaxima: MTLBuffer
    private let tileSums: MTLBuffer
    private let globalStatistics: MTLBuffer
    private let useTiled: Bool

    init(context: MetalContext, useTiled: Bool = false) throws {
        self.referencePSO = try context.pipeline("logit_softcap_softmax")
        self.tilePSO = try context.pipeline("logit_softcap_softmax_tile")
        self.reducePSO = try context.pipeline("logit_softcap_softmax_reduce")
        self.normalizePSO = try context.pipeline("logit_softcap_softmax_normalize")
        let maxTiles = Self.maxVocabulary / Self.tileSize
        guard let tileMaxima = context.device.makeBuffer(
                  length: maxTiles * MemoryLayout<Float>.stride,
                  options: .storageModePrivate),
              let tileSums = context.device.makeBuffer(
                  length: maxTiles * MemoryLayout<Float>.stride,
                  options: .storageModePrivate),
              let globalStatistics = context.device.makeBuffer(
                  length: 2 * MemoryLayout<Float>.stride,
                  options: .storageModePrivate)
        else {
            throw MetalError.noDevice
        }
        self.tileMaxima = tileMaxima
        self.tileSums = tileSums
        self.globalStatistics = globalStatistics
        self.useTiled = useTiled
    }

    /// Encodes the kernel onto `commandBuffer`. `logits` and `probs` are FP16
    /// buffers of length `v`. Gemma 4 uses `softcap=30.0`.
    func encode(commandBuffer: MTLCommandBuffer,
                       logits: MTLBuffer,
                       probs: MTLBuffer,
                       v: UInt32,
                       softcap: Float = 30.0) {
        guard useTiled, v > 0, v <= UInt32(Self.maxVocabulary) else {
            encodeReference(commandBuffer: commandBuffer, logits: logits,
                            probs: probs, v: v, softcap: softcap)
            return
        }

        let tileCount = (Int(v) + Self.tileSize - 1) / Self.tileSize
        let threads = MTLSize(width: 256, height: 1, depth: 1)
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(tilePSO)
            enc.setBuffer(logits, offset: 0, index: 0)
            enc.setBuffer(tileMaxima, offset: 0, index: 1)
            enc.setBuffer(tileSums, offset: 0, index: 2)
            var vVar = v
            var softcapVar = softcap
            enc.setBytes(&vVar, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&softcapVar, length: MemoryLayout<Float>.size, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: tileCount, height: 1, depth: 1),
                threadsPerThreadgroup: threads)
            enc.endEncoding()
        }

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(reducePSO)
            enc.setBuffer(tileMaxima, offset: 0, index: 0)
            enc.setBuffer(tileSums, offset: 0, index: 1)
            enc.setBuffer(globalStatistics, offset: 0, index: 2)
            var tileCountVar = UInt32(tileCount)
            enc.setBytes(&tileCountVar,
                         length: MemoryLayout<UInt32>.size, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: threads)
            enc.endEncoding()
        }

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(normalizePSO)
            enc.setBuffer(logits, offset: 0, index: 0)
            enc.setBuffer(probs, offset: 0, index: 1)
            enc.setBuffer(globalStatistics, offset: 0, index: 2)
            var vVar = v
            var softcapVar = softcap
            enc.setBytes(&vVar, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&softcapVar, length: MemoryLayout<Float>.size, index: 4)
            enc.dispatchThreads(
                MTLSize(width: Int(v), height: 1, depth: 1),
                threadsPerThreadgroup: threads)
            enc.endEncoding()
        }
    }

    private func encodeReference(commandBuffer: MTLCommandBuffer,
                                 logits: MTLBuffer,
                                 probs: MTLBuffer,
                                 v: UInt32,
                                 softcap: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(referencePSO)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(probs,  offset: 0, index: 1)
        var vVar       = v
        var softcapVar = softcap
        enc.setBytes(&vVar,       length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&softcapVar, length: MemoryLayout<Float>.size,  index: 3)

        let threadsPerGroup = min(
            Int(referencePSO.maxTotalThreadsPerThreadgroup), 256)
        let gridSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let tgSize   = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }
}

/// Swift wrapper for `sample`.
///
/// Reads softmaxed probabilities (output of `LogitSoftcapSoftmax`) and writes
/// one UInt32 token id. With `temperature == 0` performs greedy argmax and
/// ignores top-k / top-p / seed. With `temperature > 0` applies top-p against
/// the full distribution, caps the surviving set with top-k, then applies
/// temperature sharpening (`p^(1/T)`) and samples via a seeded PRNG. The
/// plain-temperature fast path uses the same
/// `(seed, position, row)` Gumbel stream as the fused lm_head sampler.
final class Sample {
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("sample")
    }

    /// Encodes the sampler. `probs` is an FP16 buffer of length `v`. `outToken`
    /// must point at storage for one UInt32. The seed is the full 64-bit PRNG
    /// state — tests pass identical seeds to assert determinism.
    func encode(commandBuffer: MTLCommandBuffer,
                       probs: MTLBuffer,
                       outToken: MTLBuffer,
                       v: UInt32,
                       temperature: Float = 1.0,
                       topK: UInt32 = 0,
                       topP: Float = 1.0,
                       seed: UInt64,
                       position: UInt32 = 0) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(probs,    offset: 0, index: 0)
        enc.setBuffer(outToken, offset: 0, index: 1)
        var vVar    = v
        var tVar    = temperature
        var kVar    = topK
        var pVar    = topP
        var sVar    = seed
        var posVar  = position
        enc.setBytes(&vVar, length: MemoryLayout<UInt32>.size,  index: 2)
        enc.setBytes(&tVar, length: MemoryLayout<Float>.size,   index: 3)
        enc.setBytes(&kVar, length: MemoryLayout<UInt32>.size,  index: 4)
        enc.setBytes(&pVar, length: MemoryLayout<Float>.size,   index: 5)
        enc.setBytes(&sVar, length: MemoryLayout<UInt64>.size,  index: 6)
        enc.setBytes(&posVar, length: MemoryLayout<UInt32>.size, index: 7)

        let threadsPerGroup = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let gridSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let tgSize   = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        enc.endEncoding()
    }
}
enum SampleTopK64Error: Error {
    case unsupportedVocabulary(Int)
    case scratchAllocationFailed
}

/// Three-stage Top-64 sampler for the documented Gemma 4 policy and measured
/// temperature variants. Intermediate pairs remain in private GPU memory.
final class SampleTopK64 {
    private static let tileSize = 1024
    private static let keptPerTile = 64

    private let ctx: MetalContext
    private let stage1PSO: MTLComputePipelineState
    private let reducePSO: MTLComputePipelineState
    private let finalPSO: MTLComputePipelineState
    private let stage1Values: MTLBuffer
    private let stage1Indices: MTLBuffer
    private let stage2Values: MTLBuffer
    private let stage2Indices: MTLBuffer
    private let stage1Groups: Int
    private let stage2Groups: Int
    private let stage1Count: Int
    private let stage2Count: Int

    public let vocab: Int
    public let scratchBytes: Int

    public init(context: MetalContext, vocab: Int) throws {
        guard vocab > 0, vocab <= 262_144 else {
            throw SampleTopK64Error.unsupportedVocabulary(vocab)
        }
        self.ctx = context
        self.vocab = vocab
        self.stage1PSO = try context.pipeline("sample_topk64_stage1")
        self.reducePSO = try context.pipeline("sample_topk64_reduce")
        self.finalPSO = try context.pipeline("sample_topk64_final")

        self.stage1Groups = (vocab + Self.tileSize - 1) / Self.tileSize
        self.stage1Count = stage1Groups * Self.keptPerTile
        self.stage2Groups = (stage1Count + Self.tileSize - 1) / Self.tileSize
        self.stage2Count = stage2Groups * Self.keptPerTile

        let stage1ValueBytes = stage1Count * MemoryLayout<Float>.stride
        let stage1IndexBytes = stage1Count * MemoryLayout<UInt32>.stride
        let stage2ValueBytes = stage2Count * MemoryLayout<Float>.stride
        let stage2IndexBytes = stage2Count * MemoryLayout<UInt32>.stride
        self.scratchBytes = stage1ValueBytes + stage1IndexBytes
            + stage2ValueBytes + stage2IndexBytes

        guard let stage1Values = context.device.makeBuffer(
                  length: stage1ValueBytes, options: .storageModePrivate),
              let stage1Indices = context.device.makeBuffer(
                  length: stage1IndexBytes, options: .storageModePrivate),
              let stage2Values = context.device.makeBuffer(
                  length: stage2ValueBytes, options: .storageModePrivate),
              let stage2Indices = context.device.makeBuffer(
                  length: stage2IndexBytes, options: .storageModePrivate)
        else {
            throw SampleTopK64Error.scratchAllocationFailed
        }
        self.stage1Values = stage1Values
        self.stage1Indices = stage1Indices
        self.stage2Values = stage2Values
        self.stage2Indices = stage2Indices
    }

    public func encode(commandBuffer: MTLCommandBuffer,
                       probs: MTLBuffer,
                       outToken: MTLBuffer,
                       temperature: Float,
                       topP: Float,
                       seed: UInt64) {
        let threads = MTLSize(width: 256, height: 1, depth: 1)

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(stage1PSO)
            enc.setBuffer(probs, offset: 0, index: 0)
            enc.setBuffer(stage1Values, offset: 0, index: 1)
            enc.setBuffer(stage1Indices, offset: 0, index: 2)
            var v = UInt32(vocab)
            enc.setBytes(&v, length: MemoryLayout<UInt32>.size, index: 3)
            enc.dispatchThreadgroups(MTLSize(width: stage1Groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: threads)
            enc.endEncoding()
        }

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(reducePSO)
            enc.setBuffer(stage1Values, offset: 0, index: 0)
            enc.setBuffer(stage1Indices, offset: 0, index: 1)
            enc.setBuffer(stage2Values, offset: 0, index: 2)
            enc.setBuffer(stage2Indices, offset: 0, index: 3)
            var count = UInt32(stage1Count)
            enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: stage2Groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: threads)
            enc.endEncoding()
        }

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(finalPSO)
            enc.setBuffer(stage2Values, offset: 0, index: 0)
            enc.setBuffer(stage2Indices, offset: 0, index: 1)
            enc.setBuffer(outToken, offset: 0, index: 2)
            var count = UInt32(stage2Count)
            var temp = temperature
            var p = topP
            var rngSeed = seed
            enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&temp, length: MemoryLayout<Float>.size, index: 4)
            enc.setBytes(&p, length: MemoryLayout<Float>.size, index: 5)
            enc.setBytes(&rngSeed, length: MemoryLayout<UInt64>.size, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: threads)
            enc.endEncoding()
        }
    }
}
