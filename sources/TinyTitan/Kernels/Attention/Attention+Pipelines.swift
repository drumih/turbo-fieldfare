import Foundation
import Metal
import TinyTitanCore

// The attention kernel pipelines: the simd partial builder, the specialized
// cache lookup and the partial/combine builders.
//
// Split out of `Attention.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion.
extension Attention {

    func simdPartialPipeline(
        headDim: UInt32, numQHeads: UInt32,
        numKVHeads: UInt32, numChunks: Int
    ) -> MTLComputePipelineState? {
        let key = "\(headDim)/\(numQHeads)/\(numKVHeads)/\(numChunks)"
        if let pso = simdPartialCache[key] { return pso }
        let specializedChunks = numChunks == 16 ? Optional(UInt32(numChunks)) : nil
        guard
            let pso = try? Self.specializedPipeline(
                ctx, "attention_decode_partial_simd",
                headDim: headDim, numQHeads: numQHeads,
                numKVHeads: numKVHeads,
                numChunks: specializedChunks)
        else { return nil }
        simdPartialCache[key] = pso
        return pso
    }

    static func specializedPipeline(
        _ context: MetalContext,
        _ name: String,
        headDim: UInt32,
        numQHeads: UInt32,
        numKVHeads: UInt32,
        numChunks: UInt32? = nil,
        ringCapacity: UInt32? = nil
    ) throws -> MTLComputePipelineState {
        var constants = [
            MetalFunctionConstant(index: 60, value: .uint32(headDim)),
            MetalFunctionConstant(index: 61, value: .uint32(numQHeads)),
            MetalFunctionConstant(index: 62, value: .uint32(numKVHeads)),
            MetalFunctionConstant(index: 63, value: .bool(true)),
        ]
        if let numChunks {
            constants.append(MetalFunctionConstant(index: 65, value: .uint32(numChunks)))
        }
        if let ringCapacity {
            constants.append(MetalFunctionConstant(index: 69, value: .uint32(ringCapacity)))
        }
        return try context.pipeline(name, constants: constants)
    }

    func partialPipeline(
        headDim: UInt32,
        numQHeads: UInt32,
        numKVHeads: UInt32,
        numChunks: Int,
        useGQAPartial: Bool,
        ringCapacity: UInt32 = 0
    ) -> MTLComputePipelineState {
        // `attention_decode_gqa_swa_partial` sizes its threadgroup arrays to
        // `kAttnMaxQPerKV` (2) queries per KV head and *returns without writing
        // its partials* when the span is wider — so the combine pass would
        // normalize the previous layer's values instead of failing. The geometry
        // below already refuses to select this path above that span; stating the
        // limit here, where the kernel is chosen, means a caller that reaches it
        // another way fails loudly rather than quietly.
        if useGQAPartial {
            precondition(
                numKVHeads > 0 && numQHeads / numKVHeads <= 2,
                "attention_decode_gqa_swa_partial needs at most 2 queries "
                    + "per KV head; got \(numQHeads) queries and "
                    + "\(numKVHeads) KV heads")
        }
        if ringCapacity > 0 {
            let name =
                useGQAPartial ? "attention_decode_gqa_swa_partial" : "attention_decode_partial"
            let specializedChunks = numChunks == 16 ? Optional(UInt32(numChunks)) : nil
            do {
                return try Self.specializedPipeline(
                    ctx,
                    name,
                    headDim: headDim,
                    numQHeads: numQHeads,
                    numKVHeads: numKVHeads,
                    numChunks: specializedChunks,
                    ringCapacity: ringCapacity)
            } catch {
                preconditionFailure("failed to build KV ring attention pipeline: \(error)")
            }
        }
        if useGQAPartial && headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            if numChunks == 16 {
                return psoGQAPartialSWAChunks16
            }
            return psoGQAPartialSWA
        }
        if !useGQAPartial && headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            return psoPartialSWA
        }
        if !useGQAPartial && headDim == 512 && numQHeads == 16 && numKVHeads == 2 {
            if numChunks == 16 {
                return psoPartialFullChunks16
            }
            return psoPartialFull
        }
        return useGQAPartial ? psoGQAPartial : psoPartial
    }

    func combinePipeline(
        headDim: UInt32,
        numQHeads: UInt32,
        numKVHeads: UInt32,
        numChunks: Int
    ) -> MTLComputePipelineState {
        if headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            return numChunks == 16 ? psoCombineSWAChunks16 : psoCombineSWA
        }
        if headDim == 512 && numQHeads == 16 && numKVHeads == 2 {
            return numChunks == 16 ? psoCombineFullChunks16 : psoCombineFull
        }
        return psoCombine
    }
}
