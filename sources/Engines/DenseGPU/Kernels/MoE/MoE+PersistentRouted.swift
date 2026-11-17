import Foundation
import Metal
import TinyTitanCore

// The persistent routed-expert encode path: the U16 load stages, the phase-2
// reduce, and the argument-buffer helpers they share.
//
// Split out of `MoE.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion. The three `private`
// helpers widened to internal because the kept encode paths call them.
extension MoE {

    func encodeRoutedPersistentPhase1U16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        xOffset: Int = 0,
        acts: MTLBuffer,
        actsOffset: Int = 0,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) throws {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        var expertCount = topK
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f)
                ? phase1U16SpecializedPSO
                : phase1U16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: xOffset, index: 2)
        encoder.setBuffer(acts, offset: actsOffset, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(
            ioStatus ?? alwaysReadyIOStatus,
            offset: ioStatus == nil ? 0 : ioStatusOffset,
            index: 7)
        // Phase-1 uses 16 rows per threadgroup (threadgroup-staged x), so the
        // dispatch is (topK*f)/16 groups of 512 threads.
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 15) / 16, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase1SubsetU16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        activeSlots: MTLBuffer,
        activeSlotIndices: [UInt32],
        activeCount: UInt32,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) throws {
        guard activeCount > 0 else { return }
        validate(routedBlobs: routedBlobs, topK: topK)
        precondition(activeSlotIndices.count == Int(activeCount))
        var dimension = d
        var intermediate = f
        var expertCount = topK
        var active = activeCount
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f)
                ? phase1SubsetU16SpecializedPSO
                : phase1SubsetU16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for slot in activeSlotIndices {
            encoder.useResource(routedBlobs[Int(slot)], usage: .read)
        }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(activeSlots, offset: 0, index: 7)
        encoder.setBytes(&active, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBuffer(
            ioStatus ?? alwaysReadyIOStatus,
            offset: ioStatus == nil ? 0 : ioStatusOffset,
            index: 9)
        // Phase-1 uses 16 rows per threadgroup (threadgroup-staged x).
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(activeCount * f) + 15) / 16, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase2Reduce(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        acts: MTLBuffer,
        actsOffset: Int = 0,
        routingWeights: MTLBuffer,
        routingWeightsOffset: Int = 0,
        residual: MTLBuffer,
        residualOffset: Int = 0,
        y: MTLBuffer,
        yOffset: Int = 0,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) throws {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(
            maxStreamedExperts == 8
                ? (useRealDecodeConstants(d: d, f: f)
                    ? phase2ReduceK8SpecializedPSO
                    : phase2ReduceK8PSO)
                : phase2ReduceKNPSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: actsOffset, index: 2)
        encoder.setBuffer(routingWeights, offset: routingWeightsOffset, index: 3)
        encoder.setBuffer(residual, offset: residualOffset, index: 4)
        encoder.setBuffer(y, offset: yOffset, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBuffer(
            ioStatus ?? alwaysReadyIOStatus,
            offset: ioStatus == nil ? 0 : ioStatusOffset,
            index: 8)
        // One simdgroup per expert slot. The kn kernel has no sg >= k guard
        // precisely because the launch width says k, so this must stay in
        // step with it: 32 lanes x k.
        if maxStreamedExperts != 8 {
            var k = UInt32(maxStreamedExperts)
            encoder.setBytes(&k, length: MemoryLayout<UInt32>.stride, index: 9)
        }
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * maxStreamedExperts,
                height: 1, depth: 1))
        encoder.endEncoding()
    }

    func validate(routedBlobs: [MTLBuffer], topK: UInt32) {
        precondition(topK == UInt32(maxStreamedExperts))
        precondition(routedBlobs.count == Int(topK))
    }

    func encodeRoutedArgumentBuffer(
        _ buffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedBufferOffsets: [Int]?
    ) {
        precondition(
            routedBufferOffsets == nil
                || routedBufferOffsets?.count == routedBlobs.count)
        routedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            routedArgEncoder.setBuffer(
                blob,
                offset: routedBufferOffsets?[index] ?? 0,
                index: index)
        }
    }

    func useRealDecodeConstants(d: UInt32, f: UInt32) -> Bool {
        d == realDecodeD && f == realDecodeF
    }
}
