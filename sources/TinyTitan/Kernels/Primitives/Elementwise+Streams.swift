import Foundation
import Metal
import TinyTitanCore

// The hyper-connection and PLE elementwise passes.
//
// Split out of `Elementwise.swift` (2026-10-25) under the 500-line rule as pure
// code motion. The pipelines stay owned by `Elementwise` (its initializer builds
// every one of them); the extension only encodes with them.
extension Elementwise {
    /// The hyper-connection gated read: out[d] = mean over streams of
    /// mix[s,d] * normed[s,d]. Collapses the S-wide residual to the single
    /// D-wide vector the block consumes.
    func encodeHCMixReduce(
        commandBuffer: MTLCommandBuffer,
        mix: MTLBuffer, mixOffset: Int = 0,
        normed: MTLBuffer, normedOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: Int, streams: Int, tokens: Int = 1,
        inScale: Float
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(hcMixReducePSO)
        enc.setBuffer(mix, offset: mixOffset, index: 0)
        enc.setBuffer(normed, offset: normedOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var d = UInt32(dim)
        var s = UInt32(streams)
        var t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 5)
        var gs = inScale
        enc.setBytes(&gs, length: MemoryLayout<Float>.size, index: 6)
        let w = min(hcMixReducePSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(
            MTLSize(width: dim * tokens, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// The hyper-connection gated write: streams[s,d] += blockOut[d] *
    /// inject[s], in place.
    func encodeHCInject(
        commandBuffer: MTLCommandBuffer,
        streams: MTLBuffer, streamsOffset: Int = 0,
        blockOut: MTLBuffer, blockOutOffset: Int = 0,
        inject: MTLBuffer, injectOffset: Int = 0,
        dim: Int, streamCount: Int, tokens: Int = 1,
        inScale: Float
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(hcInjectPSO)
        enc.setBuffer(streams, offset: streamsOffset, index: 0)
        enc.setBuffer(blockOut, offset: blockOutOffset, index: 1)
        enc.setBuffer(inject, offset: injectOffset, index: 2)
        var d = UInt32(dim)
        var s = UInt32(streamCount)
        var t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 5)
        var gs = inScale
        enc.setBytes(&gs, length: MemoryLayout<Float>.size, index: 6)
        let total = dim * streamCount * tokens
        let w = min(hcInjectPSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Per-stream score: key . query / sqrt(D), one scalar per stream.
    func encodePLEStreamScore(
        commandBuffer: MTLCommandBuffer,
        key: MTLBuffer, keyOffset: Int = 0,
        query: MTLBuffer, queryOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: Int, streams: Int, tokens: Int = 1
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pleScorePSO)
        enc.setBuffer(key, offset: keyOffset, index: 0)
        enc.setBuffer(query, offset: queryOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var d = UInt32(dim)
        var s = UInt32(streams)
        var t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 5)
        enc.dispatchThreads(
            MTLSize(width: streams * tokens, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(pleScorePSO.maxTotalThreadsPerThreadgroup, 32),
                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// gate = sigmoid(signed sqrt of the score), with a magnitude floor.
    func encodePLESignedSqrtGate(
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        count: Int, floor: Float = 1e-6
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pleGatePSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var n = UInt32(count)
        var f = floor
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&f, length: MemoryLayout<Float>.size, index: 3)
        enc.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(pleGatePSO.maxTotalThreadsPerThreadgroup, 32),
                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// out[s,d] = value[d] * gate[s].
    func encodePLEBroadcastScale(
        commandBuffer: MTLCommandBuffer,
        value: MTLBuffer, valueOffset: Int = 0,
        gate: MTLBuffer, gateOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: Int, streams: Int, tokens: Int = 1
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pleBroadcastPSO)
        enc.setBuffer(value, offset: valueOffset, index: 0)
        enc.setBuffer(gate, offset: gateOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var d = UInt32(dim)
        var s = UInt32(streams)
        var t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 5)
        let total = dim * streams * tokens
        enc.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(pleBroadcastPSO.maxTotalThreadsPerThreadgroup, 256),
                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Dilated depthwise causal convolution with silu, over `history + T` rows
    /// of carried state and chunk.
    func encodePLEDilatedConv(
        commandBuffer: MTLCommandBuffer,
        xpad: MTLBuffer, xpadOffset: Int = 0,
        weight: MTLBuffer, weightOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        channels: Int, tokens: Int,
        kernelSize: Int, dilation: Int
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pleConvPSO)
        enc.setBuffer(xpad, offset: xpadOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var c = UInt32(channels)
        var t = UInt32(tokens)
        var k = UInt32(kernelSize)
        var dil = UInt32(dilation)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&dil, length: MemoryLayout<UInt32>.size, index: 6)
        let total = channels * tokens
        enc.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(pleConvPSO.maxTotalThreadsPerThreadgroup, 256),
                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Collapse `[tokens, streams, dim]` to `[tokens, dim]` by a plain mean.
    func encodeHCMean(
        commandBuffer: MTLCommandBuffer,
        streams: MTLBuffer, streamsOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        dim: Int, streamCount: Int, tokens: Int
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(hcMeanPSO)
        enc.setBuffer(streams, offset: streamsOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var d = UInt32(dim)
        var s = UInt32(streamCount)
        var t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 4)
        let w = min(hcMeanPSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(
            MTLSize(width: dim * tokens, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Widen a `[tokens, dim]` block into the `[tokens, streams, dim]`
    /// residual. Distinct from the in-place broadcast: a chunk's embeddings
    /// are contiguous rows, and expanding them in place would overwrite rows
    /// that have not been read yet.
    func encodeHCExpand(
        commandBuffer: MTLCommandBuffer,
        source: MTLBuffer, sourceOffset: Int = 0,
        destination: MTLBuffer, destinationOffset: Int = 0,
        dim: Int, streamCount: Int, tokens: Int
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(hcExpandPSO)
        enc.setBuffer(source, offset: sourceOffset, index: 0)
        enc.setBuffer(destination, offset: destinationOffset, index: 1)
        var d = UInt32(dim)
        var s = UInt32(streamCount)
        var t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 4)
        let total = dim * streamCount * tokens
        let w = min(hcExpandPSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Replicate stream 0 across the remaining residual streams, in place.
    func encodeHCBroadcast(
        commandBuffer: MTLCommandBuffer,
        streams: MTLBuffer, streamsOffset: Int = 0,
        dim: Int, streamCount: Int
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(hcBroadcastPSO)
        enc.setBuffer(streams, offset: streamsOffset, index: 0)
        var d = UInt32(dim)
        var s = UInt32(streamCount)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 1)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 2)
        enc.dispatchThreads(
            MTLSize(width: dim * streamCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(hcBroadcastPSO.maxTotalThreadsPerThreadgroup, 256),
                height: 1, depth: 1))
        enc.endEncoding()
    }
}
