import Foundation
import Metal
import TinyTitanCore

/// Prefill KV writes: the contiguous and strided cache copies and the
/// quantized staging path.
///
/// Split out of `RealForwardRunner+Prefill.swift` (2026-09-28) under the
/// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
extension RealForwardRunner {
    func copyPrefillKV(
        commandBuffer: MTLCommandBuffer,
        source: MTLBuffer,
        destination: (buffer: MTLBuffer, offset: Int, stride: Int),
        sourceTokenOffset: Int,
        tokenCount: Int,
        bytesPerToken: Int
    ) throws {
        guard tokenCount > 0 else { return }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(
            from: source,
            sourceOffset: sourceTokenOffset * bytesPerToken,
            to: destination.buffer,
            destinationOffset: destination.offset,
            size: tokenCount * bytesPerToken)
        blit.endEncoding()
    }

    func copyPrefillKVToCache(
        commandBuffer: MTLCommandBuffer,
        kv: KVCacheManager,
        layer: Int,
        startPosition: Int,
        tokenCount: Int,
        slot: Int = 0,
        keySource: MTLBuffer,
        valueSource: MTLBuffer,
        bytesPerToken: Int
    ) throws {
        kv.beginPrefillWrite(
            layer: layer, slot: slot,
            startPosition: startPosition, tokenCount: tokenCount)
        if kv.precision.isQuantized {
            guard let kvQuantizer else {
                throw ModelError.internalInconsistency(
                    detail: "quantized KV cache has no quantizer")
            }
            let elements = bytesPerToken / MemoryLayout<Float16>.stride
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstSpan = min(tokenCount, capacity - physicalStart)
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: keySource,
                sourceTokenStrideElements: elements,
                destination: kv.keyRangeView(
                    layer: layer, start: startPosition,
                    count: firstSpan, slot: slot),
                tokenCount: firstSpan,
                elementCount: elements)
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: valueSource,
                sourceTokenStrideElements: elements,
                destination: kv.valueRangeView(
                    layer: layer, start: startPosition,
                    count: firstSpan, slot: slot),
                tokenCount: firstSpan,
                elementCount: elements)
            guard firstSpan < tokenCount else { return }
            let secondCount = tokenCount - firstSpan
            let secondStart = startPosition + firstSpan
            let sourceOffset = firstSpan * bytesPerToken
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: keySource,
                sourceOffset: sourceOffset,
                sourceTokenStrideElements: elements,
                destination: kv.keyRangeView(
                    layer: layer, start: secondStart,
                    count: secondCount, slot: slot),
                tokenCount: secondCount,
                elementCount: elements)
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: valueSource,
                sourceOffset: sourceOffset,
                sourceTokenStrideElements: elements,
                destination: kv.valueRangeView(
                    layer: layer, start: secondStart,
                    count: secondCount, slot: slot),
                tokenCount: secondCount,
                elementCount: elements)
            return
        }
        let capacity = kv.capacity(layer: layer)
        let physicalStart = startPosition % capacity
        let firstSpan = min(tokenCount, capacity - physicalStart)
        let keyFirst = kv.kRange(
            layer: layer, start: startPosition, count: firstSpan,
            slot: slot)
        let valueFirst = kv.vRange(
            layer: layer, start: startPosition, count: firstSpan,
            slot: slot)
        try copyPrefillKV(
            commandBuffer: commandBuffer,
            source: keySource,
            destination: keyFirst,
            sourceTokenOffset: 0,
            tokenCount: firstSpan,
            bytesPerToken: bytesPerToken)
        try copyPrefillKV(
            commandBuffer: commandBuffer,
            source: valueSource,
            destination: valueFirst,
            sourceTokenOffset: 0,
            tokenCount: firstSpan,
            bytesPerToken: bytesPerToken)
        guard firstSpan < tokenCount else { return }

        let secondCount = tokenCount - firstSpan
        let secondStart = startPosition + firstSpan
        let keySecond = kv.kRange(
            layer: layer, start: secondStart, count: secondCount,
            slot: slot)
        let valueSecond = kv.vRange(
            layer: layer, start: secondStart, count: secondCount,
            slot: slot)
        try copyPrefillKV(
            commandBuffer: commandBuffer,
            source: keySource,
            destination: keySecond,
            sourceTokenOffset: firstSpan,
            tokenCount: secondCount,
            bytesPerToken: bytesPerToken)
        try copyPrefillKV(
            commandBuffer: commandBuffer,
            source: valueSource,
            destination: valueSecond,
            sourceTokenOffset: firstSpan,
            tokenCount: secondCount,
            bytesPerToken: bytesPerToken)
    }

    func encodeQuantizedKV(
        commandBuffer: MTLCommandBuffer,
        kv: KVCacheManager,
        layer: Int,
        position: Int,
        slot: Int = 0,
        keySource: MTLBuffer,
        valueSource: MTLBuffer,
        elementCount: Int
    ) throws {
        guard let kvQuantizer else {
            throw ModelError.internalInconsistency(
                detail: "quantized KV cache has no quantizer")
        }
        try kvQuantizer.encode(
            commandBuffer: commandBuffer,
            source: keySource,
            sourceTokenStrideElements: elementCount,
            destination: kv.keyRangeView(
                layer: layer, start: position, count: 1,
                slot: slot),
            tokenCount: 1,
            elementCount: elementCount)
        try kvQuantizer.encode(
            commandBuffer: commandBuffer,
            source: valueSource,
            sourceTokenStrideElements: elementCount,
            destination: kv.valueRangeView(
                layer: layer, start: position, count: 1,
                slot: slot),
            tokenCount: 1,
            elementCount: elementCount)
    }
}
