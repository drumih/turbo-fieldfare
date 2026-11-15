import Foundation
import Metal

// Auxiliary-state snapshot/restore for the QSA indexer.
//
// Split out of `QSAIndexer.swift` (2026-10-22) under the 500-line-per-file
// rule as pure code motion; the methods are an extension because they read the
// class's per-layer key buffers.
extension QSAIndexer {
    // MARK: - Auxiliary state (Qwen3.8 interior checkpoints)

    /// Append the state a restored prefix needs: every full pooled block and
    /// the raw keys of the last, possibly partial, block.
    ///
    /// Raw keys of earlier blocks are not needed: a block is pooled once, when
    /// its last member arrives, and never read raw again. Keeping only the
    /// tail makes the payload about `position / compressRatio` rows rather
    /// than `position`, so a 100K prefix costs a few tens of MiB per layer
    /// instead of the QSA key cache's full size.
    func appendSnapshotPayload(to data: inout Data, position: Int) throws {
        guard position > 0, position <= capacity else {
            throw InferenceAuxiliaryStateError.malformed(
                "QSA position \(position) outside 1...\(capacity)")
        }
        let layers = pooled.keys.sorted()
        AuxiliaryCoding.appendUInt32(layers.count, to: &data)
        let half = MemoryLayout<Float16>.stride
        let blocks = (position + compressRatio - 1) / compressRatio
        let tail = position - (blocks - 1) * compressRatio
        for layer in layers {
            guard let pooledBuffer = pooled[layer],
                let rawBuffer = rawKeys[layer]
            else {
                throw InferenceAuxiliaryStateError.malformed(
                    "QSA layer \(layer) has only one of raw/pooled buffers")
            }
            let pooledBytes = blocks * headDim * half
            let rawTailOffset = (blocks - 1) * compressRatio * headDim * half
            let rawTailBytes = tail * headDim * half
            guard pooledBuffer.length >= pooledBytes,
                rawBuffer.length >= rawTailOffset + rawTailBytes
            else {
                throw InferenceAuxiliaryStateError.malformed(
                    "QSA layer \(layer) buffers are shorter than the prefix")
            }
            AuxiliaryCoding.appendUInt32(layer, to: &data)
            AuxiliaryCoding.appendUInt32(blocks, to: &data)
            AuxiliaryCoding.appendUInt32(tail, to: &data)
            AuxiliaryCoding.appendBytes(
                pooledBuffer.contents(), count: pooledBytes, to: &data)
            AuxiliaryCoding.appendBytes(
                rawBuffer.contents().advanced(by: rawTailOffset),
                count: rawTailBytes, to: &data)
        }
    }

    /// Restore what `appendSnapshotPayload` wrote. The caller has already
    /// restored the KV cursor to `position`.
    func restoreSnapshotPayload(
        _ bytes: UnsafeRawBufferPointer,
        offset: inout Int,
        position: Int
    ) throws {
        guard position > 0 else {
            throw InferenceAuxiliaryStateError.malformed(
                "QSA restore needs a positive position")
        }
        let half = MemoryLayout<Float16>.stride
        guard let base = bytes.baseAddress else {
            throw InferenceAuxiliaryStateError.truncated("QSA payload")
        }
        let expectedBlocks = (position + compressRatio - 1) / compressRatio
        let expectedTail = position - (expectedBlocks - 1) * compressRatio
        let layers = try AuxiliaryCoding.readUInt32(bytes, offset: &offset)
        for _ in 0..<layers {
            let layer = try AuxiliaryCoding.readUInt32(bytes, offset: &offset)
            let blocks = try AuxiliaryCoding.readUInt32(bytes, offset: &offset)
            let tail = try AuxiliaryCoding.readUInt32(bytes, offset: &offset)
            guard blocks == expectedBlocks, tail == expectedTail else {
                throw InferenceAuxiliaryStateError.malformed(
                    "QSA layer \(layer) covers \(blocks)/\(tail) blocks/tail, "
                        + "expected \(expectedBlocks)/\(expectedTail)")
            }
            let pooledBytes = blocks * headDim * half
            let rawTailBytes = tail * headDim * half
            try AuxiliaryCoding.require(
                bytes, offset: offset, count: pooledBytes + rawTailBytes,
                "QSA layer \(layer) payload")
            let buffers = try layerBuffers(layer, hold: position)
            let rawTailBytesNeeded =
                (expectedBlocks - 1) * compressRatio * headDim * half
                + rawTailBytes
            guard buffers.pooled.length >= pooledBytes,
                buffers.raw.length >= rawTailBytesNeeded
            else {
                throw InferenceAuxiliaryStateError.malformed(
                    "QSA layer \(layer) live buffers are too small")
            }
            memcpy(
                buffers.pooled.contents(),
                base.advanced(by: offset),
                pooledBytes)
            offset += pooledBytes
            let rawTailOffset =
                (expectedBlocks - 1) * compressRatio * headDim * half
            memcpy(
                buffers.raw.contents().advanced(by: rawTailOffset),
                base.advanced(by: offset),
                rawTailBytes)
            offset += rawTailBytes
        }
    }
}
