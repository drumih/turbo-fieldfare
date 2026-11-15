import Foundation
import Metal
import TinyTitanCore

/// Per-batch scratch for the fused decode path (T4).
///
/// Every buffer is batch-private: two rows never share a slot in one layer.
/// The route ids/weights are shared-memory because the router readback is what
/// turns them into a union expert plan on the CPU; the rest stay private so a
/// queued command buffer owns its contents until the GPU reaches it.
struct BatchedDecodeScratch {
    let rows: Int
    let hidden: MTLBuffer
    let normed: MTLBuffer
    let q: MTLBuffer
    let k: MTLBuffer
    let v: MTLBuffer
    let attn: MTLBuffer
    let o: MTLBuffer
    let routedX: MTLBuffer
    let h1: MTLBuffer
    let h2: MTLBuffer
    let routeIDs: MTLBuffer
    let routeWeights: MTLBuffer
    let nextIDs: MTLBuffer
    let nextWeights: MTLBuffer

    static func allocate(
        device: MTLDevice,
        config: ArchConfig,
        rows: Int
    ) throws -> BatchedDecodeScratch {
        guard rows == 2 || rows == 4 else {
            throw ModelError.unsupportedArchitecture(
                detail: "fused decode is defined for width 2 and 4, not \(rows)")
        }
        let d = config.hiddenSize
        let half = MemoryLayout<Float16>.stride
        let headDim = config.fullHeadDim
        let qDim = config.numHeads * headDim
        let qRows = config.attnOutputGate ? 2 * qDim : qDim
        let kvDim = config.numFullKVHeads * headDim
        let topK = config.topKExperts

        func privateBuffer(_ elements: Int, label: String) throws -> MTLBuffer {
            guard
                let buffer = device.makeBuffer(
                    length: max(elements, 1) * half,
                    options: .storageModePrivate)
            else { throw ModelError.residentBufferWrapFailed }
            buffer.label = label
            return buffer
        }
        func sharedBuffer(_ bytes: Int, label: String) throws -> MTLBuffer {
            guard
                let buffer = device.makeBuffer(
                    length: max(bytes, 1),
                    options: .storageModeShared)
            else { throw ModelError.residentBufferWrapFailed }
            buffer.label = label
            return buffer
        }

        return BatchedDecodeScratch(
            rows: rows,
            hidden: try privateBuffer(rows * d, label: "batch.hidden"),
            normed: try privateBuffer(rows * d, label: "batch.normed"),
            q: try privateBuffer(rows * qRows, label: "batch.q"),
            k: try privateBuffer(rows * kvDim, label: "batch.k"),
            v: try privateBuffer(rows * kvDim, label: "batch.v"),
            attn: try privateBuffer(rows * qDim, label: "batch.attn"),
            o: try privateBuffer(rows * d, label: "batch.o"),
            routedX: try privateBuffer(rows * d, label: "batch.routedX"),
            h1: try privateBuffer(rows * d, label: "batch.h1"),
            h2: try privateBuffer(rows * d, label: "batch.h2"),
            routeIDs: try sharedBuffer(rows * topK * 4, label: "batch.routeIDs"),
            routeWeights: try sharedBuffer(rows * topK * half, label: "batch.routeWeights"),
            nextIDs: try sharedBuffer(rows * topK * 4, label: "batch.nextIDs"),
            nextWeights: try sharedBuffer(rows * topK * half, label: "batch.nextWeights"))
    }
}

extension RealForwardRunner {
    /// Returns the cached batch scratch, allocating it once per width.
    func ensureBatchedDecodeScratch(rows: Int) throws -> BatchedDecodeScratch {
        if let existing = batchedDecodeScratch, existing.rows == rows {
            return existing
        }
        let made = try BatchedDecodeScratch.allocate(
            device: ctx.device, config: cfg, rows: rows)
        batchedDecodeScratch = made
        return made
    }
}
