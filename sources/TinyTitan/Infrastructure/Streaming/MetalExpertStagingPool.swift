import Foundation
import Metal
import TinyTitanCore

/// Fixed, model-wide staging buffers for the event-driven Metal I/O path.
///
/// A compute command buffer must not reference an expert-cache slot while an
/// `MTLIOCommandBuffer` is still loading that same resource. Metal reserves
/// resource hazards when commands are submitted, even when the first encoder
/// in the compute buffer waits on a shared event. Keeping I/O destinations
/// separate from cache destinations breaks that cycle:
///
/// `MTLIO -> staging -> shared-event wait -> GPU blit -> cache slot -> MoE`.
///
/// The pool grants one lease at a time. Decode has one outstanding routed
/// layer, so that is sufficient and, importantly, makes native I/O event
/// values strictly ordered. It also fixes the additional Metal-I/O footprint
/// at `slotCapacity * expertStride` rather than allocating per request or per
/// transformer layer.
/// unchecked-invariant: allocation is immutable and lease state is serialized by `lock`.
public final class MetalExpertStagingPool: @unchecked Sendable {
    private let lock = NSLock()
    private let buffers: [MTLBuffer]
    private var leased = false

    init(device: MTLDevice, byteCount: Int, slotCapacity: Int) throws {
        precondition(byteCount > 0)
        precondition(slotCapacity > 0)
        var allocated: [MTLBuffer] = []
        allocated.reserveCapacity(slotCapacity)
        for _ in 0..<slotCapacity {
            guard
                let buffer = device.makeBuffer(
                    length: byteCount,
                    options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = "TinyTitan Metal I/O staging"
            allocated.append(buffer)
        }
        self.buffers = allocated
    }

    /// Returns nil rather than blocking a decode task when another generation
    /// owns the bounded staging ring. The caller may use the host-synchronised
    /// Metal fallback in that unusual concurrent case.
    func tryAcquire(count: Int) -> MetalExpertStagingLease? {
        guard count > 0, count <= buffers.count else { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard !leased else { return nil }
        leased = true
        return MetalExpertStagingLease(
            pool: self,
            buffers: Array(buffers.prefix(count)))
    }

    fileprivate func release() {
        lock.lock()
        precondition(leased, "Metal I/O staging lease released twice")
        leased = false
        lock.unlock()
    }
}

/// Keeps staging buffers alive until the dependent GPU copy has completed.
/// unchecked-invariant: immutable buffers remain valid while `release` is serialized by `lock`.
final class MetalExpertStagingLease: @unchecked Sendable {
    let buffers: [MTLBuffer]
    private let pool: MetalExpertStagingPool
    private let lock = NSLock()
    private var released = false

    fileprivate init(pool: MetalExpertStagingPool, buffers: [MTLBuffer]) {
        self.pool = pool
        self.buffers = buffers
    }

    func release() {
        lock.lock()
        guard !released else {
            lock.unlock()
            return
        }
        released = true
        lock.unlock()
        pool.release()
    }
}

/// The explicit GPU-side transfer from Metal-I/O staging to reserved expert
/// cache slots. It intentionally owns its lease; a command buffer cannot
/// observe a staging buffer being reused until `release()` after completion.
/// unchecked-invariant: immutable transfer resources are released only after their command completes.
final class MetalExpertStagingTransfer: @unchecked Sendable {
    private let lease: MetalExpertStagingLease
    private let destinations: [MTLBuffer]
    private let destinationOffsets: [Int]
    private let byteCount: Int

    init(
        lease: MetalExpertStagingLease,
        destinations: [MTLBuffer],
        destinationOffsets: [Int],
        byteCount: Int
    ) {
        precondition(lease.buffers.count == destinations.count)
        precondition(destinations.count == destinationOffsets.count)
        precondition(byteCount > 0)
        self.lease = lease
        self.destinations = destinations
        self.destinationOffsets = destinationOffsets
        self.byteCount = byteCount
    }

    func encodeCopy(commandBuffer: MTLCommandBuffer) throws {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        for index in destinations.indices {
            let destinationOffset = destinationOffsets[index]
            guard destinationOffset >= 0,
                destinations[index].length - destinationOffset >= byteCount,
                lease.buffers[index].length >= byteCount
            else {
                blit.endEncoding()
                throw ModelError.internalInconsistency(
                    detail: "Metal I/O staging transfer range is out of bounds")
            }
            blit.copy(
                from: lease.buffers[index], sourceOffset: 0,
                to: destinations[index], destinationOffset: destinationOffset,
                size: byteCount)
        }
        blit.endEncoding()
    }

    func release() { lease.release() }
}
