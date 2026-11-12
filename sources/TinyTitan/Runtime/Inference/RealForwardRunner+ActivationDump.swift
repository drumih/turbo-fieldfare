import Foundation
import Metal

/// Writes named activation slices to disk so a run can be compared against a
/// reference implementation stage by stage.
///
/// This exists because unit tests cannot tell a correct architecture from a
/// plausible one. Every mistake worth worrying about in a new family --- a
/// transposed gate, a norm applied to the wrong axis, a convolution tap read
/// one position off --- produces smooth, confident, wrong output. Dumping the
/// intermediates and diffing them against a reference is the only check that
/// localizes such a bug to a single stage.
///
/// A dump recorded during encoding and performed after the layer is awaited.
struct PendingActivationDump {
    let name: String
    let buffer: MTLBuffer
    let count: Int
    let position: Int
    var offset: Int = 0
}

/// Off unless `TINYTITAN_ACT_DUMP` names a directory, and every call site is
/// wrapped so a normal run does no work and takes no synchronization.
extension RealForwardRunner {
    /// Highest layer index the detailed per-layer dumps cover. Layers 0-3 span
    /// one full period of this family's attention schedule: three linear
    /// layers and the first full-attention one. The residual at every layer
    /// entry is dumped regardless -- it is 20 KB a layer and it is what
    /// locates the first stage where a run leaves the reference.
    /// `TINYTITAN_ACT_DUMP_LAYER_LIMIT` raises it when a divergence starts
    /// deeper in the stack.
    var dumpLayerLimit: Int { Self.dumpLayerLimitSetting }

    static let dumpLayerLimitSetting: Int =
        ProcessInfo.processInfo.environment["TINYTITAN_ACT_DUMP_LAYER_LIMIT"]
        .flatMap { Int($0) } ?? 3

    /// Positions to dump, from 0. One is enough to check the per-layer math;
    /// more is what catches state threaded between tokens -- the KV cache,
    /// the delta-rule state, and the two convolution histories -- which a
    /// single position cannot exercise at all.
    static let dumpPositionCount =
        ProcessInfo.processInfo
        .environment["TINYTITAN_ACT_DUMP_POSITIONS"].flatMap(Int.init) ?? 1

    func activationDumpActive(position: Int) -> Bool {
        activationDumpDirectory != nil
            && position >= activationDumpFirstPosition
            && position < activationDumpFirstPosition + activationDumpPositionCount
    }

    /// Each position writes into its own subdirectory, so a later token does
    /// not quietly overwrite the one being compared.
    func activationDumpDirectory(position: Int) -> URL? {
        guard let root = activationDumpDirectory else { return nil }
        let directory = root.appendingPathComponent("pos\(position)")
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }

    /// Records the token whose forward pass the dumps describe.
    func dumpActivationToken(_ token: Int32, position: Int) {
        guard let directory = activationDumpDirectory(position: position)
        else { return }
        try? "\(token)\n".write(
            to: directory.appendingPathComponent("token.txt"),
            atomically: true, encoding: .utf8)
    }

    /// Dumps a **private** buffer by blitting it into shared staging first.
    ///
    /// The prefill scratch is `.storageModePrivate` (decode's is shared, which
    /// is why the decode call sites can read theirs directly). Reading a
    /// private resource from the CPU is what Metal's validation aborts on --
    /// `resourceOptions (0x20) specify MTLResourceStorageModePrivate, which is
    /// not CPU accessible` -- so a prefill dump goes through a blit, and the
    /// command buffer is awaited before the bytes are read.
    func dumpActivationPrivate(
        _ name: String,
        _ buffer: MTLBuffer,
        count: Int,
        position: Int,
        offset: Int = 0
    ) {
        guard activationDumpDirectory(position: position) != nil else { return }
        let bytes = count * MemoryLayout<Float16>.stride
        guard offset + bytes <= buffer.length,
            let staging = ctx.device.makeBuffer(
                length: bytes,
                options: .storageModeShared),
            let cb = ctx.queue.makeCommandBuffer(),
            let blit = cb.makeBlitCommandEncoder()
        else { return }
        blit.copy(
            from: buffer, sourceOffset: offset,
            to: staging, destinationOffset: 0, size: bytes)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        dumpActivation(name, staging, count: count, position: position)
    }

    /// Records a dump to perform after the buffer's writer has been committed.
    ///
    /// A dump taken *between* two encode calls reads a buffer the GPU has not
    /// been asked to write yet: the encode is still accumulating in an
    /// uncommitted command buffer, so a staging blit submitted now copies
    /// whatever the buffer held before -- which for scratch first used on this
    /// layer is zeros. That reads exactly like a stage computing nothing.
    /// Deferring to the end of the layer, after the wait, is what makes a
    /// mid-layer dump mean what it says.
    func dumpActivationDeferred(
        _ name: String,
        _ buffer: MTLBuffer,
        count: Int,
        position: Int,
        offset: Int = 0
    ) {
        guard activationDumpDirectory(position: position) != nil else { return }
        pendingDumps.append(
            PendingActivationDump(
                name: name, buffer: buffer,
                count: count, position: position,
                offset: offset))
    }

    /// Performs every deferred dump. Call once the layer's work is awaited.
    func flushDeferredDumps() {
        guard !pendingDumps.isEmpty else { return }
        let pending = pendingDumps
        pendingDumps.removeAll(keepingCapacity: true)
        for dump in pending {
            dumpActivationPrivate(
                dump.name, dump.buffer, count: dump.count,
                position: dump.position, offset: dump.offset)
        }
    }

    /// Writes `count` fp16 values as raw little-endian fp16 (numpy
    /// `dtype='<f2'`). Synchronizes first: the caller passes the command
    /// buffer that produced the values, because reading a shared buffer the
    /// GPU is still writing would dump a mixture of two states and look like
    /// a numerical bug.
    func dumpActivation(
        _ name: String,
        _ buffer: MTLBuffer,
        count: Int,
        position: Int,
        offset: Int = 0,
        after commandBuffer: MTLCommandBuffer? = nil
    ) {
        guard let directory = activationDumpDirectory(position: position)
        else { return }
        commandBuffer?.waitUntilCompleted()
        let bytes = count * MemoryLayout<Float16>.stride
        guard offset + bytes <= buffer.length else {
            // Reported rather than skipped in silence: a dump that quietly
            // writes nothing looks exactly like a stage computing zeros, which
            // is the confusion this facility exists to remove.
            FileHandle.standardError.write(
                Data(
                    ("TINYTITAN_ACT_DUMP: \(name) needs \(offset + bytes) bytes but the buffer "
                        + "holds \(buffer.length); nothing written\n").utf8))
            return
        }
        let data = Data(
            bytes: buffer.contents().advanced(by: offset),
            count: bytes)
        try? data.write(to: directory.appendingPathComponent("\(name).f16"))
    }
}
