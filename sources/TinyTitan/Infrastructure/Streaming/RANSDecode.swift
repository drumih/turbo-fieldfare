import Foundation
import Metal
import Synchronization
import TinyTitanFormat
import TinyTitanCore

// GPU decode of a rANS expert layer.
//
// The payload is not byte-addressable by expert offset, so this path replaces a
// raw expert pread with: read the expert's contiguous payload run, then run the
// rANS kernel into the slot. The kernel is split by phase -- gate|up blocks
// decode before the engine publishes phase 1, down blocks decode while phase 1
// runs -- because that split is what keeps the decode inside the layer's window
// instead of in front of it.

struct RANSStreamPlan {
    static let descriptorWords = 9
    var payloadOffset: UInt32
    var tableOffset: UInt32
    var outputOffset: UInt32
    var stride: UInt32
    var symbolCount: UInt32
    var streamIndex: UInt32
    var streamCount: UInt32
    var slotOffset: UInt32
    var payloadEnd: UInt32
    let phase: Int
}

/// The per-expert stream plans built by the parallel block loop at layer open.
///
/// unchecked-invariant: every write to `plans`, `payloadSizes` or `failures`
/// happens while holding `lock`, and both are read only after
/// `concurrentPerform` returns, so the unchecked conformance publishes slices
/// that are complete.
private final class RANSPlanAccumulator: @unchecked Sendable {
    let lock = NSLock()
    var plans: [[RANSStreamPlan]]
    var payloadSizes: [Int]
    var failures: [String] = []

    init(expertCount: Int) {
        plans = Array(repeating: [], count: expertCount)
        payloadSizes = Array(repeating: 0, count: expertCount)
    }
}

/// The per-device pipeline and queue, compiled and created once.
///
/// unchecked-invariant: `pipelines` and `queues` are the only mutable state and
/// every read or write of either happens while holding `lock`; the class holds no
/// other stored state, so the unchecked conformance cannot be observed unsafely.
private final class RANSGPU: @unchecked Sendable {
    static let shared = RANSGPU()
    private let lock = NSLock()
    private var pipelines: [ObjectIdentifier: MTLComputePipelineState] = [:]
    private var queues: [ObjectIdentifier: MTLCommandQueue] = [:]

    func pipeline(for device: MTLDevice) throws -> MTLComputePipelineState {
        lock.lock()
        defer { lock.unlock() }
        let key = ObjectIdentifier(device)
        if let pipeline = pipelines[key] { return pipeline }
        let source = """
            #include <metal_stdlib>
            using namespace metal;
            struct StreamDesc {
                uint payloadOffset;
                uint tableOffset;
                uint outputOffset;
                uint stride;
                uint symbolCount;
                uint streamIndex;
                uint streamCount;
                uint slotOffset;
                uint payloadEnd;
            };
            kernel void rans_decode(
                device const uchar *payload [[buffer(0)]],
                device const uint *tables [[buffer(1)]],
                constant StreamDesc *streams [[buffer(2)]],
                device uchar *output [[buffer(3)]],
                device const uint2 *slotTables [[buffer(4)]],
                device atomic_uint *failure [[buffer(5)]],
                uint gid [[thread_position_in_grid]])
            {
                StreamDesc d = streams[gid];
                uint state = uint(payload[d.payloadOffset])
                    | (uint(payload[d.payloadOffset + 1u]) << 8)
                    | (uint(payload[d.payloadOffset + 2u]) << 16)
                    | (uint(payload[d.payloadOffset + 3u]) << 24);
                uint data = d.payloadOffset + 8u;
                device const uint2 *slots = slotTables + d.slotOffset;
                for (uint position = d.streamIndex; position < d.symbolCount;
                     position += d.streamCount) {
                    while (state < (1u << 23)) {
                        if (data >= d.payloadEnd) {
                            atomic_store_explicit(failure, 1u, memory_order_relaxed);
                            return;
                        }
                        state = (state << 8) | uint(payload[data]);
                        data += 1u;
                    }
                    uint slot = state & 16383u;
                    uint2 info = slots[slot];
                    output[d.outputOffset + position * d.stride] = uchar(info.x & 0xFFu);
                    state = (info.x >> 8u) * (state >> 14u) + slot - info.y;
                }
            }
            """
        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: "rans_decode") else {
            throw ModelError.internalInconsistency(detail: "rANS kernel is missing")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelines[key] = pipeline
        return pipeline
    }

    func queue(for device: MTLDevice) -> MTLCommandQueue? {
        lock.lock()
        defer { lock.unlock() }
        let key = ObjectIdentifier(device)
        if let queue = queues[key] { return queue }
        guard let queue = device.makeCommandQueue() else { return nil }
        queues[key] = queue
        return queue
    }
}

/// The GPU resources a rANS layer keeps: models, slot tables, staging, and the
/// host-side stream plans (payload offsets relative to one expert's payload).
/// unchecked-invariant: every stored property is immutable after `init`; the
/// Metal buffers are safe for concurrent use, and the shared staging buffer is
/// serialised by the streamer's `ransStagingLock`.
final class RANSLayerResources: @unchecked Sendable {
    let layout: ExpertRANSIndex.Layout
    let phases: [UInt8]
    let tables: MTLBuffer
    let slotTables: MTLBuffer
    let pipeline: MTLComputePipelineState
    let queue: MTLCommandQueue
    let payload: MTLBuffer
    let payloadStride: Int
    let descriptors: MTLBuffer
    let failure: MTLBuffer
    let maximumWave = 8
    /// Stream plans per expert, payload offsets relative to that expert's run.
    let plans: [[RANSStreamPlan]]

    init(
        layout: ExpertRANSIndex.Layout, phases: [UInt8], device: MTLDevice,
        payloadPath: String
    ) throws {
        self.layout = layout
        self.phases = phases
        guard layout.models.count == phases.count, !layout.models.isEmpty else {
            throw ModelError.internalInconsistency(
                detail: "rANS sidecar has \(layout.models.count) models for "
                    + "\(phases.count) phases")
        }
        self.pipeline = try RANSGPU.shared.pipeline(for: device)
        guard let queue = RANSGPU.shared.queue(for: device) else {
            throw ModelError.internalInconsistency(detail: "rANS command queue unavailable")
        }
        self.queue = queue

        var tables: [UInt32] = []
        var slots: [UInt32] = []
        for model in layout.models {
            var cumulative = [UInt32](repeating: 0, count: 257)
            for symbol in 0..<256 {
                cumulative[symbol + 1] = cumulative[symbol] + UInt32(model.frequencies[symbol])
            }
            tables.append(contentsOf: cumulative)
            tables.append(contentsOf: model.frequencies.map { UInt32($0) })
            var symbol = 0
            for slot in 0..<16384 {
                while symbol < 255 && Int(cumulative[symbol + 1]) <= slot { symbol += 1 }
                slots.append(UInt32(symbol) | (UInt32(model.frequencies[symbol]) << 8))
                slots.append(cumulative[symbol])
            }
        }
        guard
            let tableBuffer = device.makeBuffer(
                bytes: tables, length: tables.count * 4, options: .storageModeShared),
            let slotBuffer = device.makeBuffer(
                bytes: slots, length: slots.count * 4, options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.tables = tableBuffer
        self.slotTables = slotBuffer

        // Stream plans: read each block's u16 length table from the payload so
        // the kernel's per-stream offsets are fixed at layer open.
        let fd = open(payloadPath, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else {
            throw ModelError.partialInstall(path: payloadPath)
        }
        defer { close(fd) }
        // Build the stream plans in parallel: one pread per block is 3,840
        // reads a layer, which measured ~17 s of prefill when done serially.
        let accumulator = RANSPlanAccumulator(expertCount: layout.blocks.count)
        DispatchQueue.concurrentPerform(iterations: layout.blocks.count) { expert in
            let blocks = layout.blocks[expert]
            let runStart = blocks.first.map { Int($0.payloadOffset) } ?? 0
            var plans: [RANSStreamPlan] = []
            var payloadBytes = 0
            for (index, block) in blocks.enumerated() {
                guard
                    let (lengths, base, lengthBytes) = Self.lengthTable(
                        block: block, fd: fd, payloadPath: payloadPath)
                else {
                    // A failed read cannot be skipped: an empty plan for the
                    // expert would silently skip its decode and leave stale
                    // weights in the slot, and a zero payload size would
                    // understate the staging stride the wave is read into.
                    accumulator.lock.lock()
                    accumulator.failures.append(
                        "expert \(expert) block \(index): stream-length table unreadable")
                    accumulator.lock.unlock()
                    return
                }
                let streams = ExpertRANSIndex.streams(symbolCount: Int(block.symbolCount))
                var streamOffset = base + lengthBytes
                for stream in 0..<streams {
                    let length = Int(lengths[stream * 2]) | Int(lengths[stream * 2 + 1]) << 8
                    let blockEnd = base + Int(block.payloadLength)
                    guard length >= 8, streamOffset + length <= blockEnd else {
                        accumulator.lock.lock()
                        accumulator.failures.append(
                            "expert \(expert) block \(index) stream \(stream): "
                                + "invalid stream range \(streamOffset)..<\(streamOffset + length) "
                                + "for block ending at \(blockEnd)")
                        accumulator.lock.unlock()
                        return
                    }
                    plans.append(
                        RANSStreamPlan(
                            payloadOffset: UInt32(streamOffset - runStart),
                            tableOffset: UInt32(index * 513),
                            outputOffset: block.rawOffset,
                            stride: UInt32(block.stride),
                            symbolCount: block.symbolCount,
                            streamIndex: UInt32(stream),
                            streamCount: UInt32(streams),
                            slotOffset: UInt32(index * 16384),
                            payloadEnd: UInt32(streamOffset + length - runStart),
                            phase: index < phases.count ? Int(phases[index]) : 0))
                    streamOffset += length
                }
                payloadBytes += Int(block.payloadLength)
            }
            accumulator.lock.lock()
            accumulator.plans[expert] = plans
            accumulator.payloadSizes[expert] = payloadBytes
            accumulator.lock.unlock()
        }
        if let first = accumulator.failures.first {
            throw ModelError.indexCorrupt(detail: "rANS layer \(payloadPath): \(first)")
        }
        let maximumPayload = accumulator.payloadSizes.max() ?? 0
        self.plans = accumulator.plans
        let stride = max(1, maximumPayload)
        self.payloadStride = stride
        let maximumStreams = maximumWave * layout.models.count * 256
        guard
            let payloadBuffer = device.makeBuffer(
                length: max(1, stride * maximumWave), options: .storageModeShared),
            let descriptorBuffer = device.makeBuffer(
                length: max(1, maximumStreams * RANSStreamPlan.descriptorWords * 4),
                options: .storageModeShared),
            let failureBuffer = device.makeBuffer(length: 4, options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.payload = payloadBuffer
        self.descriptors = descriptorBuffer
        self.failure = failureBuffer
    }

    /// The payload byte run of one expert, for the bounded reader.
    func payloadRange(expert: Int) -> (offset: UInt64, count: Int)? {
        guard layout.blocks.indices.contains(expert),
            let first = layout.blocks[expert].first,
            let last = layout.blocks[expert].last
        else { return nil }
        let start = first.payloadOffset
        let end = last.payloadOffset + UInt64(last.payloadLength)
        return (start, Int(end - start))
    }

    /// Reads one block's u16 stream-length table from the payload.
    private static func lengthTable(
        block: ExpertRANSIndex.Block, fd: Int32, payloadPath: String
    ) -> (lengths: [UInt8], base: Int, lengthBytes: Int)? {
        let streams = ExpertRANSIndex.streams(symbolCount: Int(block.symbolCount))
        let lengthBytes = streams * 2
        let base = Int(block.payloadOffset)
        var lengths = [UInt8](repeating: 0, count: lengthBytes)
        let complete = lengths.withUnsafeMutableBytes { raw -> Bool in
            guard let destination = raw.baseAddress else { return false }
            var filled = 0
            while filled < lengthBytes {
                let got = pread(
                    fd, destination.advanced(by: filled), lengthBytes - filled,
                    off_t(base) + off_t(filled))
                if got < 0 {
                    if errno == EINTR { continue }
                    return false
                }
                if got == 0 { return false }
                filled += got
            }
            return true
        }
        guard complete else { return nil }
        return (lengths, base, lengthBytes)
    }

}
