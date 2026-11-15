import Foundation
import Metal
import TinyTitanCore

/// One bounded MTLIO queue shared by all lazy per-layer readers in a model.
/// Keeping a queue per layer lets prompt ingestion create 40 queues and exhaust
/// the system's Metal-I/O worker limit before decode reaches its event path.
/// unchecked-invariant: queue is immutable and Metal serializes its commands.
public final class MetalExpertIOService: @unchecked Sendable {
    fileprivate let queue: MTLIOCommandQueue

    init(device: MTLDevice, maximumCommandsInFlight: Int = 4) throws {
        precondition(maximumCommandsInFlight > 0)
        let descriptor = MTLIOCommandQueueDescriptor()
        descriptor.type = .concurrent
        descriptor.priority = .high
        descriptor.maxCommandBufferCount = maximumCommandsInFlight
        descriptor.maxCommandsInFlight = maximumCommandsInFlight
        do {
            self.queue = try device.makeIOCommandQueue(descriptor: descriptor)
        } catch {
            throw MetalExpertReader.Failure.queueCreation(String(describing: error))
        }
    }
}

/// Experimental Metal I/O reader for direct file-to-slot-buffer loads.
///
/// The production streamer deliberately remains on `F_NOCACHE` pread until an
/// A/B run proves this path does not grow an undeclared page-cache working set.
/// Metal I/O exposes queue and command concurrency but its public contract does
/// not promise TinyTitan's bounded-cache invariant.
/// unchecked-invariant: the queue and file handle are immutable Metal objects;
/// each fetch owns a distinct command buffer and this type has no mutable state.
public final class MetalExpertReader: @unchecked Sendable {
    public enum Failure: Error, CustomStringConvertible {
        case queueCreation(String)
        case fileHandleCreation(String)
        case load(String)

        public var description: String {
            switch self {
            case .queueCreation(let detail):
                return "Metal I/O queue creation failed: \(detail)"
            case .fileHandleCreation(let detail):
                return "Metal I/O file handle creation failed: \(detail)"
            case .load(let detail):
                return "Metal I/O expert load failed: \(detail)"
            }
        }
    }

    private let ioService: MetalExpertIOService
    private let fileHandle: MTLIOFileHandle

    public convenience init(
        path: String,
        device: MTLDevice,
        maximumCommandsInFlight: Int = 4
    ) throws {
        let service = try MetalExpertIOService(
            device: device, maximumCommandsInFlight: maximumCommandsInFlight)
        try self.init(path: path, device: device, service: service)
    }

    init(path: String, device: MTLDevice, service: MetalExpertIOService) throws {
        self.ioService = service
        do {
            self.fileHandle = try device.makeIOFileHandle(
                url: URL(fileURLWithPath: path))
        } catch {
            throw Failure.fileHandleCreation(String(describing: error))
        }
    }

    public func beginFetch(
        offsets: [UInt64],
        into buffers: [MTLBuffer],
        byteCount: Int,
        destinationOffsets: [Int]? = nil,
        completionToken: ExpertIOCompletionToken? = nil,
        completion: @escaping @Sendable (Result<Void, any Error>) -> Void
    ) throws {
        precondition(offsets.count == buffers.count)
        precondition(destinationOffsets == nil || destinationOffsets?.count == buffers.count)
        precondition(byteCount > 0)
        guard !offsets.isEmpty else {
            completion(.success(()))
            return
        }
        let commandBuffer = ioService.queue.makeCommandBuffer()
        commandBuffer.label = "TinyTitan expert-slot loads"
        for index in offsets.indices {
            let offset = offsets[index]
            let buffer = buffers[index]
            let destinationOffset = destinationOffsets?[index] ?? 0
            guard destinationOffset >= 0,
                buffer.length - destinationOffset >= byteCount,
                offset <= UInt64(Int.max)
            else {
                throw Failure.load("source or destination range is out of bounds")
            }
            commandBuffer.load(
                buffer,
                offset: destinationOffset,
                size: byteCount,
                sourceHandle: fileHandle,
                sourceHandleOffset: Int(offset))
        }
        if let completionToken {
            // The status copy is ordered after every load. Metal writes
            // MTLIOStatusComplete (3) or an error state before signalling the
            // event, so the dependent compute kernels can gate safely without
            // a host callback.
            commandBuffer.copyStatus(
                buffer: completionToken.status,
                offset: completionToken.statusOffset)
            commandBuffer.signalEvent(
                completionToken.event,
                value: completionToken.value)
        }
        commandBuffer.addCompletedHandler { commandBuffer in
            guard commandBuffer.status == .complete else {
                completion(
                    .failure(
                        Failure.load(
                            commandBuffer.error?.localizedDescription
                                ?? "status \(commandBuffer.status.rawValue)")))
                return
            }
            completion(.success(()))
        }
        commandBuffer.commit()
    }

    public func fetch(
        offsets: [UInt64],
        into buffers: [MTLBuffer],
        byteCount: Int,
        destinationOffsets: [Int]? = nil
    ) throws {
        let condition = NSCondition()
        nonisolated(unsafe) var result: Result<Void, any Error>?
        try beginFetch(
            offsets: offsets, into: buffers, byteCount: byteCount,
            destinationOffsets: destinationOffsets
        ) { outcome in
            condition.lock()
            result = outcome
            condition.signal()
            condition.unlock()
        }
        condition.lock()
        while result == nil { condition.wait() }
        guard let outcome = result else {
            condition.unlock()
            throw ModelError.internalInconsistency(
                detail: "the parallel expert read finished without a result")
        }
        condition.unlock()
        try outcome.get()
    }
}
