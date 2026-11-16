import Foundation
import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct MetalExpertReaderTests {
    @Test func loadsDisjointFileRangesDirectlyIntoMetalBuffers() throws {
        let page = 16_384
        let bytes =
            [UInt8](repeating: 0x31, count: page)
            + [UInt8](repeating: 0x72, count: page)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("metal-io-reader-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(bytes).write(to: url)

        let context = try MetalContext()
        let first = try #require(
            context.device.makeBuffer(length: page, options: .storageModeShared))
        let second = try #require(
            context.device.makeBuffer(length: page, options: .storageModeShared))
        let reader = try MetalExpertReader(path: url.path, device: context.device)

        try reader.fetch(
            offsets: [0, UInt64(page)],
            into: [first, second],
            byteCount: page)

        let firstBytes = [UInt8](
            UnsafeRawBufferPointer(
                start: first.contents(), count: page))
        let secondBytes = [UInt8](
            UnsafeRawBufferPointer(
                start: second.contents(), count: page))
        #expect(firstBytes.allSatisfy { $0 == 0x31 })
        #expect(secondBytes.allSatisfy { $0 == 0x72 })
    }

    @Test func submitsEventSignalledLoadWithoutHostWait() async throws {
        let page = 16_384
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("metal-io-event-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(repeating: 0x5a, count: page).write(to: url)

        let context = try MetalContext()
        let coordinator = try #require(ExpertIOEventCoordinator(device: context.device))
        let token = try coordinator.reserve()
        let destination = try #require(
            context.device.makeBuffer(
                length: page, options: .storageModeShared))
        let reader = try MetalExpertReader(path: url.path, device: context.device)

        try await withCheckedThrowingContinuation { continuation in
            do {
                try reader.beginFetch(
                    offsets: [0], into: [destination], byteCount: page,
                    completionToken: token
                ) { result in
                    continuation.resume(with: result)
                }
            } catch {
                continuation.resume(throwing: error)
            }
        }

        #expect(token.event.signaledValue >= token.value)
        #expect(
            token.status.contents().advanced(by: token.statusOffset)
                .load(as: UInt32.self) == UInt32(MTLIOStatus.complete.rawValue))
        let bytes = UnsafeRawBufferPointer(start: destination.contents(), count: page)
        #expect(bytes.allSatisfy { $0 == 0x5a })
    }

    @Test func eventGatedStagingCopyPublishesOnlyAfterMetalIO() throws {
        let page = 16_384
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("metal-io-staging-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(repeating: 0xa7, count: page).write(to: url)

        let context = try MetalContext()
        let coordinator = try #require(ExpertIOEventCoordinator(device: context.device))
        let token = try coordinator.reserve()
        let stagingPool = try MetalExpertStagingPool(
            device: context.device, byteCount: page, slotCapacity: 1)
        let lease = try #require(stagingPool.tryAcquire(count: 1))
        let destination = try #require(
            context.device.makeBuffer(
                length: page, options: .storageModeShared))
        memset(destination.contents(), 0, page)
        let transfer = MetalExpertStagingTransfer(
            lease: lease, destinations: [destination], destinationOffsets: [0],
            byteCount: page)
        let reader = try MetalExpertReader(path: url.path, device: context.device)

        // Commit the dependent compute work immediately. There is deliberately
        // no host completion wait between MTLIO submission and the GPU copy.
        try reader.beginFetch(
            offsets: [0], into: lease.buffers, byteCount: page,
            completionToken: token
        ) { _ in }
        let compute = try #require(context.queue.makeCommandBuffer())
        compute.encodeWaitForEvent(token.event, value: token.value)
        try transfer.encodeCopy(commandBuffer: compute)
        compute.commit()
        compute.waitUntilCompleted()
        defer { transfer.release() }

        #expect(compute.status == .completed)
        #expect(
            token.status.contents().advanced(by: token.statusOffset)
                .load(as: UInt32.self) == UInt32(MTLIOStatus.complete.rawValue))
        let bytes = UnsafeRawBufferPointer(start: destination.contents(), count: page)
        #expect(bytes.allSatisfy { $0 == 0xa7 })
    }
}
