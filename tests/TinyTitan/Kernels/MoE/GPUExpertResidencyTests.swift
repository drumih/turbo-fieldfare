import Metal
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct GPUExpertResidencyTests {
    private struct Classification {
        let hits: [UInt32]
        let misses: [UInt32]
        let missExperts: [UInt32]
        let slots: [UInt32]
        let generations: [UInt64]
    }

    @Test func loadingResidentAndEvictedEntriesClassifyCorrectly() throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let context = try MetalContext()
        let streamer = try PreadExpertStreamer(
            layout: PreadExpertStreamerTests.makeLayout(path: url.path),
            device: context.device,
            slotCount: 2)
        let moe = try MoE(
            context: context,
            siluActivation: true,
            specializedD: 2048,
            specializedF: 512,
            specializedNumExperts: 4)

        let loading = try streamer.planExpertsCached(experts: [0])
        #expect(
            streamer.residencyEntry(expert: 0).state
                == ExpertResidencyEntry.loading)
        var result = try classify(
            [0, 1, 2, 3], streamer: streamer,
            moe: moe, context: context)
        #expect(result.hits.isEmpty)
        #expect(result.misses == [0, 1, 2, 3])

        _ = try streamer.executeExpertCachePlan(loading)
        _ = try streamer.loadExpertsCached(experts: [2])
        result = try classify(
            [0, 1, 2, 3], streamer: streamer,
            moe: moe, context: context)
        #expect(result.hits == [0, 2])
        #expect(result.misses == [1, 3])
        #expect(result.missExperts == [1, 3])
        #expect(result.slots[0] != ExpertResidencyEntry.notResidentSlot)
        #expect(result.slots[2] != ExpertResidencyEntry.notResidentSlot)
        #expect(result.generations[0] > 0)

        _ = try streamer.loadExpertsCached(experts: [1, 3])
        result = try classify(
            [0, 1, 2, 3], streamer: streamer,
            moe: moe, context: context)
        #expect(result.hits == [1, 3])
        #expect(result.misses == [0, 2])
        #expect(
            streamer.residencyEntry(expert: 0).state
                == ExpertResidencyEntry.empty)
    }

    private func classify(
        _ experts: [UInt32],
        streamer: PreadExpertStreamer,
        moe: MoE,
        context: MetalContext
    ) throws -> Classification {
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            try values.withUnsafeBytes { bytes in
                let base = try #require(bytes.baseAddress)
                return try #require(
                    context.device.makeBuffer(
                        bytes: base,
                        length: max(1, bytes.count),
                        options: .storageModeShared))
            }
        }
        let topK = try buffer(experts)
        let hitCount = try buffer([UInt32(0)])
        let hitPositions = try buffer([UInt32](repeating: 0, count: experts.count))
        let missCount = try buffer([UInt32(0)])
        let missPositions = try buffer([UInt32](repeating: 0, count: experts.count))
        let missExperts = try buffer([UInt32](repeating: 0, count: experts.count))
        let slots = try buffer([UInt32](repeating: 0, count: experts.count))
        let generations = try buffer([UInt64](repeating: 0, count: experts.count))
        let resources = streamer.expertResidencyResources()
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try moe.encodeResidencyClassification(
            commandBuffer: commandBuffer,
            topKIndices: topK,
            residencyTable: resources.table,
            hitCount: hitCount,
            hitPositions: hitPositions,
            missCount: missCount,
            missPositions: missPositions,
            missExperts: missExperts,
            resolvedSlots: slots,
            resolvedGenerations: generations,
            topK: UInt32(experts.count),
            numExperts: UInt32(resources.expertCount))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        func values<T>(_ buffer: MTLBuffer, count: Int, as: T.Type) -> [T] {
            Array(
                UnsafeBufferPointer(
                    start: buffer.contents().bindMemory(to: T.self, capacity: count),
                    count: count))
        }
        let hitN = Int(hitCount.contents().load(as: UInt32.self))
        let missN = Int(missCount.contents().load(as: UInt32.self))
        return Classification(
            hits: values(hitPositions, count: hitN, as: UInt32.self),
            misses: values(missPositions, count: missN, as: UInt32.self),
            missExperts: values(missExperts, count: missN, as: UInt32.self),
            slots: values(slots, count: experts.count, as: UInt32.self),
            generations: values(generations, count: experts.count, as: UInt64.self))
    }
}
