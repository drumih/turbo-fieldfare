import Foundation
import Metal
import Synchronization
import TinyTitanFormat
import TinyTitanCore

// The rANS decode path: read a wave's payload runs, then run the kernel into
// the slots, phase 0 before the publish and phase 1 behind the GPU's phase 1.
extension PreadExpertStreamer {

    /// Decodes every miss of a plan, one wave at a time.
    func executeRANSPlan(
        _ plan: ExpertCachePlan,
        firstStage: (() -> Void)? = nil
    ) throws {
        guard let resources = ransResources else { return }
        var start = 0
        while start < plan.misses.count {
            let end = min(start + resources.maximumWave, plan.misses.count)
            let misses = Array(plan.misses[start..<end])
            try decodeRANSExperts(
                experts: misses.map { plan.experts[$0] },
                slots: misses.map { plan.assignedSlots[$0] },
                resources: resources,
                firstStage: end == plan.misses.count ? firstStage : nil)
            start = end
        }
    }

    /// One expert, for the single-expert API.
    func decodeRANSExpert(expert: Int, slot: Int) throws {
        guard let resources = ransResources else { return }
        try decodeRANSExperts(
            experts: [expert], slots: [slot], resources: resources, firstStage: nil)
    }

    /// Reads the wave's payload runs and decodes them into the slots.
    ///
    /// The payload is read with the same bounded reader as a raw layer, into a
    /// shared MTLBuffer so the kernel needs no upload copy. Descriptors are
    /// written per expert and phase; the kernel runs one encoder per (expert,
    /// phase), because each slot is its own MTLBuffer.
    func decodeRANSExperts(
        experts: [Int],
        slots: [Int],
        resources: RANSLayerResources,
        firstStage: (() -> Void)?
    ) throws {
        try decodeRANSInto(
            experts: experts,
            outputs: slots.map { slotBuffers[$0] },
            offsets: slots.map { slotBufferOffsets[$0] },
            resources: resources, firstStage: firstStage)
    }

    /// The general form: any output buffers and base offsets. The ring's GPU
    /// prefetch uses it with dedicated buffers, so a predicted expert is
    /// decoded a layer ahead and adopted instead of decoded on demand.
    ///
    /// `offsets` narrows to the 32-bit descriptor field the kernel reads, so a
    /// base offset above `UInt32.max` traps there rather than wrapping; every
    /// slot is its own `MTLBuffer` at offset 0, so none gets that far.
    func decodeRANSInto(
        experts: [Int],
        outputs: [MTLBuffer],
        offsets: [UInt64],
        resources: RANSLayerResources,
        firstStage: (() -> Void)?
    ) throws {
        precondition(
            experts.count == outputs.count && experts.count == offsets.count,
            "each rANS expert needs one output")
        guard !experts.isEmpty else { return }
        guard experts.count <= resources.maximumWave else {
            throw ModelError.internalInconsistency(
                detail: "rANS wave of \(experts.count) exceeds \(resources.maximumWave)")
        }

        // One staging buffer is shared with the ring's speculative decode.
        ransStagingLock.lock()
        defer { ransStagingLock.unlock() }
        var sourceOffsets: [UInt64] = []
        var counts: [Int] = []
        var destinations: [UnsafeMutableRawPointer] = []
        let payloadBase = resources.payload.contents()
        for (position, expert) in experts.enumerated() {
            guard let range = resources.payloadRange(expert: expert) else {
                throw ModelError.internalInconsistency(
                    detail: "rANS sidecar has no payload for expert \(expert)")
            }
            sourceOffsets.append(range.offset)
            counts.append(range.count)
            destinations.append(
                payloadBase.advanced(by: position * resources.payloadStride))
        }
        let tRead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try readRanges(offsets: sourceOffsets, counts: counts, into: destinations)
        let tReadDone = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let descriptorWords = RANSStreamPlan.descriptorWords
        let descriptorStride = resources.layout.models.count * 256 * descriptorWords * 4
        let descriptorPointer = resources.descriptors.contents()

        for phase in 0...1 {
            resources.failure.contents().storeBytes(of: UInt32(0), as: UInt32.self)
            guard let command = resources.queue.makeCommandBuffer() else {
                throw ModelError.internalInconsistency(detail: "rANS command buffer unavailable")
            }
            for (position, expert) in experts.enumerated() {
                let plans = resources.plans[expert].filter { $0.phase == phase }
                guard !plans.isEmpty else { continue }
                let region = descriptorPointer.advanced(
                    by: position * descriptorStride
                ).bindMemory(to: UInt32.self, capacity: plans.count * descriptorWords)
                for (index, plan) in plans.enumerated() {
                    let base = index * descriptorWords
                    let payloadBase = UInt32(position * resources.payloadStride)
                    region[base] = plan.payloadOffset + payloadBase
                    region[base + 1] = plan.tableOffset
                    region[base + 2] = plan.outputOffset + UInt32(offsets[position])
                    region[base + 3] = plan.stride
                    region[base + 4] = plan.symbolCount
                    region[base + 5] = plan.streamIndex
                    region[base + 6] = plan.streamCount
                    region[base + 7] = plan.slotOffset
                    region[base + 8] = plan.payloadEnd + payloadBase
                }
                guard let encoder = command.makeComputeCommandEncoder() else {
                    throw ModelError.internalInconsistency(
                        detail: "rANS compute encoder unavailable")
                }
                encoder.setComputePipelineState(resources.pipeline)
                encoder.setBuffer(resources.payload, offset: 0, index: 0)
                encoder.setBuffer(resources.tables, offset: 0, index: 1)
                encoder.setBuffer(
                    resources.descriptors, offset: position * descriptorStride, index: 2)
                encoder.setBuffer(outputs[position], offset: 0, index: 3)
                encoder.setBuffer(resources.slotTables, offset: 0, index: 4)
                encoder.setBuffer(resources.failure, offset: 0, index: 5)
                encoder.dispatchThreads(
                    MTLSize(width: plans.count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                encoder.endEncoding()
            }
            command.commit()
            command.waitUntilCompleted()
            let tPhase = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if ProcessInfo.processInfo.environment["TINYTITAN_RANS_TRACE"] == "1" {
                FileHandle.standardError.write(
                    Data(
                        String(
                            format: "[rans] experts=%d phase=%d streams=%d read=%.2f gpu=%.2f ms\n",
                            experts.count, phase,
                            resources.plans[experts[0]].filter { $0.phase == phase }.count,
                            Double(tReadDone - tRead) / 1e6,
                            Double(tPhase - tReadDone) / 1e6
                        ).utf8))
            }
            if let error = command.error {
                throw ModelError.internalInconsistency(
                    detail: "rANS decode failed: \(error)")
            }
            guard resources.failure.contents().load(as: UInt32.self) == 0 else {
                throw ModelError.indexCorrupt(detail: "rANS stream exhausted during decode")
            }
            if phase == 0 { firstStage?() }
        }
    }
}
