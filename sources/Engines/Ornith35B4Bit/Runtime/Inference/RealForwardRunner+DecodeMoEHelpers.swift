import Foundation
import Metal
import TinyTitanCore

extension RealForwardRunner {
    func classifyGPUResidencyForDecode(plan: RoutedExpertFetchPlan) throws {
        let hitCount = min(
            Int(residencyHitCount.contents().load(as: UInt32.self)),
            cfg.topKExperts)
        let missCount = min(
            Int(residencyMissCount.contents().load(as: UInt32.self)),
            cfg.topKExperts)
        let hitPointer = residencyHitPositions.contents()
            .bindMemory(to: UInt32.self, capacity: cfg.topKExperts)
        let missPointer = residencyMissPositions.contents()
            .bindMemory(to: UInt32.self, capacity: cfg.topKExperts)
        for index in 0..<hitCount {
            decodeHitSlotsScratch.append(hitPointer[index])
        }
        for index in 0..<missCount {
            decodeMissSlotsScratch.append(missPointer[index])
        }
        decodeHitSlotsScratch.sort()
        decodeMissSlotsScratch.sort()
        let gpuMisses = decodeMissSlotsScratch.map(Int.init)
        let planMisses = plan.misses.sorted()
        guard Set(gpuMisses).isSuperset(of: planMisses) else {
            throw ModelError.internalInconsistency(
                detail: "GPU residency classification disagrees with cache "
                    + "plan: gpu=\(gpuMisses) plan=\(planMisses)")
        }
        totalGPUClassifiedHits &+= UInt64(hitCount)
        totalGPUClassifiedMisses &+= UInt64(missCount)
        if missCount == 0 { totalGPUResidencyAllHitLayers &+= 1 }
        DecodeExpertPartition.populate(
            topK: cfg.topKExperts,
            missIndices: plan.misses,
            hits: &decodeHitSlotsScratch,
            misses: &decodeMissSlotsScratch)
    }

    func issueDecodeRDAdvice(
        layer L: Int,
        position: Int,
        plannedFetch: RoutedExpertFetchPlan?,
        missCount: Int,
        experts: [Int]
    ) throws {
        let requestedMisses = missCount
        let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
            layer: L,
            missCount: requestedMisses)
        if let skipped = shouldSkipRDAdvice(
            position: position,
            requestedMisses: requestedMisses,
            estimatedBytes: estimatedAdviceBytes,
            canOverlapUsefulGPUWork: true)
        {
            recordRDAdvice(skipped, wallNanos: 0)
        } else {
            let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let result: ExpertIOAdviceResult
            if let plannedFetch {
                result = try model.adviseRoutedExperts(plan: plannedFetch)
            } else {
                result = try model.adviseRoutedExperts(layer: L, experts: experts)
            }
            let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
            recordRDAdvice(result, wallNanos: wallNanos)
            updateRDAdvicePolicy(after: result, position: position)
        }
    }
}
