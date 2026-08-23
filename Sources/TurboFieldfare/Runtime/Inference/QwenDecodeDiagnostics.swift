import Foundation

public protocol QwenDecodeDiagnosticsProviding: AnyObject {
    var lastQwenDecodeDiagnostics: QwenDecodeDiagnostics? { get }
}

struct QwenGPUStageTimings: Equatable {
    static let sampleCount = 8

    let mixerNanos: UInt64
    let sharedExpertNanos: UInt64
    let routerNanos: UInt64

    init?(timestamps: [UInt64]) {
        guard timestamps.count == Self.sampleCount,
              timestamps.allSatisfy({ $0 > 0 && $0 != .max }),
              zip(timestamps, timestamps.dropFirst()).allSatisfy({ $0 <= $1 }) else {
            return nil
        }
        self.mixerNanos = timestamps[2] - timestamps[1]
        self.sharedExpertNanos = timestamps[4] - timestamps[3]
        self.routerNanos = timestamps[6] - timestamps[5]
    }
}

struct QwenRoutedGPUStageTimings: Equatable {
    static let sampleCount = 8

    let phase1Nanos: UInt64
    let phase2Nanos: UInt64
    let combineNanos: UInt64

    init?(timestamps: [UInt64]) {
        guard timestamps.count == Self.sampleCount,
              timestamps.allSatisfy({ $0 > 0 && $0 != .max }),
              zip(timestamps, timestamps.dropFirst()).allSatisfy({ $0 <= $1 }) else {
            return nil
        }
        self.phase1Nanos = timestamps[2] - timestamps[1]
        self.phase2Nanos = timestamps[4] - timestamps[3]
        self.combineNanos = timestamps[6] - timestamps[5]
    }
}

public struct QwenDecodeLayerDiagnostics: Sendable, Equatable {
    public let layer: Int
    public let isFullAttention: Bool
    public let elapsedNanos: UInt64
    public let expertFetchNanos: UInt64
    public let routePlanningNanos: UInt64
    public let routedSetupNanos: UInt64
    public let routedCommandBufferEncodingNanos: UInt64
    public let routedCommandBufferCommitNanos: UInt64
    public let routedCommandBufferWaitNanos: UInt64
    public let expertReadCount: Int
    public let expertReadNanos: UInt64
    public let expertReadMaxNanos: UInt64
    public let gpuStageTimingSampled: Bool
    public let gpuMixerNanos: UInt64
    public let gpuSharedExpertNanos: UInt64
    public let gpuRouterNanos: UInt64
    public let routedGPUStageTimingSampled: Bool
    public let gpuRoutedPhase1Nanos: UInt64
    public let gpuRoutedPhase2Nanos: UInt64
    public let gpuRoutedCombineNanos: UInt64
    public let routedExperts: [Int]
    public let routingWeights: [Float]

    public init(layer: Int,
                isFullAttention: Bool,
                elapsedNanos: UInt64,
                expertFetchNanos: UInt64,
                routePlanningNanos: UInt64 = 0,
                routedSetupNanos: UInt64 = 0,
                routedCommandBufferEncodingNanos: UInt64 = 0,
                routedCommandBufferCommitNanos: UInt64 = 0,
                routedCommandBufferWaitNanos: UInt64 = 0,
                expertReadCount: Int = 0,
                expertReadNanos: UInt64 = 0,
                expertReadMaxNanos: UInt64 = 0,
                gpuStageTimingSampled: Bool = false,
                gpuMixerNanos: UInt64 = 0,
                gpuSharedExpertNanos: UInt64 = 0,
                gpuRouterNanos: UInt64 = 0,
                routedGPUStageTimingSampled: Bool = false,
                gpuRoutedPhase1Nanos: UInt64 = 0,
                gpuRoutedPhase2Nanos: UInt64 = 0,
                gpuRoutedCombineNanos: UInt64 = 0,
                routedExperts: [Int] = [],
                routingWeights: [Float] = []) {
        self.layer = layer
        self.isFullAttention = isFullAttention
        self.elapsedNanos = elapsedNanos
        self.expertFetchNanos = expertFetchNanos
        self.routePlanningNanos = routePlanningNanos
        self.routedSetupNanos = routedSetupNanos
        self.routedCommandBufferEncodingNanos = routedCommandBufferEncodingNanos
        self.routedCommandBufferCommitNanos = routedCommandBufferCommitNanos
        self.routedCommandBufferWaitNanos = routedCommandBufferWaitNanos
        self.expertReadCount = expertReadCount
        self.expertReadNanos = expertReadNanos
        self.expertReadMaxNanos = expertReadMaxNanos
        self.gpuStageTimingSampled = gpuStageTimingSampled
        self.gpuMixerNanos = gpuMixerNanos
        self.gpuSharedExpertNanos = gpuSharedExpertNanos
        self.gpuRouterNanos = gpuRouterNanos
        self.routedGPUStageTimingSampled = routedGPUStageTimingSampled
        self.gpuRoutedPhase1Nanos = gpuRoutedPhase1Nanos
        self.gpuRoutedPhase2Nanos = gpuRoutedPhase2Nanos
        self.gpuRoutedCombineNanos = gpuRoutedCombineNanos
        self.routedExperts = routedExperts
        self.routingWeights = routingWeights
    }
}

public struct QwenDecodeDiagnostics: Sendable, Equatable {
    public let wallNanos: UInt64
    public let embeddingNanos: UInt64
    public let layerNanos: UInt64
    public let logitsNanos: UInt64
    public let expertFetchNanos: UInt64
    public let expertReadCount: Int
    public let expertReadNanos: UInt64
    public let expertReadMaxNanos: UInt64
    public let mixerNanos: UInt64
    public let routerNanos: UInt64
    public let routePlanningNanos: UInt64
    public let sharedExpertNanos: UInt64
    public let routedSetupNanos: UInt64
    public let routedCommandBufferEncodingNanos: UInt64
    public let routedCommandBufferCommitNanos: UInt64
    public let routedCommandBufferWaitNanos: UInt64
    public let gpuStageTimingSampleCount: Int
    public let gpuMixerNanos: UInt64
    public let gpuSharedExpertNanos: UInt64
    public let gpuRouterNanos: UInt64
    public let routedGPUStageTimingSampleCount: Int
    public let gpuRoutedPhase1Nanos: UInt64
    public let gpuRoutedPhase2Nanos: UInt64
    public let gpuRoutedCombineNanos: UInt64
    public let routedExpertCombineNanos: UInt64
    public let layerCount: Int
    public let fullAttentionLayerCount: Int
    public let deltaNetLayerCount: Int
    public let commandBufferCount: Int
    public let routerEvaluationCount: Int
    public let routedExpertCount: Int
    public let routedExpertCacheHitCount: Int
    public let routedExpertCacheMissCount: Int
    public let routedExpertEstimatedBytes: UInt64
    public let layers: [QwenDecodeLayerDiagnostics]

    public init(wallNanos: UInt64,
                embeddingNanos: UInt64,
                layerNanos: UInt64,
                logitsNanos: UInt64,
                expertFetchNanos: UInt64,
                layerCount: Int,
                fullAttentionLayerCount: Int,
                deltaNetLayerCount: Int,
                commandBufferCount: Int,
                routerEvaluationCount: Int,
                routedExpertCount: Int,
                routedExpertCacheHitCount: Int,
                routedExpertCacheMissCount: Int,
                routedExpertEstimatedBytes: UInt64,
                mixerNanos: UInt64 = 0,
                routerNanos: UInt64 = 0,
                routePlanningNanos: UInt64 = 0,
                sharedExpertNanos: UInt64 = 0,
                routedSetupNanos: UInt64 = 0,
                routedCommandBufferEncodingNanos: UInt64 = 0,
                routedCommandBufferCommitNanos: UInt64 = 0,
                routedCommandBufferWaitNanos: UInt64 = 0,
                gpuStageTimingSampleCount: Int = 0,
                gpuMixerNanos: UInt64 = 0,
                gpuSharedExpertNanos: UInt64 = 0,
                gpuRouterNanos: UInt64 = 0,
                routedGPUStageTimingSampleCount: Int = 0,
                gpuRoutedPhase1Nanos: UInt64 = 0,
                gpuRoutedPhase2Nanos: UInt64 = 0,
                gpuRoutedCombineNanos: UInt64 = 0,
                routedExpertCombineNanos: UInt64 = 0,
                expertReadCount: Int = 0,
                expertReadNanos: UInt64 = 0,
                expertReadMaxNanos: UInt64 = 0,
                layers: [QwenDecodeLayerDiagnostics] = []) {
        self.wallNanos = wallNanos
        self.embeddingNanos = embeddingNanos
        self.layerNanos = layerNanos
        self.logitsNanos = logitsNanos
        self.expertFetchNanos = expertFetchNanos
        self.expertReadCount = expertReadCount
        self.expertReadNanos = expertReadNanos
        self.expertReadMaxNanos = expertReadMaxNanos
        self.mixerNanos = mixerNanos
        self.routerNanos = routerNanos
        self.routePlanningNanos = routePlanningNanos
        self.sharedExpertNanos = sharedExpertNanos
        self.routedSetupNanos = routedSetupNanos
        self.routedCommandBufferEncodingNanos = routedCommandBufferEncodingNanos
        self.routedCommandBufferCommitNanos = routedCommandBufferCommitNanos
        self.routedCommandBufferWaitNanos = routedCommandBufferWaitNanos
        self.gpuStageTimingSampleCount = gpuStageTimingSampleCount
        self.gpuMixerNanos = gpuMixerNanos
        self.gpuSharedExpertNanos = gpuSharedExpertNanos
        self.gpuRouterNanos = gpuRouterNanos
        self.routedGPUStageTimingSampleCount = routedGPUStageTimingSampleCount
        self.gpuRoutedPhase1Nanos = gpuRoutedPhase1Nanos
        self.gpuRoutedPhase2Nanos = gpuRoutedPhase2Nanos
        self.gpuRoutedCombineNanos = gpuRoutedCombineNanos
        self.routedExpertCombineNanos = routedExpertCombineNanos
        self.layerCount = layerCount
        self.fullAttentionLayerCount = fullAttentionLayerCount
        self.deltaNetLayerCount = deltaNetLayerCount
        self.commandBufferCount = commandBufferCount
        self.routerEvaluationCount = routerEvaluationCount
        self.routedExpertCount = routedExpertCount
        self.routedExpertCacheHitCount = routedExpertCacheHitCount
        self.routedExpertCacheMissCount = routedExpertCacheMissCount
        self.routedExpertEstimatedBytes = routedExpertEstimatedBytes
        self.layers = layers
    }
}

public struct QwenDecodeLayerAggregate: Codable, Sendable, Equatable {
    public let layer: Int
    public let isFullAttention: Bool
    public let decodeStepCount: Int
    public let elapsedNanos: UInt64
    public let expertFetchNanos: UInt64
    public let routePlanningNanos: UInt64
    public let routedSetupNanos: UInt64
    public let routedCommandBufferEncodingNanos: UInt64
    public let routedCommandBufferCommitNanos: UInt64
    public let routedCommandBufferWaitNanos: UInt64
    public let expertReadCount: Int
    public let expertReadNanos: UInt64
    public let expertReadMaxNanos: UInt64
    public let gpuStageTimingSampleCount: Int
    public let gpuMixerNanos: UInt64
    public let gpuSharedExpertNanos: UInt64
    public let gpuRouterNanos: UInt64
    public let routedGPUStageTimingSampleCount: Int
    public let gpuRoutedPhase1Nanos: UInt64
    public let gpuRoutedPhase2Nanos: UInt64
    public let gpuRoutedCombineNanos: UInt64
    public let routedExpertTrace: [[Int]]
    public let routingWeightTrace: [[Float]]

    public init(layer: Int,
                isFullAttention: Bool,
                decodeStepCount: Int,
                elapsedNanos: UInt64,
                expertFetchNanos: UInt64,
                routePlanningNanos: UInt64 = 0,
                routedSetupNanos: UInt64 = 0,
                routedCommandBufferEncodingNanos: UInt64 = 0,
                routedCommandBufferCommitNanos: UInt64 = 0,
                routedCommandBufferWaitNanos: UInt64 = 0,
                expertReadCount: Int = 0,
                expertReadNanos: UInt64 = 0,
                expertReadMaxNanos: UInt64 = 0,
                gpuStageTimingSampleCount: Int = 0,
                gpuMixerNanos: UInt64 = 0,
                gpuSharedExpertNanos: UInt64 = 0,
                gpuRouterNanos: UInt64 = 0,
                routedGPUStageTimingSampleCount: Int = 0,
                gpuRoutedPhase1Nanos: UInt64 = 0,
                gpuRoutedPhase2Nanos: UInt64 = 0,
                gpuRoutedCombineNanos: UInt64 = 0,
                routedExpertTrace: [[Int]] = [],
                routingWeightTrace: [[Float]] = []) {
        self.layer = layer
        self.isFullAttention = isFullAttention
        self.decodeStepCount = decodeStepCount
        self.elapsedNanos = elapsedNanos
        self.expertFetchNanos = expertFetchNanos
        self.routePlanningNanos = routePlanningNanos
        self.routedSetupNanos = routedSetupNanos
        self.routedCommandBufferEncodingNanos = routedCommandBufferEncodingNanos
        self.routedCommandBufferCommitNanos = routedCommandBufferCommitNanos
        self.routedCommandBufferWaitNanos = routedCommandBufferWaitNanos
        self.expertReadCount = expertReadCount
        self.expertReadNanos = expertReadNanos
        self.expertReadMaxNanos = expertReadMaxNanos
        self.gpuStageTimingSampleCount = gpuStageTimingSampleCount
        self.gpuMixerNanos = gpuMixerNanos
        self.gpuSharedExpertNanos = gpuSharedExpertNanos
        self.gpuRouterNanos = gpuRouterNanos
        self.routedGPUStageTimingSampleCount = routedGPUStageTimingSampleCount
        self.gpuRoutedPhase1Nanos = gpuRoutedPhase1Nanos
        self.gpuRoutedPhase2Nanos = gpuRoutedPhase2Nanos
        self.gpuRoutedCombineNanos = gpuRoutedCombineNanos
        self.routedExpertTrace = routedExpertTrace
        self.routingWeightTrace = routingWeightTrace
    }
}

public struct QwenDecodeDiagnosticsAggregate: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 5

    public let schemaVersion: Int
    public let decodeStepCount: Int
    public let decodeLoopWallNanos: UInt64
    public let forwardWallNanos: UInt64
    public let embeddingNanos: UInt64
    public let layerNanos: UInt64
    public let logitsNanos: UInt64
    public let mixerNanos: UInt64
    public let routerNanos: UInt64
    public let routePlanningNanos: UInt64
    public let sharedExpertNanos: UInt64
    public let routedSetupNanos: UInt64
    public let routedCommandBufferEncodingNanos: UInt64
    public let routedCommandBufferCommitNanos: UInt64
    public let routedCommandBufferWaitNanos: UInt64
    public let gpuStageTimingSampleCount: Int
    public let gpuMixerNanos: UInt64
    public let gpuSharedExpertNanos: UInt64
    public let gpuRouterNanos: UInt64
    public let routedGPUStageTimingSampleCount: Int
    public let gpuRoutedPhase1Nanos: UInt64
    public let gpuRoutedPhase2Nanos: UInt64
    public let gpuRoutedCombineNanos: UInt64
    public let expertFetchNanos: UInt64
    public let expertReadCount: Int
    public let expertReadNanos: UInt64
    public let expertReadMaxNanos: UInt64
    public let routedExpertCombineNanos: UInt64
    public let samplingNanos: UInt64
    public let commandBufferSubmissionCount: Int
    public let routerEvaluationCount: Int
    public let routedExpertCount: Int
    public let routedExpertCacheHitCount: Int
    public let routedExpertCacheMissCount: Int
    public let routedExpertEstimatedBytes: UInt64
    public let attributedWallNanos: UInt64
    public let residualWallNanos: UInt64
    public let layers: [QwenDecodeLayerAggregate]

    public init(
        schemaVersion: Int = QwenDecodeDiagnosticsAggregate.currentSchemaVersion,
        decodeStepCount: Int,
        decodeLoopWallNanos: UInt64,
        forwardWallNanos: UInt64,
        embeddingNanos: UInt64,
        layerNanos: UInt64,
        logitsNanos: UInt64,
        mixerNanos: UInt64,
        routerNanos: UInt64,
        routePlanningNanos: UInt64,
        sharedExpertNanos: UInt64,
        routedSetupNanos: UInt64 = 0,
        routedCommandBufferEncodingNanos: UInt64 = 0,
        routedCommandBufferCommitNanos: UInt64 = 0,
        routedCommandBufferWaitNanos: UInt64 = 0,
        gpuStageTimingSampleCount: Int = 0,
        gpuMixerNanos: UInt64 = 0,
        gpuSharedExpertNanos: UInt64 = 0,
        gpuRouterNanos: UInt64 = 0,
        routedGPUStageTimingSampleCount: Int = 0,
        gpuRoutedPhase1Nanos: UInt64 = 0,
        gpuRoutedPhase2Nanos: UInt64 = 0,
        gpuRoutedCombineNanos: UInt64 = 0,
        expertFetchNanos: UInt64,
        routedExpertCombineNanos: UInt64,
        samplingNanos: UInt64,
        commandBufferSubmissionCount: Int,
        routerEvaluationCount: Int,
        routedExpertCount: Int,
        routedExpertCacheHitCount: Int,
        routedExpertCacheMissCount: Int,
        routedExpertEstimatedBytes: UInt64,
        attributedWallNanos: UInt64,
        residualWallNanos: UInt64,
        expertReadCount: Int = 0,
        expertReadNanos: UInt64 = 0,
        expertReadMaxNanos: UInt64 = 0,
        layers: [QwenDecodeLayerAggregate] = []) {
        self.schemaVersion = schemaVersion
        self.decodeStepCount = decodeStepCount
        self.decodeLoopWallNanos = decodeLoopWallNanos
        self.forwardWallNanos = forwardWallNanos
        self.embeddingNanos = embeddingNanos
        self.layerNanos = layerNanos
        self.logitsNanos = logitsNanos
        self.mixerNanos = mixerNanos
        self.routerNanos = routerNanos
        self.routePlanningNanos = routePlanningNanos
        self.sharedExpertNanos = sharedExpertNanos
        self.routedSetupNanos = routedSetupNanos
        self.routedCommandBufferEncodingNanos = routedCommandBufferEncodingNanos
        self.routedCommandBufferCommitNanos = routedCommandBufferCommitNanos
        self.routedCommandBufferWaitNanos = routedCommandBufferWaitNanos
        self.gpuStageTimingSampleCount = gpuStageTimingSampleCount
        self.gpuMixerNanos = gpuMixerNanos
        self.gpuSharedExpertNanos = gpuSharedExpertNanos
        self.gpuRouterNanos = gpuRouterNanos
        self.routedGPUStageTimingSampleCount = routedGPUStageTimingSampleCount
        self.gpuRoutedPhase1Nanos = gpuRoutedPhase1Nanos
        self.gpuRoutedPhase2Nanos = gpuRoutedPhase2Nanos
        self.gpuRoutedCombineNanos = gpuRoutedCombineNanos
        self.expertFetchNanos = expertFetchNanos
        self.expertReadCount = expertReadCount
        self.expertReadNanos = expertReadNanos
        self.expertReadMaxNanos = expertReadMaxNanos
        self.routedExpertCombineNanos = routedExpertCombineNanos
        self.samplingNanos = samplingNanos
        self.commandBufferSubmissionCount = commandBufferSubmissionCount
        self.routerEvaluationCount = routerEvaluationCount
        self.routedExpertCount = routedExpertCount
        self.routedExpertCacheHitCount = routedExpertCacheHitCount
        self.routedExpertCacheMissCount = routedExpertCacheMissCount
        self.routedExpertEstimatedBytes = routedExpertEstimatedBytes
        self.attributedWallNanos = attributedWallNanos
        self.residualWallNanos = residualWallNanos
        self.layers = layers
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case decodeStepCount = "decode_step_count"
        case decodeLoopWallNanos = "decode_loop_wall_nanos"
        case forwardWallNanos = "forward_wall_nanos"
        case embeddingNanos = "embedding_nanos"
        case layerNanos = "layer_nanos"
        case logitsNanos = "logits_nanos"
        case mixerNanos = "mixer_nanos"
        case routerNanos = "router_nanos"
        case routePlanningNanos = "route_planning_nanos"
        case sharedExpertNanos = "shared_expert_nanos"
        case routedSetupNanos = "routed_setup_nanos"
        case routedCommandBufferEncodingNanos = "routed_command_buffer_encoding_nanos"
        case routedCommandBufferCommitNanos = "routed_command_buffer_commit_nanos"
        case routedCommandBufferWaitNanos = "routed_command_buffer_wait_nanos"
        case gpuStageTimingSampleCount = "gpu_stage_timing_sample_count"
        case gpuMixerNanos = "gpu_mixer_nanos"
        case gpuSharedExpertNanos = "gpu_shared_expert_nanos"
        case gpuRouterNanos = "gpu_router_nanos"
        case routedGPUStageTimingSampleCount = "routed_gpu_stage_timing_sample_count"
        case gpuRoutedPhase1Nanos = "gpu_routed_phase1_nanos"
        case gpuRoutedPhase2Nanos = "gpu_routed_phase2_nanos"
        case gpuRoutedCombineNanos = "gpu_routed_combine_nanos"
        case expertFetchNanos = "expert_fetch_nanos"
        case expertReadCount = "expert_read_count"
        case expertReadNanos = "expert_read_nanos"
        case expertReadMaxNanos = "expert_read_max_nanos"
        case routedExpertCombineNanos = "routed_expert_combine_nanos"
        case samplingNanos = "sampling_nanos"
        case commandBufferSubmissionCount = "command_buffer_submission_count"
        case routerEvaluationCount = "router_evaluation_count"
        case routedExpertCount = "routed_expert_count"
        case routedExpertCacheHitCount = "routed_expert_cache_hit_count"
        case routedExpertCacheMissCount = "routed_expert_cache_miss_count"
        case routedExpertEstimatedBytes = "routed_expert_estimated_bytes"
        case attributedWallNanos = "attributed_wall_nanos"
        case residualWallNanos = "residual_wall_nanos"
        case layers
    }
}

@inline(__always)
func saturatedAdd(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
    let (sum, overflow) = lhs.addingReportingOverflow(rhs)
    return overflow ? .max : sum
}

@inline(__always)
func saturatedAdd(_ lhs: Int, _ rhs: Int) -> Int {
    let (sum, overflow) = lhs.addingReportingOverflow(rhs)
    return overflow ? .max : sum
}

struct QwenDecodeDiagnosticsAggregateAccumulator {
    private(set) var decodeStepCount = 0
    private(set) var forwardWallNanos: UInt64 = 0
    private(set) var embeddingNanos: UInt64 = 0
    private(set) var layerNanos: UInt64 = 0
    private(set) var logitsNanos: UInt64 = 0
    private(set) var mixerNanos: UInt64 = 0
    private(set) var routerNanos: UInt64 = 0
    private(set) var routePlanningNanos: UInt64 = 0
    private(set) var sharedExpertNanos: UInt64 = 0
    private(set) var routedSetupNanos: UInt64 = 0
    private(set) var routedCommandBufferEncodingNanos: UInt64 = 0
    private(set) var routedCommandBufferCommitNanos: UInt64 = 0
    private(set) var routedCommandBufferWaitNanos: UInt64 = 0
    private(set) var gpuStageTimingSampleCount = 0
    private(set) var gpuMixerNanos: UInt64 = 0
    private(set) var gpuSharedExpertNanos: UInt64 = 0
    private(set) var gpuRouterNanos: UInt64 = 0
    private(set) var routedGPUStageTimingSampleCount = 0
    private(set) var gpuRoutedPhase1Nanos: UInt64 = 0
    private(set) var gpuRoutedPhase2Nanos: UInt64 = 0
    private(set) var gpuRoutedCombineNanos: UInt64 = 0
    private(set) var expertFetchNanos: UInt64 = 0
    private(set) var expertReadCount = 0
    private(set) var expertReadNanos: UInt64 = 0
    private(set) var expertReadMaxNanos: UInt64 = 0
    private(set) var routedExpertCombineNanos: UInt64 = 0
    private(set) var samplingNanos: UInt64 = 0
    private(set) var commandBufferSubmissionCount = 0
    private(set) var routerEvaluationCount = 0
    private(set) var routedExpertCount = 0
    private(set) var routedExpertCacheHitCount = 0
    private(set) var routedExpertCacheMissCount = 0
    private(set) var routedExpertEstimatedBytes: UInt64 = 0
    private var layers: [QwenDecodeLayerAggregate] = []

    mutating func add(_ diagnostics: QwenDecodeDiagnostics) {
        decodeStepCount = saturatedAdd(decodeStepCount, 1)
        forwardWallNanos = saturatedAdd(forwardWallNanos, diagnostics.wallNanos)
        embeddingNanos = saturatedAdd(embeddingNanos, diagnostics.embeddingNanos)
        layerNanos = saturatedAdd(layerNanos, diagnostics.layerNanos)
        logitsNanos = saturatedAdd(logitsNanos, diagnostics.logitsNanos)
        mixerNanos = saturatedAdd(mixerNanos, diagnostics.mixerNanos)
        routerNanos = saturatedAdd(routerNanos, diagnostics.routerNanos)
        routePlanningNanos = saturatedAdd(routePlanningNanos, diagnostics.routePlanningNanos)
        sharedExpertNanos = saturatedAdd(sharedExpertNanos, diagnostics.sharedExpertNanos)
        routedSetupNanos = saturatedAdd(routedSetupNanos, diagnostics.routedSetupNanos)
        routedCommandBufferEncodingNanos = saturatedAdd(
            routedCommandBufferEncodingNanos, diagnostics.routedCommandBufferEncodingNanos)
        routedCommandBufferCommitNanos = saturatedAdd(
            routedCommandBufferCommitNanos, diagnostics.routedCommandBufferCommitNanos)
        routedCommandBufferWaitNanos = saturatedAdd(
            routedCommandBufferWaitNanos, diagnostics.routedCommandBufferWaitNanos)
        gpuStageTimingSampleCount = saturatedAdd(
            gpuStageTimingSampleCount, diagnostics.gpuStageTimingSampleCount)
        gpuMixerNanos = saturatedAdd(gpuMixerNanos, diagnostics.gpuMixerNanos)
        gpuSharedExpertNanos = saturatedAdd(
            gpuSharedExpertNanos, diagnostics.gpuSharedExpertNanos)
        gpuRouterNanos = saturatedAdd(gpuRouterNanos, diagnostics.gpuRouterNanos)
        routedGPUStageTimingSampleCount = saturatedAdd(
            routedGPUStageTimingSampleCount, diagnostics.routedGPUStageTimingSampleCount)
        gpuRoutedPhase1Nanos = saturatedAdd(
            gpuRoutedPhase1Nanos, diagnostics.gpuRoutedPhase1Nanos)
        gpuRoutedPhase2Nanos = saturatedAdd(
            gpuRoutedPhase2Nanos, diagnostics.gpuRoutedPhase2Nanos)
        gpuRoutedCombineNanos = saturatedAdd(
            gpuRoutedCombineNanos, diagnostics.gpuRoutedCombineNanos)
        expertFetchNanos = saturatedAdd(expertFetchNanos, diagnostics.expertFetchNanos)
        expertReadCount = saturatedAdd(expertReadCount, diagnostics.expertReadCount)
        expertReadNanos = saturatedAdd(expertReadNanos, diagnostics.expertReadNanos)
        expertReadMaxNanos = max(expertReadMaxNanos, diagnostics.expertReadMaxNanos)
        routedExpertCombineNanos = saturatedAdd(
            routedExpertCombineNanos, diagnostics.routedExpertCombineNanos)
        commandBufferSubmissionCount = saturatedAdd(
            commandBufferSubmissionCount, diagnostics.commandBufferCount)
        routerEvaluationCount = saturatedAdd(
            routerEvaluationCount, diagnostics.routerEvaluationCount)
        routedExpertCount = saturatedAdd(routedExpertCount, diagnostics.routedExpertCount)
        routedExpertCacheHitCount = saturatedAdd(
            routedExpertCacheHitCount, diagnostics.routedExpertCacheHitCount)
        routedExpertCacheMissCount = saturatedAdd(
            routedExpertCacheMissCount, diagnostics.routedExpertCacheMissCount)
        routedExpertEstimatedBytes = saturatedAdd(
            routedExpertEstimatedBytes, diagnostics.routedExpertEstimatedBytes)

        for layer in diagnostics.layers {
            guard let index = layers.firstIndex(where: { $0.layer == layer.layer }) else {
                layers.append(QwenDecodeLayerAggregate(
                    layer: layer.layer,
                    isFullAttention: layer.isFullAttention,
                    decodeStepCount: 1,
                    elapsedNanos: layer.elapsedNanos,
                    expertFetchNanos: layer.expertFetchNanos,
                    routePlanningNanos: layer.routePlanningNanos,
                    routedSetupNanos: layer.routedSetupNanos,
                    routedCommandBufferEncodingNanos: layer.routedCommandBufferEncodingNanos,
                    routedCommandBufferCommitNanos: layer.routedCommandBufferCommitNanos,
                    routedCommandBufferWaitNanos: layer.routedCommandBufferWaitNanos,
                    expertReadCount: layer.expertReadCount,
                    expertReadNanos: layer.expertReadNanos,
                    expertReadMaxNanos: layer.expertReadMaxNanos,
                    gpuStageTimingSampleCount: layer.gpuStageTimingSampled ? 1 : 0,
                    gpuMixerNanos: layer.gpuMixerNanos,
                    gpuSharedExpertNanos: layer.gpuSharedExpertNanos,
                    gpuRouterNanos: layer.gpuRouterNanos,
                    routedGPUStageTimingSampleCount:
                        layer.routedGPUStageTimingSampled ? 1 : 0,
                    gpuRoutedPhase1Nanos: layer.gpuRoutedPhase1Nanos,
                    gpuRoutedPhase2Nanos: layer.gpuRoutedPhase2Nanos,
                    gpuRoutedCombineNanos: layer.gpuRoutedCombineNanos,
                    routedExpertTrace: [layer.routedExperts],
                    routingWeightTrace: [layer.routingWeights]))
                continue
            }
            let current = layers[index]
            layers[index] = QwenDecodeLayerAggregate(
                layer: current.layer,
                isFullAttention: current.isFullAttention,
                decodeStepCount: saturatedAdd(current.decodeStepCount, 1),
                elapsedNanos: saturatedAdd(current.elapsedNanos, layer.elapsedNanos),
                expertFetchNanos: saturatedAdd(current.expertFetchNanos,
                                               layer.expertFetchNanos),
                routePlanningNanos: saturatedAdd(current.routePlanningNanos,
                                                 layer.routePlanningNanos),
                routedSetupNanos: saturatedAdd(current.routedSetupNanos,
                                               layer.routedSetupNanos),
                routedCommandBufferEncodingNanos: saturatedAdd(
                    current.routedCommandBufferEncodingNanos,
                    layer.routedCommandBufferEncodingNanos),
                routedCommandBufferCommitNanos: saturatedAdd(
                    current.routedCommandBufferCommitNanos,
                    layer.routedCommandBufferCommitNanos),
                routedCommandBufferWaitNanos: saturatedAdd(
                    current.routedCommandBufferWaitNanos,
                    layer.routedCommandBufferWaitNanos),
                expertReadCount: saturatedAdd(current.expertReadCount,
                                              layer.expertReadCount),
                expertReadNanos: saturatedAdd(current.expertReadNanos,
                                              layer.expertReadNanos),
                expertReadMaxNanos: max(current.expertReadMaxNanos,
                                        layer.expertReadMaxNanos),
                gpuStageTimingSampleCount: saturatedAdd(
                    current.gpuStageTimingSampleCount,
                    layer.gpuStageTimingSampled ? 1 : 0),
                gpuMixerNanos: saturatedAdd(current.gpuMixerNanos,
                                            layer.gpuMixerNanos),
                gpuSharedExpertNanos: saturatedAdd(current.gpuSharedExpertNanos,
                                                   layer.gpuSharedExpertNanos),
                gpuRouterNanos: saturatedAdd(current.gpuRouterNanos,
                                             layer.gpuRouterNanos),
                routedGPUStageTimingSampleCount: saturatedAdd(
                    current.routedGPUStageTimingSampleCount,
                    layer.routedGPUStageTimingSampled ? 1 : 0),
                gpuRoutedPhase1Nanos: saturatedAdd(current.gpuRoutedPhase1Nanos,
                                                   layer.gpuRoutedPhase1Nanos),
                gpuRoutedPhase2Nanos: saturatedAdd(current.gpuRoutedPhase2Nanos,
                                                   layer.gpuRoutedPhase2Nanos),
                gpuRoutedCombineNanos: saturatedAdd(current.gpuRoutedCombineNanos,
                                                    layer.gpuRoutedCombineNanos),
                routedExpertTrace: current.routedExpertTrace + [layer.routedExperts],
                routingWeightTrace: current.routingWeightTrace + [layer.routingWeights])
        }
    }

    mutating func addSamplingNanos(_ nanos: UInt64) {
        samplingNanos = saturatedAdd(samplingNanos, nanos)
    }

    func makeDiagnostics(decodeLoopWallNanos: UInt64) -> QwenDecodeDiagnosticsAggregate {
        let attributedWallNanos = saturatedAdd(forwardWallNanos, samplingNanos)
        let residualWallNanos = decodeLoopWallNanos > attributedWallNanos
            ? decodeLoopWallNanos - attributedWallNanos
            : 0
        return QwenDecodeDiagnosticsAggregate(
            decodeStepCount: decodeStepCount,
            decodeLoopWallNanos: decodeLoopWallNanos,
            forwardWallNanos: forwardWallNanos,
            embeddingNanos: embeddingNanos,
            layerNanos: layerNanos,
            logitsNanos: logitsNanos,
            mixerNanos: mixerNanos,
            routerNanos: routerNanos,
            routePlanningNanos: routePlanningNanos,
            sharedExpertNanos: sharedExpertNanos,
            routedSetupNanos: routedSetupNanos,
            routedCommandBufferEncodingNanos: routedCommandBufferEncodingNanos,
            routedCommandBufferCommitNanos: routedCommandBufferCommitNanos,
            routedCommandBufferWaitNanos: routedCommandBufferWaitNanos,
            gpuStageTimingSampleCount: gpuStageTimingSampleCount,
            gpuMixerNanos: gpuMixerNanos,
            gpuSharedExpertNanos: gpuSharedExpertNanos,
            gpuRouterNanos: gpuRouterNanos,
            routedGPUStageTimingSampleCount: routedGPUStageTimingSampleCount,
            gpuRoutedPhase1Nanos: gpuRoutedPhase1Nanos,
            gpuRoutedPhase2Nanos: gpuRoutedPhase2Nanos,
            gpuRoutedCombineNanos: gpuRoutedCombineNanos,
            expertFetchNanos: expertFetchNanos,
            routedExpertCombineNanos: routedExpertCombineNanos,
            samplingNanos: samplingNanos,
            commandBufferSubmissionCount: commandBufferSubmissionCount,
            routerEvaluationCount: routerEvaluationCount,
            routedExpertCount: routedExpertCount,
            routedExpertCacheHitCount: routedExpertCacheHitCount,
            routedExpertCacheMissCount: routedExpertCacheMissCount,
            routedExpertEstimatedBytes: routedExpertEstimatedBytes,
            attributedWallNanos: attributedWallNanos,
            residualWallNanos: residualWallNanos,
            expertReadCount: expertReadCount,
            expertReadNanos: expertReadNanos,
            expertReadMaxNanos: expertReadMaxNanos,
            layers: layers)
    }
}