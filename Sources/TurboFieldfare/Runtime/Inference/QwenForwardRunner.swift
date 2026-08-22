import Foundation
import Metal

/// Decode runner for the text-only Qwen3.6 MoE model.
///
/// The runner intentionally owns the two different forms of causal state:
/// Gated DeltaNet state for linear-attention layers and an FP16 KV cache for
/// every fourth full-attention layer. It does not share Gemma's fused layer
/// tail because Qwen's residual block and untied output head are different.
private struct QwenPromptStateSnapshot {
    let position: Int
    let greedyToken: UInt32
    let deltaStates: [Int: QwenGatedDeltaNetSnapshot]
    let fullCaches: [Int: QwenFullAttentionKVSnapshot]
}

private struct QwenDecodeDiagnosticsAccumulator {
    var wallNanos: UInt64 = 0
    var embeddingNanos: UInt64 = 0
    var layerNanos: UInt64 = 0
    var logitsNanos: UInt64 = 0
    var expertFetchNanos: UInt64 = 0
    var expertReadCount = 0
    var expertReadNanos: UInt64 = 0
    var expertReadMaxNanos: UInt64 = 0
    var currentLayerExpertReadMaxNanos: UInt64 = 0
    var currentLayerRoutedExperts: [Int] = []
    var currentLayerRoutingWeights: [Float] = []
    var mixerNanos: UInt64 = 0
    var routerNanos: UInt64 = 0
    var routePlanningNanos: UInt64 = 0
    var sharedExpertNanos: UInt64 = 0
    var routedExpertCombineNanos: UInt64 = 0
    var layerCount = 0
    var fullAttentionLayerCount = 0
    var deltaNetLayerCount = 0
    var commandBufferCount = 0
    var routerEvaluationCount = 0
    var routedExpertCount = 0
    var routedExpertCacheHitCount = 0
    var routedExpertCacheMissCount = 0
    var routedExpertEstimatedBytes: UInt64 = 0
    var layers: [QwenDecodeLayerDiagnostics] = []

    func makeDiagnostics() -> QwenDecodeDiagnostics {
        QwenDecodeDiagnostics(
            wallNanos: wallNanos,
            embeddingNanos: embeddingNanos,
            layerNanos: layerNanos,
            logitsNanos: logitsNanos,
            expertFetchNanos: expertFetchNanos,
            layerCount: layerCount,
            fullAttentionLayerCount: fullAttentionLayerCount,
            deltaNetLayerCount: deltaNetLayerCount,
            commandBufferCount: commandBufferCount,
            routerEvaluationCount: routerEvaluationCount,
            routedExpertCount: routedExpertCount,
            routedExpertCacheHitCount: routedExpertCacheHitCount,
            routedExpertCacheMissCount: routedExpertCacheMissCount,
            routedExpertEstimatedBytes: routedExpertEstimatedBytes,
            mixerNanos: mixerNanos,
            routerNanos: routerNanos,
            routePlanningNanos: routePlanningNanos,
            sharedExpertNanos: sharedExpertNanos,
            routedExpertCombineNanos: routedExpertCombineNanos,
            expertReadCount: expertReadCount,
            expertReadNanos: expertReadNanos,
            expertReadMaxNanos: expertReadMaxNanos,
            layers: layers)
    }
}

public final class QwenForwardRunner: ChunkedPrefillRunner, PromptStateSnapshotting,
    ContextWindowReporting, ForwardRunner, QwenDecodeDiagnosticsProviding, @unchecked Sendable {
    private let model: Model
    private let context: MetalContext
    private let config: ArchConfig
    private let embed: EmbedLookupInt4
    private let rms: RMSNorm
    private let deltaNet: QwenGatedDeltaNet
    private let deltaElementwise: QwenElementwise
    private let attention: QwenFullAttention
    private let moe: QwenMoE
    private let head: QwenUntiedLMHead
    private let fusionHead: LMHeadChainInt4
    private let useFusedGreedyHead: Bool
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMSNorm: PrefillRMSNorm
    private let prefillProjection: QwenPrefillProjectionBatch
    private let prefillDeltaNet: QwenPrefillDeltaNet
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let deltaStates: QwenGatedDeltaNetStateManager
    private let fullCaches: [QwenFullAttentionKVCache?]
    private let prefillScratchCache = QwenPrefillScratchCache()

    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let mixerOutput: MTLBuffer
    private let projection: MTLBuffer
    private let query: MTLBuffer
    private let queryGate: MTLBuffer
    private let key: MTLBuffer
    private let value: MTLBuffer
    private let normalizedQuery: MTLBuffer
    private let normalizedKey: MTLBuffer
    private let attentionOutput: MTLBuffer
    private let gatedAttention: MTLBuffer
    private let qkv: MTLBuffer
    private let deltaZ: MTLBuffer
    private let deltaA: MTLBuffer
    private let deltaB: MTLBuffer
    private let deltaQuery: MTLBuffer
    private let deltaKey: MTLBuffer
    private let deltaValue: MTLBuffer
    private let deltaConvolution: MTLBuffer
    private let deltaDecay: MTLBuffer
    private let deltaBeta: MTLBuffer
    private let deltaOutput: MTLBuffer
    private let sharedOutput: MTLBuffer
    private let routedOutput: MTLBuffer
    private let combinedOutput: MTLBuffer
    private let moeActs: MTLBuffer
    private let sharedGateScratch: MTLBuffer
    private let sharedUpScratch: MTLBuffer
    private let sharedActScratch: MTLBuffer
    private let routeIndices: MTLBuffer
    private let routeWeights: MTLBuffer
    private let greedyTokenBuffer: MTLBuffer

    public let maxContext: Int
    private var position = 0
    private var commandBufferSubmissionCount = 0
    private var promptStateSnapshot: QwenPromptStateSnapshot?
    private var activeDecodeDiagnostics: QwenDecodeDiagnosticsAccumulator?

    public private(set) var lastQwenDecodeDiagnostics: QwenDecodeDiagnostics?

    public init(model: Model,
                context: MetalContext,
                maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        guard model.config.modelFamily == .qwen36MoeText else {
            throw ModelError.archMismatch(field: "modelFamily",
                                           expected: "qwen36MoeText",
                                           actual: "\(model.config.modelFamily)")
        }
        guard maxContext > 0 else {
            throw ModelError.archMismatch(field: "maxContext",
                                           expected: "positive",
                                           actual: "\(maxContext)")
        }
        self.model = model
        self.context = context
        self.config = model.config
        self.maxContext = maxContext
        self.embed = try EmbedLookupInt4(context: context)
        self.rms = try RMSNorm(context: context)
        self.deltaNet = try QwenGatedDeltaNet(context: context)
        self.deltaElementwise = try QwenElementwise(context: context)
        self.attention = try QwenFullAttention(context: context)
        self.moe = try QwenMoE(context: context)
        self.head = try QwenUntiedLMHead(
            context: context,
            geometry: QwenLMHeadGeometry(vocabularySize: config.vocabSize,
                                          hiddenSize: config.hiddenSize))
        self.fusionHead = try LMHeadChainInt4(
            context: context,
            maxD: config.hiddenSize,
            maxVocab: config.vocabSize)
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.prefillRMSNorm = try PrefillRMSNorm(context: context)
        self.prefillProjection = try QwenPrefillProjectionBatch(context: context)
        self.prefillDeltaNet = try QwenPrefillDeltaNet(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(
            context: context,
            maxD: config.hiddenSize)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(context: context)
        self.prefillMoE = try PrefillMoE(context: context)
        self.deltaStates = try QwenGatedDeltaNetStateManager(
            context: context,
            layerCount: config.numLayers,
            geometry: QwenGatedDeltaNetGeometry(
                keyHeads: config.linearNumKeyHeads,
                valueHeads: config.linearNumValueHeads,
                keyHeadDim: config.linearKeyHeadDim,
                valueHeadDim: config.linearValueHeadDim,
                convolutionKernel: config.linearConvKernelDim),
            convolutionChannels: config.linearNumKeyHeads * config.linearKeyHeadDim * 2
                + config.linearNumValueHeads * config.linearValueHeadDim)

        let device = context.device
        func makeBuffer(_ elements: Int,
                        stride: Int = MemoryLayout<Float16>.stride) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * stride,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return buffer
        }

        let hiddenSize = config.hiddenSize
        let qWidth = config.numHeads * config.fullHeadDim
        let kvWidth = config.numFullKVHeads * config.fullHeadDim
        let deltaKeyWidth = config.linearNumKeyHeads * config.linearKeyHeadDim
        let deltaValueWidth = config.linearNumValueHeads * config.linearValueHeadDim
        let deltaQKVWidth = deltaKeyWidth * 2 + deltaValueWidth
        let sharedWidth = config.intermediateSize

        self.hidden = try makeBuffer(hiddenSize)
        let normedWidth = max(hiddenSize, deltaValueWidth)
        self.normed = try makeBuffer(normedWidth)
        self.mixerOutput = try makeBuffer(hiddenSize)
        self.projection = try makeBuffer(max(qWidth * 2, deltaQKVWidth))
        self.query = try makeBuffer(max(qWidth, deltaKeyWidth))
        self.queryGate = try makeBuffer(qWidth)
        self.key = try makeBuffer(max(kvWidth, deltaKeyWidth))
        self.value = try makeBuffer(max(kvWidth, deltaValueWidth))
        self.normalizedQuery = try makeBuffer(qWidth)
        self.normalizedKey = try makeBuffer(kvWidth)
        self.attentionOutput = try makeBuffer(qWidth)
        self.gatedAttention = try makeBuffer(qWidth)
        self.qkv = try makeBuffer(deltaQKVWidth)
        self.deltaZ = try makeBuffer(deltaValueWidth)
        self.deltaA = try makeBuffer(config.linearNumValueHeads,
                                     stride: MemoryLayout<Float16>.stride)
        self.deltaB = try makeBuffer(config.linearNumValueHeads,
                                     stride: MemoryLayout<Float16>.stride)
        self.deltaQuery = try makeBuffer(deltaKeyWidth)
        self.deltaKey = try makeBuffer(deltaKeyWidth)
        self.deltaValue = try makeBuffer(deltaValueWidth)
        self.deltaConvolution = try makeBuffer(deltaQKVWidth)
        self.deltaDecay = try makeBuffer(config.linearNumValueHeads,
                                         stride: MemoryLayout<Float>.stride)
        self.deltaBeta = try makeBuffer(config.linearNumValueHeads,
                                        stride: MemoryLayout<Float>.stride)
        self.deltaOutput = try makeBuffer(deltaValueWidth)
        self.sharedOutput = try makeBuffer(hiddenSize)
        self.routedOutput = try makeBuffer(hiddenSize)
        self.combinedOutput = try makeBuffer(hiddenSize)
        self.zeroBuffer = try makeBuffer(hiddenSize)
        memset(self.zeroBuffer.contents(), 0, self.zeroBuffer.length)
        self.moeActs = try makeBuffer(config.topKExperts * config.moeIntermediateSize)
        self.sharedGateScratch = try makeBuffer(sharedWidth)
        self.sharedUpScratch = try makeBuffer(sharedWidth)
        self.sharedActScratch = try makeBuffer(sharedWidth)
        self.routeIndices = try makeBuffer(config.topKExperts,
                                           stride: MemoryLayout<UInt32>.stride)
        self.routeWeights = try makeBuffer(config.topKExperts)
        self.greedyTokenBuffer = try makeBuffer(1, stride: MemoryLayout<UInt32>.stride)

        var caches = Array<QwenFullAttentionKVCache?>(
            repeating: nil,
            count: config.numLayers)
        for layer in 0..<config.numLayers where config.fullAttentionLayerMask[layer] != 0 {
            caches[layer] = try QwenFullAttentionKVCache(
                device: device,
                capacity: maxContext,
                geometry: QwenFullAttentionGeometry(
                    queryHeads: config.numHeads,
                    keyValueHeads: config.numFullKVHeads,
                    headDimension: config.fullHeadDim,
                    rotaryDimension: Int(Double(config.fullHeadDim)
                        * config.partialRotaryFactor),
                    ropeTheta: Float(config.fullRopeTheta)))
        }
        self.fullCaches = caches

        for layer in 0..<config.numLayers {
            _ = try model.inputNorm(layer: layer)
            _ = try model.postAttnNorm(layer: layer)
            _ = try model.qwenMoEWeights(layer: layer)
            if config.fullAttentionLayerMask[layer] != 0 {
                _ = try model.qwenFullAttentionWeights(layer: layer)
            } else {
                _ = try model.qwenDeltaNetWeights(layer: layer)
            }
        }
        _ = model.finalNorm
        _ = model.lmHead
    }

    public func reset() {
        position = 0
        promptStateSnapshot = nil
        activeDecodeDiagnostics = nil
        lastQwenDecodeDiagnostics = nil
        deltaStates.reset()
        for cache in fullCaches {
            cache?.reset()
        }
    }

    public var continuationPosition: Int { position }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition > 0, expectedPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected position \(expectedPosition), current \(position)")
        }
    }

    public func savePromptState() {
        var savedDeltaStates: [Int: QwenGatedDeltaNetSnapshot] = [:]
        var savedFullCaches: [Int: QwenFullAttentionKVSnapshot] = [:]
        for layer in 0..<config.numLayers {
            if let cache = fullCaches[layer] {
                savedFullCaches[layer] = cache.snapshot()
            } else {
                savedDeltaStates[layer] = deltaStates.state(layer: layer).snapshot()
            }
        }
        promptStateSnapshot = QwenPromptStateSnapshot(
            position: position,
            greedyToken: lastGreedyToken,
            deltaStates: savedDeltaStates,
            fullCaches: savedFullCaches)
    }

    public func restorePromptState(expectedPosition: Int) throws {
        guard let snapshot = promptStateSnapshot,
              expectedPosition > 0,
              snapshot.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "prompt replay expected position \(expectedPosition) has no matching snapshot")
        }
        for (layer, state) in snapshot.deltaStates {
            deltaStates.state(layer: layer).restore(state)
        }
        for (layer, state) in snapshot.fullCaches {
            fullCaches[layer]?.restore(state)
        }
        position = snapshot.position
        lastGreedyToken = snapshot.greedyToken
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config runtimeConfig: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        guard startPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "prefill start \(startPosition) != current position \(position)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: position, seed: .logitsWritten)
        }
        let scratchLayout = QwenPrefillScratchLayout(config: config, runtime: runtimeConfig)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              chunkTokens: scratchLayout.chunkTokens)
        if spans.count > 1 {
            var workCounter = PrefillWorkCounter()
            var finalResult: PrefillResult?
            for span in spans {
                let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
                let upper = tokens.index(lower, offsetBy: span.tokenCount)
                let result = try await prefillChunked(
                    tokens: tokens[lower..<upper],
                    startPosition: span.startPosition,
                    outputMode: outputMode,
                    config: runtimeConfig,
                    into: logits) { done in
                        onProgress(span.tokenOffset + done)
                    }
                if let work = result.work {
                    workCounter.merge(work)
                }
                finalResult = result
            }
            guard let finalResult else {
                throw PrefillError.chunkedUnsupported("Qwen prefill produced no chunks")
            }
            return PrefillResult(newPosition: finalResult.newPosition,
                                 seed: finalResult.seed,
                                 work: workCounter.diagnostics)
        }
        let scratch = try prefillScratchCache.buffers(
            device: context.device,
            layout: scratchLayout)
        let tokenIDs = scratch.tokenIDs.contents().assumingMemoryBound(to: UInt32.self)
        for (index, token) in tokens.enumerated() {
            tokenIDs[index] = UInt32(bitPattern: token)
        }

        let commandBufferStart = commandBufferSubmissionCount
        var workCounter = PrefillWorkCounter()
        let embedding = model.embedding
        try runSync { commandBuffer in
            prefillEmbed.encode(commandBuffer: commandBuffer,
                                table: embedding.buffer,
                                tableOffset: Int(embedding.offset),
                                scales: embedding.buffer,
                                scalesOffset: Int(embedding.scaleOffset),
                                biases: embedding.buffer,
                                biasesOffset: Int(embedding.biasOffset),
                                tokens: scratch.tokenIDs,
                                out: scratch.hidden,
                                t: UInt32(tokens.count),
                                d: UInt32(config.hiddenSize),
                                outScale: 1)
        }
        workCounter.recordChunkPass()

        let tokenCount = tokens.count
        for layer in 0..<config.numLayers {
            try Task.checkCancellation()
            try await encodePrefillLayer(layer: layer,
                                         scratch: scratch,
                                         tokenCount: tokenCount,
                                         startPosition: startPosition)
            workCounter.recordChunkPass()
        }

        let finalNorm = model.finalNorm
        let lmHead = model.lmHead
        let useFusedHead = useFusedGreedyHead && outputMode == .greedyIfAvailable
        try runSync { commandBuffer in
            if useFusedHead {
                fusionHead.encodeGreedyDecode(
                    commandBuffer: commandBuffer,
                    hidden: scratch.hidden,
                    hiddenOffset: (tokenCount - 1) * config.hiddenSize
                        * MemoryLayout<Float16>.stride,
                    normWeight: finalNorm.buffer,
                    normOffset: Int(finalNorm.offset),
                    weights: lmHead.buffer,
                    weightsOffset: Int(lmHead.offset),
                    scales: lmHead.buffer,
                    scalesOffset: Int(lmHead.scaleOffset),
                    biases: lmHead.buffer,
                    biasesOffset: Int(lmHead.biasOffset),
                    outToken: greedyTokenBuffer,
                    d: UInt32(config.hiddenSize),
                    vocab: UInt32(config.vocabSize))
            } else {
                prefillFinalRowHead.encodeLogits(
                    commandBuffer: commandBuffer,
                    hiddenBlock: scratch.hidden,
                    row: tokenCount - 1,
                    rowStrideElements: config.hiddenSize,
                    normWeight: finalNorm.buffer,
                    normWeightOffset: Int(finalNorm.offset),
                    weights: lmHead.buffer,
                    weightsOffset: Int(lmHead.offset),
                    scales: lmHead.buffer,
                    scalesOffset: Int(lmHead.scaleOffset),
                    biases: lmHead.buffer,
                    biasesOffset: Int(lmHead.biasOffset),
                    logits: logits,
                    d: UInt32(config.hiddenSize),
                    vocab: UInt32(config.vocabSize),
                    rmsEps: 1e-6)
            }
        }
        if useFusedHead {
            lastGreedyToken = greedyTokenBuffer.contents().load(as: UInt32.self)
        } else {
            let values = logits.contents().assumingMemoryBound(to: Float16.self)
            var bestIndex = 0
            var bestValue = values[0]
            for index in 1..<config.vocabSize where values[index] > bestValue {
                bestIndex = index
                bestValue = values[index]
            }
            lastGreedyToken = UInt32(bestIndex)
        }
        workCounter.recordChunkPass()
        position += tokenCount
        for row in 0..<tokenCount {
            onProgress(row + 1)
        }
        workCounter.recordCommandBuffers(commandBufferSubmissionCount - commandBufferStart)
        return PrefillResult(newPosition: position,
                             seed: useFusedHead ? .greedyToken(lastGreedyToken) : .logitsWritten,
                             work: workCounter.diagnostics)
    }

    public func produce(token: Int32,
                        position requestedPosition: Int,
                        into logits: MTLBuffer) async throws {
        try Task.checkCancellation()
        guard requestedPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(position) != position \(requestedPosition)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }

        let decodeStart = nowNanos()
        let commandBufferStart = commandBufferSubmissionCount
        activeDecodeDiagnostics = QwenDecodeDiagnosticsAccumulator()
        defer {
            activeDecodeDiagnostics?.wallNanos = nowNanos() - decodeStart
            activeDecodeDiagnostics?.commandBufferCount = commandBufferSubmissionCount
                - commandBufferStart
            lastQwenDecodeDiagnostics = activeDecodeDiagnostics?.makeDiagnostics()
            activeDecodeDiagnostics = nil
        }

        let embeddingStart = nowNanos()
        let embedding = model.embedding
        try runSync { commandBuffer in
            embed.encode(commandBuffer: commandBuffer,
                         table: embedding.buffer,
                         tableOffset: Int(embedding.offset),
                         scales: embedding.buffer,
                         scalesOffset: Int(embedding.scaleOffset),
                         biases: embedding.buffer,
                         biasesOffset: Int(embedding.biasOffset),
                         out: hidden,
                         tokenId: UInt32(bitPattern: token),
                         d: UInt32(config.hiddenSize),
                         outScale: 1)
        }
        activeDecodeDiagnostics?.embeddingNanos = nowNanos() - embeddingStart
        try await finishCurrentToken(into: logits, emitHead: true)
    }

    private func finishCurrentToken(into logits: MTLBuffer,
                                    emitHead: Bool) async throws {

        let layerStart = nowNanos()
        for layer in 0..<config.numLayers {
            try Task.checkCancellation()
            let layerDecodeStart = nowNanos()
            let expertFetchStart = activeDecodeDiagnostics?.expertFetchNanos ?? 0
            let expertReadCountStart = activeDecodeDiagnostics?.expertReadCount ?? 0
            let expertReadNanosStart = activeDecodeDiagnostics?.expertReadNanos ?? 0
            activeDecodeDiagnostics?.currentLayerExpertReadMaxNanos = 0
            activeDecodeDiagnostics?.currentLayerRoutedExperts = []
            activeDecodeDiagnostics?.currentLayerRoutingWeights = []
            activeDecodeDiagnostics?.layerCount += 1
            let isFullAttention = config.fullAttentionLayerMask[layer] != 0
            if isFullAttention {
                activeDecodeDiagnostics?.fullAttentionLayerCount += 1
            } else {
                activeDecodeDiagnostics?.deltaNetLayerCount += 1
            }
            try await encodeLayer(layer: layer)
            let layerExpertFetchNanos = (activeDecodeDiagnostics?.expertFetchNanos ?? 0)
                - expertFetchStart
            let layerDiagnostics = QwenDecodeLayerDiagnostics(
                layer: layer,
                isFullAttention: isFullAttention,
                elapsedNanos: nowNanos() - layerDecodeStart,
                expertFetchNanos: layerExpertFetchNanos,
                expertReadCount: (activeDecodeDiagnostics?.expertReadCount ?? 0)
                    - expertReadCountStart,
                expertReadNanos: (activeDecodeDiagnostics?.expertReadNanos ?? 0)
                    - expertReadNanosStart,
                expertReadMaxNanos: activeDecodeDiagnostics?.currentLayerExpertReadMaxNanos ?? 0,
                routedExperts: activeDecodeDiagnostics?.currentLayerRoutedExperts ?? [],
                routingWeights: activeDecodeDiagnostics?.currentLayerRoutingWeights ?? [])
            activeDecodeDiagnostics?.layers.append(layerDiagnostics)
        }
        activeDecodeDiagnostics?.layerNanos = nowNanos() - layerStart

        if emitHead {
            let logitsStart = nowNanos()
            let finalNorm = model.finalNorm
            let lmHead = model.lmHead
            try runSync { commandBuffer in
                if useFusedGreedyHead {
                    fusionHead.encodeGreedyDecode(
                        commandBuffer: commandBuffer,
                        hidden: hidden,
                        normWeight: finalNorm.buffer,
                        normOffset: Int(finalNorm.offset),
                        weights: lmHead.buffer,
                        weightsOffset: Int(lmHead.offset),
                        scales: lmHead.buffer,
                        scalesOffset: Int(lmHead.scaleOffset),
                        biases: lmHead.buffer,
                        biasesOffset: Int(lmHead.biasOffset),
                        outToken: greedyTokenBuffer,
                        d: UInt32(config.hiddenSize),
                        vocab: UInt32(config.vocabSize))
                } else {
                    prefillFinalRowHead.encodeLogits(
                        commandBuffer: commandBuffer,
                        hiddenBlock: hidden,
                        row: 0,
                        rowStrideElements: config.hiddenSize,
                        normWeight: finalNorm.buffer,
                        normWeightOffset: Int(finalNorm.offset),
                        weights: lmHead.buffer,
                        weightsOffset: Int(lmHead.offset),
                        scales: lmHead.buffer,
                        scalesOffset: Int(lmHead.scaleOffset),
                        biases: lmHead.buffer,
                        biasesOffset: Int(lmHead.biasOffset),
                        logits: logits,
                        d: UInt32(config.hiddenSize),
                        vocab: UInt32(config.vocabSize),
                        rmsEps: 1e-6)
                }
            }
            if useFusedGreedyHead {
                lastGreedyToken = greedyTokenBuffer.contents().load(as: UInt32.self)
            } else {
                let values = logits.contents().assumingMemoryBound(to: Float16.self)
                var bestIndex = 0
                var bestValue = values[0]
                for index in 1..<config.vocabSize where values[index] > bestValue {
                    bestIndex = index
                    bestValue = values[index]
                }
                lastGreedyToken = UInt32(bestIndex)
            }
            activeDecodeDiagnostics?.logitsNanos = nowNanos() - logitsStart
        }
        position += 1
    }

    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }
    public private(set) var lastGreedyToken: UInt32 = 0

    private func encodeLayer(layer: Int) async throws {
        let inputNorm = try model.inputNorm(layer: layer)
        let postAttentionNorm = try model.postAttnNorm(layer: layer)
        let moeWeights = try model.qwenMoEWeights(layer: layer)
        let router = moeWeights.router
        let isFull = config.fullAttentionLayerMask[layer] != 0
        activeDecodeDiagnostics?.routerEvaluationCount += 1
        let mixerStart = nowNanos()
        var sharedExpertStart = mixerStart
        var routerStart = mixerStart

        try runSync { commandBuffer in
            rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: hidden,
                            weight: inputNorm.buffer,
                            weightOffset: Int(inputNorm.offset),
                            out: normed,
                            d: UInt32(config.hiddenSize),
                            eps: 1e-6)
            if isFull {
                try encodeFullAttention(commandBuffer: commandBuffer, layer: layer)
            } else {
                try encodeDeltaNet(commandBuffer: commandBuffer, layer: layer)
            }
            deltaElementwise.encodeResidualAdd(commandBuffer: commandBuffer,
                                               lhs: hidden,
                                               rhs: mixerOutput,
                                               output: hidden,
                                               count: UInt32(config.hiddenSize))
            rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: hidden,
                            weight: postAttentionNorm.buffer,
                            weightOffset: Int(postAttentionNorm.offset),
                            out: normed,
                            d: UInt32(config.hiddenSize),
                            eps: 1e-6)
            sharedExpertStart = nowNanos()
            let sharedGate = sharedProjection(moeWeights.sharedExpertGate,
                                               rows: config.intermediateSize,
                                               cols: config.hiddenSize)
            let sharedUp = sharedProjection(moeWeights.sharedExpertUp,
                                             rows: config.intermediateSize,
                                             cols: config.hiddenSize)
            let sharedDown = sharedProjection(moeWeights.sharedExpertDown,
                                               rows: config.hiddenSize,
                                               cols: config.intermediateSize)
            try moe.encodeSharedExpert(commandBuffer: commandBuffer,
                                       x: normed,
                                       gate: sharedGate,
                                       up: sharedUp,
                                       down: sharedDown,
                                       y: sharedOutput,
                                       scratchGate: sharedGateScratch,
                                       scratchUp: sharedUpScratch,
                                       scratchAct: sharedActScratch)
            routerStart = nowNanos()
            moe.encodeRouter(commandBuffer: commandBuffer,
                             weights: router.buffer,
                             weightsOffset: Int(router.offset),
                             scales: router.buffer,
                             scalesOffset: Int(router.scaleOffset),
                             biases: router.buffer,
                             biasesOffset: Int(router.biasOffset),
                             hidden: normed,
                             outIndices: routeIndices,
                             outWeights: routeWeights,
                             numExperts: UInt32(config.numExperts),
                             d: UInt32(config.hiddenSize))
        }
        let moeCompleted = nowNanos()
        activeDecodeDiagnostics?.mixerNanos += moeCompleted - mixerStart
        activeDecodeDiagnostics?.sharedExpertNanos += moeCompleted - sharedExpertStart
        activeDecodeDiagnostics?.routerNanos += moeCompleted - routerStart

        try await encodeRoutedMoE(layer: layer, moeWeights: moeWeights)
    }

    private func encodePrefillLayer(layer: Int,
                                    scratch: QwenPrefillScratchBuffers,
                                    tokenCount: Int,
                                    startPosition: Int) async throws {
        let inputNorm = try model.inputNorm(layer: layer)
        let postAttentionNorm = try model.postAttnNorm(layer: layer)
        let isFull = config.fullAttentionLayerMask[layer] != 0

        try runSync { commandBuffer in
            prefillRMSNorm.encodeBF16W(
                commandBuffer: commandBuffer,
                x: scratch.hidden,
                weight: inputNorm.buffer,
                weightOffset: Int(inputNorm.offset),
                out: scratch.normed,
                t: UInt32(tokenCount),
                d: UInt32(config.hiddenSize),
                eps: 1e-6)
            if isFull {
                try encodeFullAttentionBatch(commandBuffer: commandBuffer,
                                             layer: layer,
                                             scratch: scratch,
                                             tokenCount: tokenCount,
                                             startPosition: startPosition)
            } else {
                try encodeDeltaNetBatch(commandBuffer: commandBuffer,
                                        layer: layer,
                                        scratch: scratch,
                                        tokenCount: tokenCount)
            }
            deltaElementwise.encodeResidualAddBatch(
                commandBuffer: commandBuffer,
                lhs: scratch.hidden,
                rhs: scratch.combinedOutput,
                output: scratch.hidden,
                tokenCount: UInt32(tokenCount),
                dimension: UInt32(config.hiddenSize))
            prefillRMSNorm.encodeBF16W(
                commandBuffer: commandBuffer,
                x: scratch.hidden,
                weight: postAttentionNorm.buffer,
                weightOffset: Int(postAttentionNorm.offset),
                out: scratch.normed,
                t: UInt32(tokenCount),
                d: UInt32(config.hiddenSize),
                eps: 1e-6)
        }

        try await encodePrefillMoE(layer: layer,
                                   scratch: scratch,
                                   tokenCount: tokenCount)
    }

    private func encodePrefillMoE(layer: Int,
                                  scratch: QwenPrefillScratchBuffers,
                                  tokenCount: Int) async throws {
        let moeWeights = try model.qwenMoEWeights(layer: layer)
        let router = moeWeights.router
        let sharedGate = sharedProjection(moeWeights.sharedExpertGate,
                                           rows: config.intermediateSize,
                                           cols: config.hiddenSize)
        let sharedUp = sharedProjection(moeWeights.sharedExpertUp,
                                        rows: config.intermediateSize,
                                        cols: config.hiddenSize)
        let sharedDown = sharedProjection(moeWeights.sharedExpertDown,
                                          rows: config.hiddenSize,
                                          cols: config.intermediateSize)
        let sharedRouterGate = sharedProjection(moeWeights.sharedRouterGate,
                                                rows: 1,
                                                cols: config.hiddenSize)

        try runSync { commandBuffer in
            try moe.encodeSharedExpertBlock(
                commandBuffer: commandBuffer,
                x: scratch.normed,
                y: scratch.sharedOutput,
                gate: sharedGate,
                up: sharedUp,
                down: sharedDown,
                scratchGate: scratch.sharedGateScratch,
                scratchUp: scratch.sharedUpScratch,
                scratchAct: scratch.sharedActScratch,
                queryCount: tokenCount,
                d: config.hiddenSize,
                intermediate: config.intermediateSize,
                xStrideElements: config.hiddenSize,
                yStrideElements: config.hiddenSize)
            moe.encodeRouterBlock(
                commandBuffer: commandBuffer,
                weights: router.buffer,
                weightsOffset: Int(router.offset),
                scales: router.buffer,
                scalesOffset: Int(router.scaleOffset),
                biases: router.buffer,
                biasesOffset: Int(router.biasOffset),
                hidden: scratch.normed,
                outIndices: scratch.routeIDs,
                outWeights: scratch.routeWeights,
                queryCount: UInt32(tokenCount),
                numExperts: UInt32(config.numExperts),
                d: UInt32(config.hiddenSize),
                hiddenStrideElements: UInt32(config.hiddenSize))
        }

        let routeIDs = scratch.routeIDs.contents().assumingMemoryBound(to: UInt32.self)
        let routeWeights = scratch.routeWeights.contents().assumingMemoryBound(to: Float16.self)
        let indices = Array(UnsafeBufferPointer(start: routeIDs,
                            count: tokenCount * config.topKExperts))
        let weights = Array(UnsafeBufferPointer(start: routeWeights,
                            count: tokenCount * config.topKExperts))
        let pairs = PrefillRouter.makeTokenExpertPairs(
            indices: indices,
            weights: weights,
            queryCount: tokenCount,
            topK: config.topKExperts)
        let tileExpertCount = min(16, model.routedExpertCacheSlotCount(layer: layer) ?? 16)
        guard tileExpertCount > 0 else {
            throw PrefillError.chunkedUnsupported("Qwen routed MoE has no expert-cache slots")
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: tokenCount,
            topK: config.topKExperts,
            numExperts: config.numExperts,
            tileExpertCount: tileExpertCount,
            expertSortKeys: model.routedExpertPhysicalOffsets(layer: layer))
        let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
            device: context.device,
            routes: routes)
        let offsets = model.routedExpertOffsets(layer: layer)

        for tileIndex in routes.tiles.indices {
            try Task.checkCancellation()
            let tile = routes.tiles[tileIndex]
            let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                model: model,
                layer: layer,
                tileIndex: tileIndex,
                routes: routes)
            try fetch.binding.validateCoversPairs(
                routes.sortedPairs,
                pairStart: Int(tile.pairStart),
                pairCount: Int(tile.pairCount))
            let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                device: context.device,
                binding: fetch.binding)
            let params = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart,
                pairCount: tile.pairCount,
                d: UInt32(config.hiddenSize),
                routedIntermediate: UInt32(config.moeIntermediateSize),
                topK: UInt32(config.topKExperts),
                activation: .silu,
                hiddenStrideElements: UInt32(config.hiddenSize),
                binding: fetch.binding,
                offsets: offsets)
            try withExtendedLifetime((fetch, argumentBuffer)) {
                try runSync { commandBuffer in
                    _ = prefillGroupedMoE.encodeStreamedBatched(
                        commandBuffer: commandBuffer,
                        hidden: scratch.normed,
                        sortedPairs: metadata.sortedPairs,
                        routePartials: scratch.routePartials,
                        gateUpActScratch: scratch.routedActs,
                        downScratch: scratch.routedDownScratch,
                        argumentBuffer: argumentBuffer,
                        binding: fetch.binding,
                        params: params,
                        pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                }
            }
        }

        let routedOutput = scratch.routedOutput
        try runSync { commandBuffer in
            prefillMoE.encodeReduceTokenMajor(
                commandBuffer: commandBuffer,
                routePartials: scratch.routePartials,
                routeWeights: scratch.routeWeights,
                h2: routedOutput,
                queryCount: UInt32(tokenCount),
                topK: UInt32(config.topKExperts),
                d: UInt32(config.hiddenSize))
            moe.encodeSharedGateAndCombineBlock(
                commandBuffer: commandBuffer,
                x: scratch.normed,
                gate: sharedRouterGate,
                sharedOutput: scratch.sharedOutput,
                routedOutput: routedOutput,
                y: scratch.combinedOutput,
                queryCount: tokenCount,
                d: UInt32(config.hiddenSize))
            deltaElementwise.encodeResidualAddBatch(
                commandBuffer: commandBuffer,
                lhs: scratch.hidden,
                rhs: scratch.combinedOutput,
                output: scratch.hidden,
                tokenCount: UInt32(tokenCount),
                dimension: UInt32(config.hiddenSize))
        }
    }

    private func encodeRoutedMoE(layer: Int,
                                 moeWeights: QwenMoEWeights) async throws {
        let routePlanningStart = nowNanos()
        let indices = routeIndices.contents().assumingMemoryBound(to: UInt32.self)
        let experts = (0..<config.topKExperts).map { Int(indices[$0]) }
        if activeDecodeDiagnostics != nil {
            let weights = routeWeights.contents().assumingMemoryBound(to: Float16.self)
            activeDecodeDiagnostics?.currentLayerRoutedExperts = experts
            activeDecodeDiagnostics?.currentLayerRoutingWeights =
                (0..<config.topKExperts).map { Float(weights[$0]) }
        }
        guard let plan = try model.planRoutedExperts(layer: layer, experts: experts) else {
            throw ModelError.indexCorrupt(detail: "Qwen layer \(layer) has no expert plan")
        }
        activeDecodeDiagnostics?.routedExpertCount += plan.experts.count
        activeDecodeDiagnostics?.routedExpertCacheHitCount += plan.hits
        activeDecodeDiagnostics?.routedExpertCacheMissCount += plan.misses.count
        activeDecodeDiagnostics?.routedExpertEstimatedBytes += try model.routedExpertAdviceByteEstimate(
            layer: layer,
            missCount: plan.misses.count)
        activeDecodeDiagnostics?.routePlanningNanos += nowNanos() - routePlanningStart
        let expertViews: [TensorView]
        if var diagnostics = activeDecodeDiagnostics {
            let expertFetchStart = nowNanos()
            let fetchResult = try await model.fetchRoutedExpertsWithDiagnostics(plan: plan)
            diagnostics.expertFetchNanos += nowNanos() - expertFetchStart
            diagnostics.expertReadCount += fetchResult.readDiagnostics.readCount
            diagnostics.expertReadNanos += fetchResult.readDiagnostics.totalNanos
            diagnostics.expertReadMaxNanos = max(
                diagnostics.expertReadMaxNanos,
                fetchResult.readDiagnostics.maxNanos)
            diagnostics.currentLayerExpertReadMaxNanos =
                fetchResult.readDiagnostics.maxNanos
            activeDecodeDiagnostics = diagnostics
            expertViews = fetchResult.views
        } else {
            expertViews = try await model.fetchRoutedExperts(plan: plan)
        }
        guard let argumentBuffer = moe.makeRoutedArgumentBuffer(
            routedBlobs: expertViews.map(\.buffer)) else {
            throw ModelError.residentBufferWrapFailed
        }
        let expertBuffers = expertViews.map(\.buffer)
        let offsets = model.routedExpertOffsets(layer: layer)
        let sharedGate = sharedProjection(moeWeights.sharedRouterGate,
                                           rows: 1,
                                           cols: config.hiddenSize)
        let routedExpertCombineStart = nowNanos()
        try runSync { commandBuffer in
            moe.encodeRoutedPhase1(commandBuffer: commandBuffer,
                                   routedArgBuffer: argumentBuffer,
                                   routedBlobs: expertBuffers,
                                   routedOffsets: offsets,
                                   x: normed,
                                   acts: moeActs,
                                   d: UInt32(config.hiddenSize),
                                   f: UInt32(config.moeIntermediateSize))
            moe.encodeRoutedPhase2(commandBuffer: commandBuffer,
                                   routedArgBuffer: argumentBuffer,
                                   routedBlobs: expertBuffers,
                                   routedOffsets: offsets,
                                   acts: moeActs,
                                   routingWeights: routeWeights,
                                   residual: zeroBuffer,
                                   y: routedOutput,
                                   d: UInt32(config.hiddenSize),
                                   f: UInt32(config.moeIntermediateSize))
            moe.encodeSharedGateAndCombine(commandBuffer: commandBuffer,
                                           x: normed,
                                           gate: sharedGate,
                                           sharedOutput: sharedOutput,
                                           routedOutput: routedOutput,
                                           y: combinedOutput,
                                           d: UInt32(config.hiddenSize))
            deltaElementwise.encodeResidualAdd(commandBuffer: commandBuffer,
                                               lhs: hidden,
                                               rhs: combinedOutput,
                                               output: hidden,
                                               count: UInt32(config.hiddenSize))
        }
        activeDecodeDiagnostics?.routedExpertCombineNanos += nowNanos()
            - routedExpertCombineStart
    }

    private let zeroBuffer: MTLBuffer

    private func encodeFullAttentionBatch(commandBuffer: MTLCommandBuffer,
                                          layer: Int,
                                          scratch: QwenPrefillScratchBuffers,
                                          tokenCount: Int,
                                          startPosition: Int) throws {
        let weights = try model.qwenFullAttentionWeights(layer: layer)
        let qWidth = config.numHeads * config.fullHeadDim
        let kvWidth = config.numFullKVHeads * config.fullHeadDim
        encodeProjectionBatch(commandBuffer: commandBuffer,
                               weights: weights.q,
                               input: scratch.normed,
                               output: scratch.projection,
                               tokenCount: tokenCount,
                               outputWidth: qWidth * 2)
        encodeProjectionBatch(commandBuffer: commandBuffer,
                               weights: weights.k,
                               input: scratch.normed,
                               output: scratch.key,
                               tokenCount: tokenCount,
                               outputWidth: kvWidth)
        encodeProjectionBatch(commandBuffer: commandBuffer,
                               weights: weights.v,
                               input: scratch.normed,
                               output: scratch.value,
                               tokenCount: tokenCount,
                               outputWidth: kvWidth)
        attention.encodeSplitQueryGateBatch(
            commandBuffer: commandBuffer,
            projection: scratch.projection,
            query: scratch.query,
            gate: scratch.attentionGate,
            tokenCount: UInt32(tokenCount))
        attention.encodeQueryKeyBatch(
            commandBuffer: commandBuffer,
            query: scratch.query,
            key: scratch.key,
            queryNorm: weights.qNorm.buffer,
            queryNormOffset: Int(weights.qNorm.offset),
            keyNorm: weights.kNorm.buffer,
            keyNormOffset: Int(weights.kNorm.offset),
            normalizedQuery: scratch.query,
            normalizedKey: scratch.key,
            position: UInt32(startPosition),
            tokenCount: UInt32(tokenCount),
            epsilon: 1e-6)
        guard let cache = fullCaches[layer] else {
            throw ModelError.indexCorrupt(detail: "missing full-attention cache for layer \(layer)")
        }
        guard cache.count == startPosition else {
            throw PrefillError.prefillCursorMismatch(
                "full-attention cache \(cache.count) != prefill start \(startPosition)")
        }
        cache.appendBatch(commandBuffer: commandBuffer,
                          key: scratch.key,
                          value: scratch.value,
                          tokenCount: tokenCount)
        attention.encodeBatch(commandBuffer: commandBuffer,
                              query: scratch.query,
                              cache: cache,
                              output: scratch.projection,
                              startPosition: UInt32(startPosition),
                              tokenCount: UInt32(tokenCount))
        attention.encodeOutputGateBatch(commandBuffer: commandBuffer,
                                        attention: scratch.projection,
                                        gate: scratch.attentionGate,
                                        output: scratch.projection,
                                        tokenCount: UInt32(tokenCount))
        encodeProjectionBatch(commandBuffer: commandBuffer,
                              weights: weights.o,
                              input: scratch.projection,
                              output: scratch.combinedOutput,
                              tokenCount: tokenCount,
                              outputWidth: config.hiddenSize,
                              inputWidth: qWidth)
    }

    private func encodeDeltaNetBatch(commandBuffer: MTLCommandBuffer,
                                     layer: Int,
                                     scratch: QwenPrefillScratchBuffers,
                                     tokenCount: Int) throws {
        let weights = try model.qwenDeltaNetWeights(layer: layer)
        let keyWidth = config.linearNumKeyHeads * config.linearKeyHeadDim
        let valueWidth = config.linearNumValueHeads * config.linearValueHeadDim
        encodeProjectionBatch(commandBuffer: commandBuffer,
                               weights: weights.qkv,
                               input: scratch.normed,
                               output: scratch.projection,
                               tokenCount: tokenCount,
                               outputWidth: keyWidth * 2 + valueWidth)
        encodeProjectionBatch(commandBuffer: commandBuffer,
                               weights: weights.z,
                               input: scratch.normed,
                               output: scratch.value,
                               tokenCount: tokenCount,
                               outputWidth: valueWidth)
        encodeProjectionBatch(commandBuffer: commandBuffer,
                               weights: weights.b,
                               input: scratch.normed,
                               output: scratch.key,
                               tokenCount: tokenCount,
                               outputWidth: config.linearNumValueHeads)
        encodeProjectionBatch(commandBuffer: commandBuffer,
                               weights: weights.a,
                               input: scratch.normed,
                               output: scratch.query,
                               tokenCount: tokenCount,
                               outputWidth: config.linearNumValueHeads)
        let state = deltaStates.state(layer: layer)
        prefillDeltaNet.encodeCausalConvolution(
            commandBuffer: commandBuffer,
            input: scratch.projection,
            weights: weights.convolution.buffer,
            weightsOffset: Int(weights.convolution.offset),
            output: scratch.deltaConvolution,
            state: state,
            tokenCount: UInt32(tokenCount))
        deltaElementwise.encodeDeltaParametersBatch(
            commandBuffer: commandBuffer,
            a: scratch.query,
            betaInput: scratch.key,
            aLog: weights.aLog.buffer,
            aLogOffset: Int(weights.aLog.offset),
            dtBias: weights.dtBias.buffer,
            dtBiasOffset: Int(weights.dtBias.offset),
            decay: scratch.deltaDecay,
            beta: scratch.deltaBeta,
            tokenCount: UInt32(tokenCount),
            headCount: UInt32(config.linearNumValueHeads))
        prefillDeltaNet.encodeSplitQKV(
            commandBuffer: commandBuffer,
            input: scratch.deltaConvolution,
            query: scratch.query,
            key: scratch.key,
            value: scratch.projection,
            tokenCount: UInt32(tokenCount),
            keyWidth: UInt32(keyWidth),
            valueWidth: UInt32(valueWidth))
        prefillDeltaNet.encodeRecurrent(
            commandBuffer: commandBuffer,
            query: scratch.query,
            key: scratch.key,
            value: scratch.projection,
            decay: scratch.deltaDecay,
            beta: scratch.deltaBeta,
            output: scratch.deltaConvolution,
            state: state,
            tokenCount: UInt32(tokenCount))
        deltaElementwise.encodeGatedNormBatch(
            commandBuffer: commandBuffer,
            input: scratch.deltaConvolution,
            gate: scratch.value,
            weight: weights.norm.buffer,
            weightOffset: Int(weights.norm.offset),
            output: scratch.query,
            tokenCount: UInt32(tokenCount),
            headCount: UInt32(config.linearNumValueHeads),
            headDimension: UInt32(config.linearValueHeadDim),
            epsilon: 1e-6)
        encodeProjectionBatch(commandBuffer: commandBuffer,
                               weights: weights.out,
                               input: scratch.query,
                               output: scratch.combinedOutput,
                               tokenCount: tokenCount,
                               outputWidth: config.hiddenSize,
                               inputWidth: valueWidth)
    }

    private func encodeDeltaNet(commandBuffer: MTLCommandBuffer,
                                layer: Int) throws {
        let weights = try model.qwenDeltaNetWeights(layer: layer)
        let deltaWidth = config.linearNumKeyHeads * config.linearKeyHeadDim * 2
            + config.linearNumValueHeads * config.linearValueHeadDim
        encodeProjection(commandBuffer: commandBuffer,
                         weights: weights.qkv,
                         input: normed,
                         output: qkv,
                         outputWidth: deltaWidth)
        encodeProjection(commandBuffer: commandBuffer,
                         weights: weights.z,
                         input: normed,
                         output: deltaZ,
                         outputWidth: config.linearNumValueHeads * config.linearValueHeadDim)
        encodeProjection(commandBuffer: commandBuffer,
                         weights: weights.b,
                         input: normed,
                         output: deltaB,
                         outputWidth: config.linearNumValueHeads)
        encodeProjection(commandBuffer: commandBuffer,
                         weights: weights.a,
                         input: normed,
                         output: deltaA,
                         outputWidth: config.linearNumValueHeads)
        deltaNet.encodeCausalConvolution(commandBuffer: commandBuffer,
                                         input: qkv,
                                         weights: weights.convolution.buffer,
                                         weightsOffset: Int(weights.convolution.offset),
                                         output: deltaConvolution,
                                         state: deltaStates.state(layer: layer))
        copy(commandBuffer: commandBuffer,
             source: deltaConvolution,
             sourceOffset: 0,
             destination: deltaQuery,
             size: config.linearNumKeyHeads * config.linearKeyHeadDim
                * MemoryLayout<Float16>.stride)
        copy(commandBuffer: commandBuffer,
             source: deltaConvolution,
             sourceOffset: config.linearNumKeyHeads * config.linearKeyHeadDim
                * MemoryLayout<Float16>.stride,
             destination: deltaKey,
             size: config.linearNumKeyHeads * config.linearKeyHeadDim
                * MemoryLayout<Float16>.stride)
        copy(commandBuffer: commandBuffer,
             source: deltaConvolution,
             sourceOffset: config.linearNumKeyHeads * config.linearKeyHeadDim * 2
                * MemoryLayout<Float16>.stride,
             destination: deltaValue,
             size: config.linearNumValueHeads * config.linearValueHeadDim
                * MemoryLayout<Float16>.stride)
        deltaElementwise.encodeDeltaParameters(commandBuffer: commandBuffer,
                                               a: deltaA,
                                               betaInput: deltaB,
                                               aLog: weights.aLog.buffer,
                                               aLogOffset: Int(weights.aLog.offset),
                                               dtBias: weights.dtBias.buffer,
                                               dtBiasOffset: Int(weights.dtBias.offset),
                                               decay: deltaDecay,
                                               beta: deltaBeta,
                                               count: UInt32(config.linearNumValueHeads))
        deltaNet.encodeRecurrent(commandBuffer: commandBuffer,
                                 query: deltaQuery,
                                 key: deltaKey,
                                 value: deltaValue,
                                 decay: deltaDecay,
                                 beta: deltaBeta,
                                 output: deltaOutput,
                                 state: deltaStates.state(layer: layer))
        deltaElementwise.encodeGatedNorm(commandBuffer: commandBuffer,
                                         input: deltaOutput,
                                         gate: deltaZ,
                                         weight: weights.norm.buffer,
                                         weightOffset: Int(weights.norm.offset),
                                         output: normed,
                                         headCount: UInt32(config.linearNumValueHeads),
                                         headDimension: UInt32(config.linearValueHeadDim),
                                         epsilon: 1e-6)
        encodeProjection(commandBuffer: commandBuffer,
                         weights: weights.out,
                         input: normed,
                         output: mixerOutput,
                         outputWidth: config.hiddenSize,
                         inputWidth: config.linearNumValueHeads * config.linearValueHeadDim)
    }

    private func encodeFullAttention(commandBuffer: MTLCommandBuffer,
                                     layer: Int) throws {
        let weights = try model.qwenFullAttentionWeights(layer: layer)
        let qWidth = config.numHeads * config.fullHeadDim
        let kvWidth = config.numFullKVHeads * config.fullHeadDim
        encodeProjection(commandBuffer: commandBuffer,
                         weights: weights.q,
                         input: normed,
                         output: projection,
                         outputWidth: qWidth * 2)
        encodeProjection(commandBuffer: commandBuffer,
                         weights: weights.k,
                         input: normed,
                         output: key,
                         outputWidth: kvWidth)
        encodeProjection(commandBuffer: commandBuffer,
                         weights: weights.v,
                         input: normed,
                         output: value,
                         outputWidth: kvWidth)
        attention.encodeSplitQueryGate(commandBuffer: commandBuffer,
                                       projection: projection,
                                       query: query,
                                       gate: queryGate)
        guard let cache = fullCaches[layer] else {
            throw ModelError.indexCorrupt(detail: "missing full-attention cache for layer \(layer)")
        }
        attention.encodeQueryKey(commandBuffer: commandBuffer,
                                 query: query,
                                 key: key,
                                 queryNorm: (try model.qwenFullAttentionWeights(layer: layer)).qNorm.buffer,
                                 queryNormOffset: Int((try model.qwenFullAttentionWeights(layer: layer)).qNorm.offset),
                                 keyNorm: (try model.qwenFullAttentionWeights(layer: layer)).kNorm.buffer,
                                 keyNormOffset: Int((try model.qwenFullAttentionWeights(layer: layer)).kNorm.offset),
                                 normalizedQuery: normalizedQuery,
                                 normalizedKey: normalizedKey,
                                 position: UInt32(position),
                                 epsilon: 1e-6)
        cache.append(commandBuffer: commandBuffer,
                     key: normalizedKey,
                     value: value)
        attention.encode(commandBuffer: commandBuffer,
                         query: normalizedQuery,
                         keyValueCache: cache,
                         output: attentionOutput)
        attention.encodeOutputGate(commandBuffer: commandBuffer,
                                   attention: attentionOutput,
                                   gate: queryGate,
                                   output: gatedAttention)
        encodeProjection(commandBuffer: commandBuffer,
                         weights: weights.o,
                         input: gatedAttention,
                         output: mixerOutput,
                         outputWidth: config.hiddenSize,
                         inputWidth: qWidth)
    }

    private func encodeProjection(commandBuffer: MTLCommandBuffer,
                                  weights: TensorView,
                                  input: MTLBuffer,
                                  output: MTLBuffer,
                                  outputWidth: Int,
                                  inputWidth: Int? = nil) {
        _ = prefillProjection.encode(commandBuffer: commandBuffer,
                                     weights: weights.buffer,
                                     weightsOffset: Int(weights.offset),
                                     scales: weights.buffer,
                                     scalesOffset: Int(weights.scaleOffset),
                                     biases: weights.buffer,
                                     biasesOffset: Int(weights.biasOffset),
                                     input: input,
                                     output: output,
                                     tokenCount: 1,
                                     outputWidth: outputWidth,
                                     inputWidth: inputWidth ?? config.hiddenSize)
    }

    private func encodeProjectionBatch(commandBuffer: MTLCommandBuffer,
                                       weights: TensorView,
                                       input: MTLBuffer,
                                       output: MTLBuffer,
                                       tokenCount: Int,
                                       outputWidth: Int,
                                       inputWidth: Int? = nil) {
        _ = prefillProjection.encode(commandBuffer: commandBuffer,
                                     weights: weights.buffer,
                                     weightsOffset: Int(weights.offset),
                                     scales: weights.buffer,
                                     scalesOffset: Int(weights.scaleOffset),
                                     biases: weights.buffer,
                                     biasesOffset: Int(weights.biasOffset),
                                     input: input,
                                     output: output,
                                     tokenCount: tokenCount,
                                     outputWidth: outputWidth,
                                     inputWidth: inputWidth ?? config.hiddenSize)
    }

    private func sharedProjection(_ view: TensorView,
                                  rows: Int,
                                  cols: Int) -> SharedExpertProjection {
        SharedExpertProjection(weights: view.buffer,
                               scales: view.buffer,
                               biases: view.buffer,
                               weightsOffset: Int(view.offset),
                               scalesOffset: Int(view.scaleOffset),
                               biasesOffset: Int(view.biasOffset),
                               rows: UInt32(rows),
                               cols: UInt32(cols))
    }

    private func copy(commandBuffer: MTLCommandBuffer,
                      source: MTLBuffer,
                      sourceOffset: Int,
                      destination: MTLBuffer,
                      destinationOffset: Int = 0,
                      size: Int) {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(from: source,
                  sourceOffset: sourceOffset,
                  to: destination,
                  destinationOffset: destinationOffset,
                  size: size)
        blit.endEncoding()
    }

    private func runSync(_ body: (MTLCommandBuffer) throws -> Void) throws {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try body(commandBuffer)
        commandBuffer.commit()
        commandBufferSubmissionCount += 1
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
        guard commandBuffer.status == .completed else {
            throw ModelError.residentBufferWrapFailed
        }
    }

    private func nowNanos() -> UInt64 {
        DispatchTime.now().uptimeNanoseconds
    }

}

extension QwenForwardRunner: FusedGreedyLogitProducer {}
