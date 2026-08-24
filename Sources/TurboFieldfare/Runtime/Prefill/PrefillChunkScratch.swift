import Metal

struct PrefillChunkScratchLayout: Sendable, Equatable {
    let chunkTokens: Int
    let hiddenSize: Int
    let maxQElementsPerToken: Int
    let maxKVElementsPerToken: Int
    let sharedIntermediate: Int
    let routedIntermediate: Int
    let topK: Int
    let routedPairMicrobatchRows: Int

    init(config: ArchConfig,
                chunkTokens: Int,
                routedPairMicrobatchRows: Int = 32,
                chunkTokenLimit: Int = PrefillRuntimeConfig.maxChunkTokens) {
        self.chunkTokens = max(1, min(chunkTokens, chunkTokenLimit))
        self.hiddenSize = config.hiddenSize
        self.maxQElementsPerToken = config.numHeads * max(config.headDim, config.fullHeadDim)
        self.maxKVElementsPerToken = max(config.numKVHeads * config.headDim,
                                         config.numFullKVHeads * config.fullHeadDim)
        self.sharedIntermediate = config.intermediateSize
        self.routedIntermediate = config.moeIntermediateSize
        self.topK = config.topKExperts
        self.routedPairMicrobatchRows = max(1, min(routedPairMicrobatchRows, 128))
    }


    var hiddenElements: Int { chunkTokens * hiddenSize }
    var normedElements: Int { hiddenElements }
    var qElements: Int { chunkTokens * maxQElementsPerToken }
    var kStageElements: Int { chunkTokens * maxKVElementsPerToken }
    var vStageElements: Int { kStageElements }
    var attentionOutputElements: Int { qElements }
    var denseXElements: Int { hiddenElements }
    var routedXElements: Int { hiddenElements }
    var routerXElements: Int { hiddenElements }
    var h1Elements: Int { hiddenElements }
    var h2Elements: Int { hiddenElements }
    var routePartialElements: Int { chunkTokens * topK * hiddenSize }
    var routeIDElements: Int { chunkTokens * topK }
    var routeWeightElements: Int { routeIDElements }
    var sharedExpertScratchElements: Int { sharedIntermediate }
    var routedGateUpActElements: Int { 3 * routedPairMicrobatchRows * routedIntermediate }
    var routedDownOutputElements: Int { routedPairMicrobatchRows * hiddenSize }

    var devicePrivateBytes: Int {
        let fp16Elements = hiddenElements
            + normedElements
            + qElements
            + kStageElements
            + vStageElements
            + attentionOutputElements
            + denseXElements
            + routedXElements
            + routerXElements
            + h1Elements
            + h2Elements
            + routePartialElements
            + 3 * sharedExpertScratchElements
            + routedGateUpActElements
            + routedDownOutputElements
        return fp16Elements * MemoryLayout<Float16>.stride
    }

    var sharedMetadataBytes: Int {
        routeIDElements * MemoryLayout<UInt32>.stride
            + routeWeightElements * MemoryLayout<Float16>.stride
    }

    var totalPersistentBytes: Int {
        devicePrivateBytes + sharedMetadataBytes
    }
}

struct PrefillChunkScratchBuffers {
    let layout: PrefillChunkScratchLayout
    let hidden: MTLBuffer
    let normed: MTLBuffer
    let q: MTLBuffer
    let kStage: MTLBuffer
    let vStage: MTLBuffer
    let attentionOutput: MTLBuffer
    let denseX: MTLBuffer
    let routedX: MTLBuffer
    let routerX: MTLBuffer
    let h1: MTLBuffer
    let h2: MTLBuffer
    let routePartials: MTLBuffer
    let routeIDs: MTLBuffer
    let routeWeights: MTLBuffer
    let sharedGateScratch: MTLBuffer
    let sharedUpScratch: MTLBuffer
    let sharedActScratch: MTLBuffer
    let routedGateUpActScratch: MTLBuffer
    let routedDownScratch: MTLBuffer

    static func allocate(device: MTLDevice,
                         layout: PrefillChunkScratchLayout) throws -> PrefillChunkScratchBuffers {
        func privateBuffer(_ elements: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * MemoryLayout<Float16>.stride,
                options: .storageModePrivate)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        func sharedBuffer(_ bytes: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(bytes, 1),
                                                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        return PrefillChunkScratchBuffers(
            layout: layout,
            hidden: try privateBuffer(layout.hiddenElements, label: "prefill.hidden"),
            normed: try privateBuffer(layout.normedElements, label: "prefill.normed"),
            q: try privateBuffer(layout.qElements, label: "prefill.q"),
            kStage: try privateBuffer(layout.kStageElements, label: "prefill.kStage"),
            vStage: try privateBuffer(layout.vStageElements, label: "prefill.vStage"),
            attentionOutput: try privateBuffer(layout.attentionOutputElements, label: "prefill.attnOut"),
            denseX: try privateBuffer(layout.denseXElements, label: "prefill.denseX"),
            routedX: try privateBuffer(layout.routedXElements, label: "prefill.routedX"),
            routerX: try privateBuffer(layout.routerXElements, label: "prefill.routerX"),
            h1: try privateBuffer(layout.h1Elements, label: "prefill.h1"),
            h2: try privateBuffer(layout.h2Elements, label: "prefill.h2"),
            routePartials: try privateBuffer(layout.routePartialElements, label: "prefill.routePartials"),
            routeIDs: try sharedBuffer(layout.routeIDElements * MemoryLayout<UInt32>.stride,
                                       label: "prefill.routeIDs"),
            routeWeights: try sharedBuffer(layout.routeWeightElements * MemoryLayout<Float16>.stride,
                                           label: "prefill.routeWeights"),
            sharedGateScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                                 label: "prefill.sharedGateScratch"),
            sharedUpScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                               label: "prefill.sharedUpScratch"),
            sharedActScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                                label: "prefill.sharedActScratch"),
            routedGateUpActScratch: try privateBuffer(layout.routedGateUpActElements,
                                                      label: "prefill.routedGateUpActScratch"),
            routedDownScratch: try privateBuffer(layout.routedDownOutputElements,
                                                 label: "prefill.routedDownScratch"))
    }
}

struct QwenPrefillScratchLayout: Sendable, Equatable {
    let chunkTokens: Int
    let hiddenSize: Int
    let projectionElementsPerToken: Int
    let queryElementsPerToken: Int
    let keyElementsPerToken: Int
    let valueElementsPerToken: Int
    let linearValueHeads: Int
    let sharedIntermediate: Int
    let routedIntermediate: Int
    let topK: Int
    let routedPairMicrobatchRows: Int

    init(config: ArchConfig, runtime: PrefillRuntimeConfig) {
        let qWidth = config.numHeads * config.fullHeadDim
        let kvWidth = config.numFullKVHeads * config.fullHeadDim
        let deltaKeyWidth = config.linearNumKeyHeads * config.linearKeyHeadDim
        let deltaValueWidth = config.linearNumValueHeads * config.linearValueHeadDim
        self.chunkTokens = max(1, min(runtime.chunkTokens, PrefillRuntimeConfig.maxChunkTokens))
        self.hiddenSize = config.hiddenSize
        self.projectionElementsPerToken = max(qWidth * 2, deltaKeyWidth * 2 + deltaValueWidth)
        self.queryElementsPerToken = max(qWidth, deltaKeyWidth)
        self.keyElementsPerToken = max(kvWidth, deltaKeyWidth)
        self.valueElementsPerToken = max(kvWidth, deltaValueWidth)
        self.linearValueHeads = config.linearNumValueHeads
        self.sharedIntermediate = config.intermediateSize
        self.routedIntermediate = config.moeIntermediateSize
        self.topK = config.topKExperts
        self.routedPairMicrobatchRows = 32
    }

    var hiddenElements: Int { chunkTokens * hiddenSize }
    var normedElements: Int { hiddenElements }
    var projectionElements: Int { chunkTokens * projectionElementsPerToken }
    var queryElements: Int { chunkTokens * queryElementsPerToken }
    var keyElements: Int { chunkTokens * keyElementsPerToken }
    var valueElements: Int { chunkTokens * valueElementsPerToken }
    var deltaConvolutionElements: Int { projectionElements }
    var deltaParameterElements: Int { chunkTokens * linearValueHeads }
    var attentionGateElements: Int { queryElements }
    var tokenIDElements: Int { chunkTokens }
    var routeElements: Int { chunkTokens * topK }
    var routePartialElements: Int { routeElements * hiddenSize }
    var sharedExpertScratchElements: Int { chunkTokens * sharedIntermediate }
    var routedExpertActElements: Int { 3 * routedPairMicrobatchRows * routedIntermediate }
    var routedDownScratchElements: Int { routedPairMicrobatchRows * hiddenSize }
    var expertOutputElements: Int { hiddenElements }
}

struct QwenPrefillScratchBuffers {
    let layout: QwenPrefillScratchLayout
    let tokenIDs: MTLBuffer
    let hidden: MTLBuffer
    let normed: MTLBuffer
    let projection: MTLBuffer
    let query: MTLBuffer
    let key: MTLBuffer
    let value: MTLBuffer
    let deltaConvolution: MTLBuffer
    let deltaDecay: MTLBuffer
    let deltaBeta: MTLBuffer
    let attentionGate: MTLBuffer
    let routeIDs: MTLBuffer
    let routeWeights: MTLBuffer
    let routePartials: MTLBuffer
    let sharedGateScratch: MTLBuffer
    let sharedUpScratch: MTLBuffer
    let sharedActScratch: MTLBuffer
    let routedActs: MTLBuffer
    let routedDownScratch: MTLBuffer
    let sharedOutput: MTLBuffer
    let routedOutput: MTLBuffer
    let combinedOutput: MTLBuffer

    static func allocate(device: MTLDevice,
                         layout: QwenPrefillScratchLayout) throws -> QwenPrefillScratchBuffers {
        func privateBuffer(_ elements: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * MemoryLayout<Float16>.stride,
                options: .storageModePrivate) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        func sharedBuffer(_ bytes: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(bytes, 1),
                                                 options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        func floatBuffer(_ elements: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * MemoryLayout<Float>.stride,
                options: .storageModePrivate) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        return QwenPrefillScratchBuffers(
            layout: layout,
            tokenIDs: try sharedBuffer(layout.tokenIDElements * MemoryLayout<UInt32>.stride,
                                       label: "qwen.prefill.tokenIDs"),
            hidden: try privateBuffer(layout.hiddenElements, label: "qwen.prefill.hidden"),
            normed: try privateBuffer(layout.normedElements, label: "qwen.prefill.normed"),
            projection: try privateBuffer(layout.projectionElements, label: "qwen.prefill.projection"),
            query: try privateBuffer(layout.queryElements, label: "qwen.prefill.query"),
            key: try privateBuffer(layout.keyElements, label: "qwen.prefill.key"),
            value: try privateBuffer(layout.valueElements, label: "qwen.prefill.value"),
            deltaConvolution: try privateBuffer(layout.deltaConvolutionElements,
                                                label: "qwen.prefill.deltaConvolution"),
            deltaDecay: try floatBuffer(layout.deltaParameterElements,
                                        label: "qwen.prefill.deltaDecay"),
            deltaBeta: try floatBuffer(layout.deltaParameterElements,
                                       label: "qwen.prefill.deltaBeta"),
            attentionGate: try privateBuffer(layout.attentionGateElements,
                                             label: "qwen.prefill.attentionGate"),
            routeIDs: try sharedBuffer(layout.routeElements * MemoryLayout<UInt32>.stride,
                                       label: "qwen.prefill.routeIDs"),
            routeWeights: try sharedBuffer(layout.routeElements * MemoryLayout<Float16>.stride,
                                           label: "qwen.prefill.routeWeights"),
            routePartials: try privateBuffer(layout.routePartialElements,
                                              label: "qwen.prefill.routePartials"),
            sharedGateScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                                  label: "qwen.prefill.sharedGateScratch"),
            sharedUpScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                                label: "qwen.prefill.sharedUpScratch"),
            sharedActScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                                label: "qwen.prefill.sharedActScratch"),
            routedActs: try privateBuffer(layout.routedExpertActElements,
                                          label: "qwen.prefill.routedActs"),
            routedDownScratch: try privateBuffer(layout.routedDownScratchElements,
                                                 label: "qwen.prefill.routedDownScratch"),
            sharedOutput: try privateBuffer(layout.expertOutputElements,
                                            label: "qwen.prefill.sharedOutput"),
            routedOutput: try privateBuffer(layout.expertOutputElements,
                                            label: "qwen.prefill.routedOutput"),
            combinedOutput: try privateBuffer(layout.expertOutputElements,
                                              label: "qwen.prefill.combinedOutput"))
    }
}

final class QwenPrefillScratchCache {
    private var cached: QwenPrefillScratchBuffers?

    func buffers(device: MTLDevice,
                 layout: QwenPrefillScratchLayout) throws -> QwenPrefillScratchBuffers {
        if let cached, cached.layout == layout {
            return cached
        }
        let scratch = try QwenPrefillScratchBuffers.allocate(device: device, layout: layout)
        cached = scratch
        return scratch
    }
}
