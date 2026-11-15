import Foundation
import Metal
import TinyTitanCore

struct PrefillAttentionParams: Sendable, Equatable {
    var startPosition: UInt32
    var queryCount: UInt32
    var headDim: UInt32
    var numQHeads: UInt32
    var numKVHeads: UInt32
    var kvValidCount: UInt32
    var slidingWindow: UInt32
    var kvTokenStrideElements: UInt32
    var qTokenStrideElements: UInt32
    var oTokenStrideElements: UInt32
    var scale: Float
    var kvBits: UInt32
    var kvTokenStrideBytes: UInt32
    var kvValueBytes: UInt32
    var kvGroupSize: UInt32

    init(
        startPosition: UInt32,
        queryCount: UInt32,
        headDim: UInt32,
        numQHeads: UInt32,
        numKVHeads: UInt32,
        kvValidCount: UInt32,
        slidingWindow: UInt32,
        kvTokenStrideElements: UInt32,
        qTokenStrideElements: UInt32,
        oTokenStrideElements: UInt32,
        scale: Float,
        kvBits: UInt32 = 16,
        kvTokenStrideBytes: UInt32 = 0,
        kvValueBytes: UInt32 = 0,
        kvGroupSize: UInt32 = UInt32(KVCacheManager.quantizationGroupSize)
    ) {
        self.startPosition = startPosition
        self.queryCount = queryCount
        self.headDim = headDim
        self.numQHeads = numQHeads
        self.numKVHeads = numKVHeads
        self.kvValidCount = kvValidCount
        self.slidingWindow = slidingWindow
        self.kvTokenStrideElements = kvTokenStrideElements
        self.qTokenStrideElements = qTokenStrideElements
        self.oTokenStrideElements = oTokenStrideElements
        self.scale = scale
        self.kvBits = kvBits
        self.kvTokenStrideBytes = kvTokenStrideBytes
        self.kvValueBytes = kvValueBytes
        self.kvGroupSize = kvGroupSize
    }
}

enum PrefillAttentionError: Error, CustomStringConvertible {
    case tensorOpsUnavailable(reason: String)
    case commandEncoderFailed

    public var description: String {
        switch self {
        case .tensorOpsUnavailable(let reason):
            return "TensorOps 2D prefill attention requested but unavailable: \(reason)"
        case .commandEncoderFailed:
            return "Failed to create Metal compute command encoder"
        }
    }
}

final class PrefillAttention {
    private let context: MetalContext
    private let psoCausalTiled: MTLComputePipelineState
    private let psoCausalQSATiled: MTLComputePipelineState?
    /// Eight queries per threadgroup, one simdgroup each: the K/V row is read
    /// once per threadgroup instead of once per query, and the reduction is a
    /// `simd_sum` with no threadgroup barrier in the key loop.
    private let psoCausalQ8: MTLComputePipelineState?
    /// One byte, bound whenever no selection is in play; `useKeep` is what
    /// actually turns the mask off.
    private let emptyKeepMask: MTLBuffer
    private let psoFullTensorOps2DValidityV2: MTLComputePipelineState?
    /// K7: recorded once at init so an explicit TensorOps path request can
    /// throw the real reason instead of a bare `preconditionFailure`.
    private let tensorOpsUnavailableReason: String
    /// K0: the capability seam and the switch are inputs, not properties of the
    /// host, so tests can drive every branch without an Apple10 machine.
    private let tensorOpsDecisionValue: TensorOpsDecision

    /// What this kernel decided about tensor operations. Test-visible, and the
    /// source of the one startup log line.
    var tensorOpsDecision: TensorOpsDecision { tensorOpsDecisionValue }
    var tensorOpsPipelineBuilt: Bool { psoFullTensorOps2DValidityV2 != nil }

    /// - Parameters:
    ///   - capabilities: what this machine can do; omitted, the live probe on
    ///     `context.device`.
    ///   - tensorOpsMode: `TINYTITAN_TENSOROPS`, already parsed; omitted, it is
    ///     read from the environment here.
    ///   - tensorOpsPipelineBuilder: how the tensor-ops pipeline is compiled.
    ///     Omitted, `context.pipeline`; a test passes a throwing builder to
    ///     prove the `on` fallback without depending on the host's shader
    ///     compiler.
    init(
        context: MetalContext,
        capabilities: HardwareCapabilities? = nil,
        tensorOpsMode: TensorOpsMode? = nil,
        tensorOpsPipelineBuilder: ((String) throws -> MTLComputePipelineState)? = nil
    ) throws {
        self.context = context
        self.psoCausalTiled = try context.pipeline("attention_prefill_causal_tiled")
        // Tile-synchronised variant: one barrier per tile of keys instead of
        // one per key. TINYTITAN_QSA_TILED=0 falls back for A/B on one build.
        self.psoCausalQSATiled =
            (try? context.pipeline(
                "attention_prefill_causal_qsa_tiled"))
        self.psoCausalQ8 =
            (try? context.pipeline("attention_prefill_causal_q8"))
        // Four bytes, not one: the kernels declare `keepIdx`/`keepIndices` as
        // `device const uint*`, and Metal's own validation aborts a binding
        // whose length is shorter than the argument it is bound to ("space for
        // 1 bytes, but argument has a length(4)"). The value is never read when
        // `useKeep` is 0, so zero is also the honest placeholder: a stray read
        // keeps nothing rather than whatever the allocator left there.
        guard
            let empty = context.device.makeBuffer(
                length: MemoryLayout<UInt32>.size, options: .storageModeShared)
        else {
            throw PrefillAttentionError.commandEncoderFailed
        }
        empty.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
        empty.label = "prefillAttention.keepMask.unused"
        self.emptyKeepMask = empty
        let liveCapabilities = capabilities ?? HardwareCapabilities.probe(device: context.device)
        let resolution = TensorOpsMode.resolve()
        let mode = tensorOpsMode ?? resolution.mode
        let note = tensorOpsMode == nil ? resolution.note : nil
        let plan = TensorOpsPolicy.plan(
            capabilities: liveCapabilities, mode: mode, note: note)
        let buildPipeline = tensorOpsPipelineBuilder ?? { try context.pipeline($0) }
        switch plan {
        case .attempt(let reason):
            do {
                self.psoFullTensorOps2DValidityV2 = try buildPipeline(
                    "attention_prefill_full_tensorops_2d_validity_v2")
                self.tensorOpsUnavailableReason = ""
                self.tensorOpsDecisionValue = TensorOpsDecision(
                    mode: mode, path: .tensorOps, reason: reason)
            } catch {
                self.psoFullTensorOps2DValidityV2 = nil
                self.tensorOpsUnavailableReason = "\(error)"
                self.tensorOpsDecisionValue = TensorOpsDecision(
                    mode: mode,
                    path: .fallback,
                    reason: "pipeline creation failed: \(error)")
            }
        case .fallback(let reason):
            self.psoFullTensorOps2DValidityV2 = nil
            self.tensorOpsUnavailableReason = reason
            self.tensorOpsDecisionValue = TensorOpsDecision(
                mode: mode, path: .fallback, reason: reason)
        }
    }

    func encodeCausal(
        commandBuffer: MTLCommandBuffer,
        q: MTLBuffer, qOffset: Int = 0,
        k: MTLBuffer, kOffset: Int = 0,
        v: MTLBuffer, vOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        params: PrefillAttentionParams,
        kvRingCapacity: UInt32 = 0,
        keepMask: MTLBuffer? = nil,
        keepStride: Int = 0,
        keepIndices: MTLBuffer? = nil,
        keepIndexStride: Int = 0,
        keepCounts: MTLBuffer? = nil,
        keepRows: MTLBuffer? = nil,
        path: RuntimePrefillAttentionPath = .causalTiled
    ) throws {
        validate(params)

        let selection = try selectPipeline(
            path: path, params: params, kvRingCapacity: kvRingCapacity, keepMask: keepMask)
        let pipeline = selection.pipeline
        let useTensorOps = selection.useTensorOps
        let useQ8 = selection.useQ8
        let headDim = Int(params.headDim)
        let threadWidth = max(1, pipeline.threadExecutionWidth)
        let threadCount =
            useTensorOps
            ? 128
            : (useQ8 ? 256 : roundUp(max(threadWidth, headDim), toMultipleOf: threadWidth))
        precondition(
            threadCount <= pipeline.maxTotalThreadsPerThreadgroup,
            "tiled prefill attention requires headDim <= maxTotalThreadsPerThreadgroup")

        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw PrefillAttentionError.commandEncoderFailed
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        // Only the tiled kernel reads the selection; the TensorOps path would
        // ignore it, which is the wrong kind of quiet for a mask.
        precondition(
            keepMask == nil || !useTensorOps,
            "sparse key selection is not implemented for the "
                + "TensorOps prefill path")
        // useKeep 2 means "the selection arrived compacted": loop over the
        // index list instead of scanning every visible key for a mask byte.
        // TINYTITAN_QSA_COMPACT=0 keeps the mask scan, so the two forms can be
        // compared on one build. They must agree token for token: the
        // compacted list is the same selection, only enumerated.
        let compactionAllowed =
            ProcessInfo.processInfo
            .environment["TINYTITAN_QSA_COMPACT"] != "0"
        let compacted =
            compactionAllowed
            && keepMask != nil && keepIndices != nil && keepCounts != nil
        // useKeep 3 is the host-KV window: the compacted list is logical keys
        // and `keepRows` is the window row for each of them.
        let windowed = compacted && keepRows != nil
        var useKeep = UInt32(
            keepMask == nil ? 0 : (windowed ? 3 : (compacted ? 2 : 1)))
        var stride = UInt32(keepStride)
        var indexStride = UInt32(keepIndexStride)
        enc.setBuffer(keepMask ?? emptyKeepMask, offset: 0, index: 5)
        enc.setBytes(&useKeep, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBuffer(keepIndices ?? emptyKeepMask, offset: 0, index: 8)
        enc.setBuffer(keepCounts ?? emptyKeepMask, offset: 0, index: 9)
        enc.setBytes(&indexStride, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBuffer(keepRows ?? emptyKeepMask, offset: 0, index: 11)
        let groups =
            useTensorOps
            ? MTLSize(
                width: Int(params.queryCount),
                height: Int(params.numQHeads) / 8,
                depth: 1)
            : (useQ8
                ? MTLSize(
                    width: (Int(params.queryCount) + 7) / 8,
                    height: Int(params.numQHeads),
                    depth: 1)
                : MTLSize(
                    width: Int(params.queryCount),
                    height: Int(params.numQHeads),
                    depth: 1))
        enc.dispatchThreadgroups(
            groups,
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Which kernel a causal dispatch runs, and the two shape flags its thread
    /// and group arithmetic needs. Extract of `encodeCausal`'s selection block:
    /// the rules, comments and precedence are unchanged.
    private struct PipelineSelection {
        let pipeline: MTLComputePipelineState
        let useTensorOps: Bool
        let useQ8: Bool
    }

    private func selectPipeline(
        path: RuntimePrefillAttentionPath,
        params: PrefillAttentionParams,
        kvRingCapacity: UInt32,
        keepMask: MTLBuffer?
    ) throws -> PipelineSelection {
        let requestsTensorOps =
            path == .fullTensorOps2DPreferred
            || path == .fullTensorOps2DValidityV2
        // The pinned model uses 512/16/2 only for full attention; its
        // sliding-window layers use 256/16/8. A future model that reuses this
        // shape for sliding attention must add a full-visibility check here.
        let tensorOpsShape =
            requestsTensorOps
            && params.kvBits == 16
            && kvRingCapacity == 0
            && params.headDim == 512
            && params.numQHeads == 16
            && params.numKVHeads == 2
            && params.scale == 1.0
        let tensorOpsPipeline = tensorOpsShape ? psoFullTensorOps2DValidityV2 : nil
        let useTensorOps = tensorOpsPipeline != nil
        // On by default since 2026-10-20: 40.51 s against 50.31 s prefill on a
        // 2.2K chunk, with a 4/4 identical answer battery. The summation order
        // differs from the scalar kernel, so TINYTITAN_PREFILL_ATTN_Q8=0 is the
        // fallback for byte-identity work.
        let useQ8 =
            keepMask == nil
            && kvRingCapacity == 0
            && params.headDim % 32 == 0
            && params.headDim / 32 <= 8
            && psoCausalQ8 != nil
            && ProcessInfo.processInfo.environment["TINYTITAN_PREFILL_ATTN_Q8"] != "0"
        let pipeline: MTLComputePipelineState
        if let tensorOpsPipeline {
            pipeline = tensorOpsPipeline
        } else if tensorOpsShape && path == .fullTensorOps2DValidityV2 {
            // K7: the caller explicitly requested the TensorOps path — fail
            // loudly with the recorded reason instead of crashing or silently
            // running a different kernel. Only auto-selected paths fall back.
            throw PrefillAttentionError.tensorOpsUnavailable(
                reason: tensorOpsUnavailableReason.isEmpty
                    ? "TensorOps pipeline failed to compile"
                    : tensorOpsUnavailableReason)
        } else {
            // Explicit mode also falls back for incompatible shapes. Benchmark
            // fixtures must use 512/16/2 to prove that TensorOps ran.
            // The tiled QSA kernel is only better when there is a selection
            // to iterate: with none, `iterations` is the whole visible range
            // and its per-tile bookkeeping buys nothing.
            let wantQSATiled =
                keepMask != nil
                && ProcessInfo.processInfo.environment["TINYTITAN_QSA_TILED"] != "0"
            if wantQSATiled, let qsa = psoCausalQSATiled, kvRingCapacity == 0 {
                pipeline = qsa
            } else if useQ8, let q8 = psoCausalQ8 {
                pipeline = q8
            } else {
                pipeline = causalTiledPipeline(kvRingCapacity: kvRingCapacity)
            }
        }
        return PipelineSelection(
            pipeline: pipeline, useTensorOps: useTensorOps, useQ8: useQ8)
    }

    private func validate(_ params: PrefillAttentionParams) {
        precondition(params.headDim > 0, "headDim must be positive")
        precondition(params.queryCount > 0, "queryCount must be positive")
        precondition(params.numQHeads > 0, "numQHeads must be positive")
        precondition(params.numKVHeads > 0, "numKVHeads must be positive")
        precondition(
            params.numQHeads % params.numKVHeads == 0,
            "numQHeads must be divisible by numKVHeads")
        precondition(
            params.qTokenStrideElements >= params.numQHeads * params.headDim,
            "q token stride is too small")
        precondition(
            params.oTokenStrideElements >= params.numQHeads * params.headDim,
            "output token stride is too small")
        if params.kvBits == 16 {
            precondition(
                params.kvTokenStrideElements >= params.numKVHeads * params.headDim,
                "KV token stride is too small")
        } else {
            precondition(
                params.kvBits == 4 || params.kvBits == 8,
                "KV bits must be 4, 8, or 16")
            precondition(
                params.kvTokenStrideBytes > 0,
                "quantized KV token stride must be positive")
        }
        precondition(
            params.startPosition + params.queryCount <= params.kvValidCount,
            "kvValidCount must include all in-flight query rows")
    }

    private func roundUp(_ value: Int, toMultipleOf multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
    }

    private func causalTiledPipeline(kvRingCapacity: UInt32) -> MTLComputePipelineState {
        guard kvRingCapacity > 0 else { return psoCausalTiled }
        do {
            return try context.pipeline(
                "attention_prefill_causal_tiled",
                constants: [MetalFunctionConstant(index: 76, value: .uint32(kvRingCapacity))])
        } catch {
            preconditionFailure("failed to build KV ring prefill attention pipeline: \(error)")
        }
    }
}
