import Foundation
import Metal
import TinyTitanCore

struct AttentionSplitGeometry: Sendable, Equatable {
    let effectiveLength: Int
    let numChunks: Int
    let chunkLength: Int
    let partialThreadgroups: Int
    let useSWAGroupedPartial: Bool
}

/// Swift wrapper for sliding-window and full-causal decode attention.
///
/// The kernels assume a single decoded query token (`M_q = 1`) and a
/// contiguous KV cache of length `seqLen`. The MPP-tensor-core prefill path
/// (`M_q > 1`) is separate.
///
/// Buffer contracts (FP16 throughout):
///   - `q`   : `[numQHeads, headDim]`
///   - `k`   : `[seqLen, numKVHeads, headDim]`
///   - `v`   : same shape as `k`. Full-layer K and V must remain distinct after
///             their separate per-head normalization and RoPE paths.
///   - `out` : `[numQHeads, headDim]`
final class Attention {
    let ctx: MetalContext
    let psoPartial: MTLComputePipelineState
    /// Host-KV streaming: pass 1 over a compact selection with an indirection
    /// row per key. The serial form backs head_dim > 256; the simd form is
    /// preferred whenever the shape allows it.
    let psoWindowPartial: MTLComputePipelineState
    let psoWindowPartialSimd: MTLComputePipelineState?
    let psoGQAPartial: MTLComputePipelineState
    let psoCombine: MTLComputePipelineState
    let psoPartialSWA: MTLComputePipelineState
    let psoPartialFull: MTLComputePipelineState
    let psoGQAPartialSWA: MTLComputePipelineState
    let psoGQAPartialSWAChunks16: MTLComputePipelineState
    let psoPartialFullChunks16: MTLComputePipelineState
    let psoCombineSWA: MTLComputePipelineState
    let psoCombineFull: MTLComputePipelineState
    let psoCombineSWAChunks16: MTLComputePipelineState
    let psoCombineFullChunks16: MTLComputePipelineState

    /// Mirrors `kAttnThreads` in `attention.metal`. The kernel was authored
    /// with a hardcoded 256-thread group so its threadgroup-memory scratch
    /// (q_smem[512] + reduce[8] + bcast) sizes are correct.
    static let threadsPerGroup: Int = 256

    /// Project ceilings for the split-KV partial scratch. `kAttnMaxHeadDim` in
    /// attention.metal is 512; `maxChunks` bounds the split factor, so the
    /// scratch is `maxQHeads * maxChunks` floats plus the same count of
    /// `maxHeadDim` output partials.
    ///
    /// The head ceiling has grown with the families: Qwen 3.6 has 16, Qwen3.8
    /// has 24, and Laguna-S-2.1's sliding-window layers run 72 against 8 KV
    /// heads. At 96 the scratch is about 12 MB, allocated once per `Attention`;
    /// the kernels read the real head count from their arguments.
    static let maxQHeads = 96
    static let maxHeadDim = 512
    static let maxChunks = 64
    /// Full attention uses 16 base chunks by default.
    private static let defaultFullChunks = 16
    private static let defaultGQASWAChunks = 8

    // Partial state written by pass 1, read by pass 2. One shared allocation:
    // attention runs once per layer, serially, and pass 2 hazard-tracks pass 1
    // within the same command buffer — no race (mirrors MoE.routerLogits).
    let mPartial: MTLBuffer
    let dPartial: MTLBuffer
    let oPartial: MTLBuffer

    init(context: MetalContext) throws {
        self.ctx = context
        self.psoPartial = try context.pipeline("attention_decode_partial")
        self.psoWindowPartial = try context.pipeline(
            "attention_decode_window_partial")
        self.psoWindowPartialSimd = try? context.pipeline(
            "attention_decode_window_partial_simd")
        self.psoGQAPartial = try context.pipeline("attention_decode_gqa_swa_partial")
        self.psoCombine = try context.pipeline("attention_decode_combine")
        self.psoPartialSWA = try Self.specializedPipeline(
            context,
            "attention_decode_partial",
            headDim: 256,
            numQHeads: 16,
            numKVHeads: 8)
        self.psoPartialFull = try Self.specializedPipeline(
            context,
            "attention_decode_partial",
            headDim: 512,
            numQHeads: 16,
            numKVHeads: 2)
        self.psoGQAPartialSWA = try Self.specializedPipeline(
            context,
            "attention_decode_gqa_swa_partial",
            headDim: 256,
            numQHeads: 16,
            numKVHeads: 8)
        self.psoGQAPartialSWAChunks16 = try Self.specializedPipeline(
            context,
            "attention_decode_gqa_swa_partial",
            headDim: 256,
            numQHeads: 16,
            numKVHeads: 8,
            numChunks: 16)
        self.psoPartialFullChunks16 = try Self.specializedPipeline(
            context,
            "attention_decode_partial",
            headDim: 512,
            numQHeads: 16,
            numKVHeads: 2,
            numChunks: 16)
        self.psoCombineSWA = try Self.specializedPipeline(
            context,
            "attention_decode_combine",
            headDim: 256,
            numQHeads: 16,
            numKVHeads: 8)
        self.psoCombineFull = try Self.specializedPipeline(
            context,
            "attention_decode_combine",
            headDim: 512,
            numQHeads: 16,
            numKVHeads: 2)
        self.psoCombineSWAChunks16 = try Self.specializedPipeline(
            context,
            "attention_decode_combine",
            headDim: 256,
            numQHeads: 16,
            numKVHeads: 8,
            numChunks: 16)
        self.psoCombineFullChunks16 = try Self.specializedPipeline(
            context,
            "attention_decode_combine",
            headDim: 512,
            numQHeads: 16,
            numKVHeads: 2,
            numChunks: 16)
        let md = Self.maxQHeads * Self.maxChunks
        guard
            let m = context.device.makeBuffer(
                length: md * MemoryLayout<Float>.size,
                options: .storageModeShared),
            let d = context.device.makeBuffer(
                length: md * MemoryLayout<Float>.size,
                options: .storageModeShared),
            let o = context.device.makeBuffer(
                length: md * Self.maxHeadDim * MemoryLayout<Float>.size,
                options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("attention split-KV scratch")
        }
        self.mPartial = m
        self.dPartial = d
        self.oPartial = o
        self.splitStateLock = NSLock()
        // A `uint` placeholder, not a byte: the decode kernels bind this where
        // they declare `device const uint*`, which is what the prefill side's
        // comment explains at length.
        guard
            let empty = context.device.makeBuffer(
                length: MemoryLayout<UInt32>.size, options: .storageModeShared)
        else {
            throw MetalError.bufferAllocationFailed("attention keep-mask placeholder")
        }
        empty.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
        empty.label = "attention.keepMask.unused"
        self.emptyKeepMask = empty
        self.splitInFlight = false
    }

    // K15: the split-KV partial scratch (mPartial/dPartial/oPartial) is shared
    // instance state written by pass 1 and read by pass 2. The runtime encodes
    // attention strictly serially — one layer at a time, one command buffer —
    // so no two encodeSplit calls may be in flight. Guard that contract at
    // runtime: a reentrant encodeSplit would corrupt pass-1 state and is a
    // programming error, so it throws loudly instead of silently corrupting.
    let splitStateLock: NSLock
    /// Simdgroup-per-key pass 1 (attention_decode_partial_simd), used when a
    /// sparse selection is in play. Built per (head_dim, heads, chunks) on
    /// first use. TINYTITAN_ATTN_SIMD_PARTIAL=0 keeps the serial kernel.
    var simdPartialCache: [String: MTLComputePipelineState] = [:]
    /// Test hook: forces the simd (true) or serial (false) pass 1.
    var simdPartialOverride: Bool?
    private static let simdPartialDefault =
        ProcessInfo.processInfo.environment["TINYTITAN_ATTN_SIMD_PARTIAL"] != "0"
    var simdPartialEnabled: Bool { simdPartialOverride ?? Self.simdPartialDefault }
    /// One byte, bound whenever no selection is in play: Metal requires the
    /// argument, and `use_keep` is what actually turns the mask off.
    let emptyKeepMask: MTLBuffer
    var splitInFlight: Bool

    /// Number of K/V chunks for a range of `effLen` positions — the split
    /// factor used by the production split path.
    static func chunkCount(effLen: Int, preferGQASWA: Bool = false) -> Int {
        let eff = max(1, effLen)
        let defaultChunks = preferGQASWA ? defaultGQASWAChunks : defaultFullChunks
        return max(1, min(defaultChunks, min(maxChunks, eff)))
    }

    static func splitGeometry(
        numQHeads: UInt32,
        numKVHeads: UInt32,
        seqLen: UInt32,
        kvStart: UInt32,
        preferGQASWA: Bool
    ) -> AttentionSplitGeometry {
        let qPerKV = Int(numQHeads / numKVHeads)
        let useSWAGQAPartial = preferGQASWA && qPerKV <= 2
        let effectiveLength = Int(seqLen) - Int(kvStart)
        let baseChunks = Self.chunkCount(
            effLen: effectiveLength,
            preferGQASWA: useSWAGQAPartial)
        let numChunks =
            useSWAGQAPartial
            ? max(baseChunks, min(Self.maxChunks, baseChunks * qPerKV))
            : baseChunks
        let chunkLength = (max(1, effectiveLength) + numChunks - 1) / numChunks
        let partialHeadGroups = useSWAGQAPartial ? Int(numKVHeads) : Int(numQHeads)
        return AttentionSplitGeometry(
            effectiveLength: effectiveLength,
            numChunks: numChunks,
            chunkLength: chunkLength,
            partialThreadgroups: partialHeadGroups * numChunks,
            useSWAGroupedPartial: useSWAGQAPartial)
    }

    /// Sliding-window attention. `window` caps the K/V positions to the most
    /// recent `window` entries (`[max(0, seqLen-window), seqLen)`).
    /// `scale` defaults to `rsqrt(head_dim)` for generic callers;
    /// callers with a configured attention scale pass it explicitly.
    func encodeSWA(
        commandBuffer: MTLCommandBuffer,
        q: MTLBuffer, qOffset: Int = 0,
        k: MTLBuffer, kOffset: Int = 0,
        v: MTLBuffer, vOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        headDim: UInt32,
        numQHeads: UInt32,
        numKVHeads: UInt32,
        seqLen: UInt32,
        window: UInt32,
        scale: Float? = nil,
        ringCapacity: UInt32 = 0,
        kvFormat: KVView? = nil
    ) throws {
        precondition(
            numQHeads % numKVHeads == 0,
            "numQHeads must be a multiple of numKVHeads for GQA")
        precondition(
            headDim <= 512,
            "head_dim must be <= 512 (kernel scratch is sized for the full-attn case)")
        let sc = scale ?? Self.defaultScale(headDim: headDim)
        let kvStart = seqLen > window ? seqLen - window : 0

        try encodeSplit(
            commandBuffer: commandBuffer,
            q: q, qOffset: qOffset, k: k, kOffset: kOffset,
            v: v, vOffset: vOffset, out: out, outOffset: outOffset,
            headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
            seqLen: seqLen, kvStart: kvStart, scale: sc,
            preferGQASWA: true,
            ringCapacity: ringCapacity,
            kvFormat: kvFormat)
    }

    /// Full attention. Separate normalization and RoPE make the cache streams
    /// distinct here. `scale` mirrors `encodeSWA`.
    func encodeFull(
        commandBuffer: MTLCommandBuffer,
        q: MTLBuffer, qOffset: Int = 0,
        k: MTLBuffer, kOffset: Int = 0,
        v: MTLBuffer, vOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        headDim: UInt32,
        numQHeads: UInt32,
        numKVHeads: UInt32,
        seqLen: UInt32,
        scale: Float? = nil,
        kvFormat: KVView? = nil,
        keepMask: MTLBuffer? = nil,
        windowSelection: KVWindowSelection? = nil
    ) throws {
        precondition(
            numQHeads % numKVHeads == 0,
            "numQHeads must be a multiple of numKVHeads for GQA")
        precondition(
            headDim <= 512,
            "head_dim must be <= 512 (kernel scratch is sized for the full-attn case)")
        precondition(
            seqLen > 0 || windowSelection != nil,
            "full attention requires at least one KV position")
        let sc = scale ?? Self.defaultScale(headDim: headDim)

        try encodeSplit(
            commandBuffer: commandBuffer,
            q: q, qOffset: qOffset, k: k, kOffset: kOffset,
            v: v, vOffset: vOffset, out: out, outOffset: outOffset,
            headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
            seqLen: windowSelection.map { UInt32($0.count) } ?? seqLen,
            kvStart: 0, scale: sc,
            preferGQASWA: false,
            kvFormat: kvFormat,
            keepMask: keepMask,
            windowSelection: windowSelection)
    }

    /// Two-pass split-KV (Flash-Decoding) dispatch shared by SWA and full
    /// attention — they differ only by `kvStart`. Pass 1 fans the head's
    /// `[kvStart, seqLen)` range across `chunkCount` threadgroups per head;
    /// pass 2 merges the partials. Both encoders go on the same command buffer
    /// so pass 2 hazard-tracks the partial scratch written by pass 1.
    private func encodeSplit(
        commandBuffer: MTLCommandBuffer,
        q: MTLBuffer, qOffset: Int,
        k: MTLBuffer, kOffset: Int,
        v: MTLBuffer, vOffset: Int,
        out: MTLBuffer, outOffset: Int,
        headDim: UInt32, numQHeads: UInt32, numKVHeads: UInt32,
        seqLen: UInt32, kvStart: UInt32, scale: Float,
        preferGQASWA: Bool,
        ringCapacity: UInt32 = 0,
        kvFormat: KVView? = nil,
        keepMask: MTLBuffer? = nil,
        windowSelection: KVWindowSelection? = nil
    ) throws {
        precondition(
            Int(numQHeads) <= Self.maxQHeads,
            "numQHeads \(numQHeads) exceeds split-KV scratch (max \(Self.maxQHeads))")
        precondition(
            Int(headDim) <= Self.maxHeadDim,
            "head_dim \(headDim) exceeds split-KV scratch (max \(Self.maxHeadDim))")
        precondition(
            ringCapacity == 0 || preferGQASWA,
            "KV ring is only valid for SWA attention")
        // Only the full-attention kernel reads the selection. Binding it for
        // the GQA/SWA path would be silently ignored, which is the wrong kind
        // of quiet for a mask whose whole job is to change what is attended.
        precondition(
            keepMask == nil || !preferGQASWA,
            "sparse key selection is not implemented for the GQA/SWA path")
        precondition(
            windowSelection == nil || (keepMask == nil && !preferGQASWA),
            "a windowed selection replaces the keep mask and is full-attention only")
        splitStateLock.lock()
        let reentered = splitInFlight
        splitInFlight = true
        splitStateLock.unlock()
        defer {
            splitStateLock.lock()
            splitInFlight = false
            splitStateLock.unlock()
        }
        guard !reentered else {
            throw MetalError.invalidState(
                "encodeSplit re-entered while the previous split-KV pass was in flight; attention must encode serially per layer"
            )
        }
        let geometry = Self.splitGeometry(
            numQHeads: numQHeads,
            numKVHeads: numKVHeads,
            seqLen: windowSelection.map { UInt32($0.count) } ?? seqLen,
            kvStart: kvStart,
            preferGQASWA: preferGQASWA)
        let useSWAGQAPartial = geometry.useSWAGroupedPartial
        let nChunks = geometry.numChunks
        let chunkLen = geometry.chunkLength
        var partialPSO: MTLComputePipelineState
        if windowSelection != nil {
            if headDim % 32 == 0, headDim <= 256, simdPartialEnabled,
                let simd = psoWindowPartialSimd
            {
                partialPSO = simd
            } else {
                partialPSO = psoWindowPartial
            }
        } else {
            partialPSO = partialPipeline(
                headDim: headDim,
                numQHeads: numQHeads,
                numKVHeads: numKVHeads,
                numChunks: nChunks,
                useGQAPartial: useSWAGQAPartial,
                ringCapacity: ringCapacity)
            if keepMask != nil, !useSWAGQAPartial, ringCapacity == 0,
                headDim % 32 == 0, headDim <= 256,
                simdPartialEnabled,
                let simd = simdPartialPipeline(
                    headDim: headDim, numQHeads: numQHeads,
                    numKVHeads: numKVHeads, numChunks: nChunks)
            {
                partialPSO = simd
            }
        }
        let tgWidth = min(Self.threadsPerGroup, Int(partialPSO.maxTotalThreadsPerThreadgroup))

        guard let p1 = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        p1.setComputePipelineState(partialPSO)
        p1.setBuffer(q, offset: qOffset, index: 0)
        p1.setBuffer(k, offset: kOffset, index: 1)
        p1.setBuffer(v, offset: vOffset, index: 2)
        p1.setBuffer(mPartial, offset: 0, index: 3)
        p1.setBuffer(dPartial, offset: 0, index: 4)
        p1.setBuffer(oPartial, offset: 0, index: 5)
        var hd = headDim
        var nq = numQHeads
        var nkv = numKVHeads
        var sl = seqLen
        var ks = kvStart
        var cl = UInt32(chunkLen)
        var nc = UInt32(nChunks)
        var sc = scale
        p1.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 6)
        p1.setBytes(&nq, length: MemoryLayout<UInt32>.size, index: 7)
        p1.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 8)
        p1.setBytes(&sl, length: MemoryLayout<UInt32>.size, index: 9)
        p1.setBytes(&ks, length: MemoryLayout<UInt32>.size, index: 10)
        p1.setBytes(&cl, length: MemoryLayout<UInt32>.size, index: 11)
        p1.setBytes(&nc, length: MemoryLayout<UInt32>.size, index: 12)
        p1.setBytes(&sc, length: MemoryLayout<Float>.size, index: 13)
        var kvBits = UInt32(kvFormat?.precision.rawValue ?? 16)
        var kvStride = UInt32(kvFormat?.stride ?? 0)
        var kvValueBytes = UInt32(kvFormat?.valueBytes ?? 0)
        var kvGroupSize = UInt32(kvFormat?.groupSize ?? KVCacheManager.quantizationGroupSize)
        p1.setBytes(&kvBits, length: MemoryLayout<UInt32>.size, index: 14)
        p1.setBytes(&kvStride, length: MemoryLayout<UInt32>.size, index: 15)
        p1.setBytes(&kvValueBytes, length: MemoryLayout<UInt32>.size, index: 16)
        p1.setBytes(&kvGroupSize, length: MemoryLayout<UInt32>.size, index: 17)
        if let windowSelection {
            p1.setBuffer(windowSelection.indices, offset: 0, index: 18)
            p1.setBuffer(windowSelection.rows, offset: 0, index: 19)
        } else {
            var useKeep = UInt32(keepMask == nil ? 0 : 1)
            p1.setBuffer(keepMask ?? emptyKeepMask, offset: 0, index: 18)
            p1.setBytes(&useKeep, length: MemoryLayout<UInt32>.size, index: 19)
        }
        let partialGroups = geometry.partialThreadgroups
        p1.dispatchThreadgroups(
            MTLSize(width: partialGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        p1.endEncoding()

        guard let p2 = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        let combinePSO = combinePipeline(
            headDim: headDim,
            numQHeads: numQHeads,
            numKVHeads: numKVHeads,
            numChunks: nChunks)
        p2.setComputePipelineState(combinePSO)
        p2.setBuffer(mPartial, offset: 0, index: 0)
        p2.setBuffer(dPartial, offset: 0, index: 1)
        p2.setBuffer(oPartial, offset: 0, index: 2)
        p2.setBuffer(out, offset: outOffset, index: 3)
        var hd2 = headDim
        var nc2 = UInt32(nChunks)
        p2.setBytes(&hd2, length: MemoryLayout<UInt32>.size, index: 4)
        p2.setBytes(&nc2, length: MemoryLayout<UInt32>.size, index: 5)
        let combineTGWidth = min(
            Self.threadsPerGroup,
            Int(combinePSO.maxTotalThreadsPerThreadgroup))
        p2.dispatchThreadgroups(
            MTLSize(width: Int(numQHeads), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: combineTGWidth, height: 1, depth: 1))
        p2.endEncoding()
    }

    /// `1 / sqrt(head_dim)` — the classic transformer scaling. Used as the
    /// default for callers without a configured attention scale (and for the
    /// existing tests that pre-date the runtime scale arg).
    static func defaultScale(headDim: UInt32) -> Float {
        Float(1.0) / Float(headDim).squareRoot()
    }

}
