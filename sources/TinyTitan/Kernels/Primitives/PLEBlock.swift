import Foundation
import Metal

/// Qwen3.8-Flash-Next's per-layer n-gram embedding block (layer 1, 0-based).
///
/// The block runs *before* the attention read gate and rewrites the wide
/// residual in place. Given the rows this token's n-gram context hashes to
/// (see `PLEHash`) and their gathered embeddings (`NgramTableReader`), it
/// produces two terms and adds both:
///
///     key        = keyProj   @ emb                    // -> streams * dim
///     value      = valueProj @ emb                    // -> dim
///     key_w      = groupedRMSNorm(key,     normKey)
///     query      = groupedRMSNorm(streams, normQuery)
///     s[t]       = <key_w[t], query[t]> / sqrt(dim)    // one per stream
///     gate       = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))
///     gated      = value (x) gate                      // -> streams * dim
///     normalized = groupedRMSNorm(gated, normConv)
///     conv       = dilated depthwise causal conv over time, K taps, dil = N
///     streams   += gated + silu(conv)
///
/// Two details here are silent if wrong, so they are stated rather than
/// implied: `gated` is added *unnormalized* while the convolution consumes
/// the normalized copy, and the gate's square root has a 1e-6 magnitude floor
/// that matters exactly where the derivative would blow up.
///
/// The convolution carries `(K - 1) * dilation` rows of history across
/// tokens. Decode advances one row at a time, so the state is double-buffered
/// and rotated with a blit rather than shifted in place -- an overlapping
/// same-buffer copy is not something Metal will do for us.
final class PLEBlock {
    /// One INT4 affine projection, laid out the way the repacker writes a
    /// quantized tensor: weights, scales and biases in one allocation.
    struct Projection {
        let weights: MTLBuffer
        let weightsOffset: Int
        let scales: MTLBuffer
        let scalesOffset: Int
        let biases: MTLBuffer
        let biasesOffset: Int
        /// The tensor's own dtype, not the slot's -- a promoted family sits
        /// at bf16 inside an otherwise-quantized slot.
        var isBF16: Bool = false
    }

    /// A bf16 vector parameter (the three norms and the conv taps).
    struct Vector {
        let buffer: MTLBuffer
        let offset: Int
    }

    /// A batched GEMM the caller supplies: `(commandBuffer, projection, x,
    /// y, rows, columns, tokens)`.
    typealias BatchedProjection = (
        MTLCommandBuffer, Projection, MTLBuffer,
        MTLBuffer, Int, Int, Int
    ) throws -> Void

    struct Weights {
        let keyProj: Projection
        let valueProj: Projection
        let normKey: Vector
        let normQuery: Vector
        let normConv: Vector
        /// `[hcDim, kernelSize]`, channel-major: tap `k` of channel `c` is at
        /// `c * kernelSize + k`.
        let conv1d: Vector
    }

    let dim: Int
    let streams: Int
    let embedDim: Int
    let kernelSize: Int
    let dilation: Int
    /// Rows of convolution state carried between tokens.
    var history: Int { (kernelSize - 1) * dilation }
    private var hcDim: Int { dim * streams }

    private let rms: RMSNorm
    private let gemv: SlotGEMV
    private let elementwise: Elementwise

    /// Host-written gather destination: `embedDim` fp16 values.
    let embedding: MTLBuffer
    // Not private: the parity harness reads these back to compare the block
    // stage by stage against the reference implementation.
    let keyBuf: MTLBuffer  // [hcDim]
    let valueBuf: MTLBuffer  // [dim]
    let keyNormed: MTLBuffer  // [hcDim]
    let queryNormed: MTLBuffer  // [hcDim]
    let scoreBuf: MTLBuffer  // [streams]
    let gateBuf: MTLBuffer  // [streams]
    let gatedBuf: MTLBuffer  // [hcDim]
    let convOut: MTLBuffer  // [hcDim]
    /// `[history + maxRows, hcDim]`, ping-ponged so the row shift is a
    /// non-overlapping blit.
    ///
    /// Not private: `rewindWindow`'s contract -- which of the pair it reads,
    /// and from which row -- is only observable on these buffers, and it is the
    /// one piece of state a speculative pass can leave desynchronized in
    /// silence. The parity dump reads the others for the same reason.
    var xpad: [MTLBuffer]
    private var xpadIndex = 0

    /// Rows the scratch is sized for: one for decode, a whole chunk for
    /// prefill.
    let maxRows: Int

    init(
        context: MetalContext, dim: Int, streams: Int, embedDim: Int,
        kernelSize: Int, dilation: Int, maxRows: Int = 1,
        weightBits: Int = 4
    ) throws {
        precondition(dim > 0 && streams > 0 && embedDim > 0)
        precondition(kernelSize > 1 && dilation > 0 && maxRows > 0)
        self.maxRows = maxRows
        self.dim = dim
        self.streams = streams
        self.embedDim = embedDim
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.rms = try RMSNorm(context: context)
        self.gemv = try SlotGEMV(context: context, weightBits: weightBits)
        self.elementwise = try Elementwise(context: context)
        let wide = dim * streams
        let f16 = MemoryLayout<Float16>.stride
        func make(_ count: Int) throws -> MTLBuffer {
            guard
                let b = context.device.makeBuffer(
                    length: max(count, 1) * f16, options: .storageModeShared)
            else {
                throw MetalError.bufferAllocationFailed("PLE scratch")
            }
            return b
        }
        self.embedding = try make(embedDim * maxRows)
        self.keyBuf = try make(wide * maxRows)
        self.valueBuf = try make(dim * maxRows)
        self.keyNormed = try make(wide * maxRows)
        self.queryNormed = try make(wide * maxRows)
        self.scoreBuf = try make(streams * maxRows)
        self.gateBuf = try make(streams * maxRows)
        self.gatedBuf = try make(wide * maxRows)
        self.convOut = try make(wide * maxRows)
        let padRows = (kernelSize - 1) * dilation + maxRows
        self.xpad = [try make(padRows * wide), try make(padRows * wide)]
        resetState()
    }

    /// Rewinds the convolution window to `acceptedRows` of the pass that just
    /// ran `passRows`.
    ///
    /// Speculative decoding runs the target over rows it may discard, and this
    /// window is the one piece of the target's state that neither self-heals
    /// nor is covered by the KV rewind: the KV row is overwritten by its
    /// replacement and the pooled indexer blocks are recomputed from raw keys,
    /// but a rolling convolution advanced two rows for one accepted token
    /// stays desynchronized for the rest of the generation, silently.
    ///
    /// The rewind is exact and costs one blit, because the buffer the pass
    /// read from is still intact: the pair ping-pongs, so advancing by a
    /// different amount is just a different source offset into it.
    func rewindWindow(acceptedRows: Int, passRows: Int) {
        precondition(
            acceptedRows >= 0 && acceptedRows <= passRows,
            "cannot accept \(acceptedRows) of \(passRows) rows")
        guard acceptedRows != passRows else { return }
        let rowBytes = hcDim * MemoryLayout<Float16>.stride
        let source = xpad[1 - xpadIndex]
        let destination = xpad[xpadIndex]
        memcpy(
            destination.contents(),
            source.contents().advanced(by: acceptedRows * rowBytes),
            history * rowBytes)
    }

    // MARK: - Auxiliary state (Qwen3.8 interior checkpoints)

    /// Append the convolution history the next token reads.
    ///
    /// `xpad[xpadIndex]` holds exactly `history` valid rows at offset 0 after
    /// `encodeRows`; the padding rows above them are scratch and are not part
    /// of the state.
    func appendSnapshotPayload(to data: inout Data) {
        let rowBytes = hcDim * MemoryLayout<Float16>.stride
        let bytes = history * rowBytes
        AuxiliaryCoding.appendUInt32(xpadIndex, to: &data)
        AuxiliaryCoding.appendUInt32(history, to: &data)
        AuxiliaryCoding.appendBytes(
            xpad[xpadIndex].contents(), count: bytes, to: &data)
    }

    /// Restore what `appendSnapshotPayload` wrote. Both ping-pong buffers are
    /// cleared first so the unused one cannot leak an earlier prompt.
    func restoreSnapshotPayload(
        _ bytes: UnsafeRawBufferPointer,
        offset: inout Int
    ) throws {
        let index = try AuxiliaryCoding.readUInt32(bytes, offset: &offset)
        let storedHistory = try AuxiliaryCoding.readUInt32(bytes, offset: &offset)
        guard index < 2 else {
            throw InferenceAuxiliaryStateError.malformed(
                "PLE xpad index \(index) is not 0 or 1")
        }
        guard storedHistory == history else {
            throw InferenceAuxiliaryStateError.malformed(
                "PLE history \(storedHistory) does not match \(history)")
        }
        let rowBytes = hcDim * MemoryLayout<Float16>.stride
        let count = history * rowBytes
        try AuxiliaryCoding.require(
            bytes, offset: offset, count: count, "PLE history")
        guard let base = bytes.baseAddress else {
            throw InferenceAuxiliaryStateError.truncated("PLE history")
        }
        for buffer in xpad {
            memset(buffer.contents(), 0, buffer.length)
        }
        memcpy(
            xpad[index].contents(),
            base.advanced(by: offset),
            count)
        offset += count
        xpadIndex = index
    }

    /// Clears the carried convolution history. Call between completions: a
    /// state left over from a previous prompt would leak that prompt's
    /// n-grams into the first tokens of the next one.
    func resetState() {
        for buffer in xpad {
            memset(buffer.contents(), 0, buffer.length)
        }
        xpadIndex = 0
    }

    /// Encodes the block for a single token, rewriting `streams` in place.
    ///
    /// `embedding` must already hold this token's gathered rows. The state
    /// rotation is encoded on the same command buffer, after the read, so the
    /// caller only has to keep tokens in order.
    func encodeDecode(
        commandBuffer: MTLCommandBuffer,
        streamsBuffer: MTLBuffer,
        weights: Weights,
        eps: Float
    ) throws {
        try encodeRows(
            commandBuffer: commandBuffer,
            streamsBuffer: streamsBuffer, weights: weights,
            tokens: 1, eps: eps, project: nil)
    }

    /// Decode's own GEMV arithmetic for every row, weights read once
    /// (`SlotGEMV.encodeRows`): the projection a multi-row pass passes to
    /// `encodeRows` when it must reproduce scalar decode's numbers (the wide
    /// verify), instead of the prefill QMM.
    var decodeOrderProjection: BatchedProjection {
        { [gemv] commandBuffer, projection, x, y, rows, columns, tokens in
            try gemv.encodeRows(
                commandBuffer: commandBuffer,
                weights: projection.weights, weightsOffset: projection.weightsOffset,
                scales: projection.scales, scalesOffset: projection.scalesOffset,
                biases: projection.biases, biasesOffset: projection.biasesOffset,
                x: x, y: y, m: rows, n: columns, rows: tokens,
                isBF16: projection.isBF16)
        }
    }

    /// The block over `tokens` rows.
    ///
    /// `project` batches the two embedding projections; passing nil uses the
    /// per-row GEMV, which is what a decode step wants. Everything else is
    /// the same arithmetic with a row count threaded through, so decode and
    /// prefill cannot drift apart.
    func encodeRows(
        commandBuffer: MTLCommandBuffer,
        streamsBuffer: MTLBuffer,
        weights: Weights,
        tokens: Int,
        eps: Float,
        project: BatchedProjection?
    ) throws {
        precondition(
            tokens <= maxRows,
            "PLEBlock scratch holds \(maxRows) rows, asked for \(tokens)")
        let wide = hcDim
        if let project {
            try project(
                commandBuffer, weights.keyProj, embedding, keyBuf,
                wide, embedDim, tokens)
            try project(
                commandBuffer, weights.valueProj, embedding, valueBuf,
                dim, embedDim, tokens)
        } else {
            try gemv.encode(
                commandBuffer: commandBuffer,
                weights: weights.keyProj.weights,
                weightsOffset: weights.keyProj.weightsOffset,
                scales: weights.keyProj.scales,
                scalesOffset: weights.keyProj.scalesOffset,
                biases: weights.keyProj.biases,
                biasesOffset: weights.keyProj.biasesOffset,
                x: embedding, y: keyBuf,
                m: UInt32(wide), n: UInt32(embedDim),
                isBF16: weights.keyProj.isBF16)
            try gemv.encode(
                commandBuffer: commandBuffer,
                weights: weights.valueProj.weights,
                weightsOffset: weights.valueProj.weightsOffset,
                scales: weights.valueProj.scales,
                scalesOffset: weights.valueProj.scalesOffset,
                biases: weights.valueProj.biases,
                biasesOffset: weights.valueProj.biasesOffset,
                x: embedding, y: valueBuf,
                m: UInt32(dim), n: UInt32(embedDim),
                isBF16: weights.valueProj.isBF16)
        }
        try rms.encodeBF16WGrouped(
            commandBuffer: commandBuffer,
            x: keyBuf,
            weight: weights.normKey.buffer,
            weightOffset: weights.normKey.offset,
            out: keyNormed,
            groupDim: UInt32(dim), numGroups: streams,
            eps: eps, tokens: tokens)
        try rms.encodeBF16WGrouped(
            commandBuffer: commandBuffer,
            x: streamsBuffer,
            weight: weights.normQuery.buffer,
            weightOffset: weights.normQuery.offset,
            out: queryNormed,
            groupDim: UInt32(dim), numGroups: streams,
            eps: eps, tokens: tokens)
        try elementwise.encodePLEStreamScore(
            commandBuffer: commandBuffer,
            key: keyNormed, query: queryNormed,
            out: scoreBuf,
            dim: dim, streams: streams,
            tokens: tokens)
        try elementwise.encodePLESignedSqrtGate(
            commandBuffer: commandBuffer,
            x: scoreBuf, out: gateBuf,
            count: streams * tokens)
        try elementwise.encodePLEBroadcastScale(
            commandBuffer: commandBuffer,
            value: valueBuf, gate: gateBuf,
            out: gatedBuf,
            dim: dim, streams: streams,
            tokens: tokens)
        // The convolution's newest row is the normalized copy of `gated`,
        // written straight into the last slot of the padded window.
        let rowBytes = wide * MemoryLayout<Float16>.stride
        let current = xpad[xpadIndex]
        try rms.encodeBF16WGrouped(
            commandBuffer: commandBuffer,
            x: gatedBuf,
            weight: weights.normConv.buffer,
            weightOffset: weights.normConv.offset,
            out: current, outOffset: history * rowBytes,
            groupDim: UInt32(dim), numGroups: streams,
            eps: eps, tokens: tokens)
        try elementwise.encodePLEDilatedConv(
            commandBuffer: commandBuffer,
            xpad: current,
            weight: weights.conv1d.buffer,
            weightOffset: weights.conv1d.offset,
            out: convOut,
            channels: wide, tokens: tokens,
            kernelSize: kernelSize,
            dilation: dilation)
        // Both terms land on the residual: the gated value unnormalized, and
        // the convolution, whose kernel has already applied the silu.
        try elementwise.encodeResidualAdd(
            commandBuffer: commandBuffer,
            hidden: streamsBuffer,
            delta: gatedBuf, count: wide * tokens)
        try elementwise.encodeResidualAdd(
            commandBuffer: commandBuffer,
            hidden: streamsBuffer,
            delta: convOut, count: wide * tokens)
        // Advance the window by one row: rows 1...history of the buffer we
        // just used become rows 0..<history of the next one.
        let next = xpad[1 - xpadIndex]
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.copy(
            from: current, sourceOffset: tokens * rowBytes,
            to: next, destinationOffset: 0,
            size: history * rowBytes)
        blit.endEncoding()
        xpadIndex = 1 - xpadIndex
    }
}
