import Foundation
import Metal

// The PLE n-gram block: gathering rows, rewinding accepted speculations and
// the decode/prefill encoders.
//
// Split out of `RealForwardRunner+Residual.swift` (2026-09-28) under the
// 500-line-per-file rule (Task 8 of the cleanup runbook) as pure code motion.
// Its two `private` view helpers moved with their only callers.
// MARK: - PLE n-gram block

/// AUD2-036: a prefill gather writes into an allocation sized by the runner's
/// own `prefillChunkTokens`, while the chunk scheduler accepts a caller's
/// `PrefillRuntimeConfig` independently. The write is refused here rather than
/// allowed to run past the block's embedding buffer.
enum PLEPrefillGather {
    static func validate(rows: Int, maxRows: Int) throws {
        guard rows <= maxRows else {
            throw PrefillError.chunkedUnsupported(
                "PLE prefill gather of \(rows) rows exceeds the \(maxRows)-row embedding buffer; the caller's chunk is larger than this runner's prefillChunkTokens"
            )
        }
    }
}

extension RealForwardRunner {
    /// Reads this token's n-gram rows off storage into the block's input.
    ///
    /// A no-op for families without PLE. The gather is 16 rows of 320 bytes —
    /// 5 KiB — which is three orders of magnitude under one token's routed
    /// expert traffic, so it is done inline rather than scheduled.
    func gatherPLERows(token: Int32) throws {
        guard let ple = pleBlock, let hash = pleHash, let table = ngramTable
        else { return }
        pleContext.insert(token, at: 0)
        if pleContext.count > hash.ngramSize {
            pleContext.removeLast(pleContext.count - hash.ngramSize)
        }
        let rows = hash.rows(context: pleContext)
        try table.gatherConcurrentlyRows(
            rows: rows, into: ple.embedding.contents(),
            threads: Self.pleGatherThreads)
    }

    /// Readers for one token's n-gram rows. One is the measured-off arm: a
    /// token's 16 random reads ran serially and cost 14-15 ms on the 125B.
    static let pleGatherThreads: Int = {
        guard
            let raw = ProcessInfo.processInfo
                .environment["TINYTITAN_PLE_GATHER_THREADS"],
            let value = Int(raw), value > 0
        else { return 8 }
        return min(value, 16)
    }()

    /// Rewinds the n-gram block's carried state after a speculative pass whose
    /// rows were not all accepted.
    func rewindPLE(acceptedRows: Int, passRows: Int) {
        guard pleBlock != nil, acceptedRows < passRows else { return }
        pleBlock?.rewindWindow(acceptedRows: acceptedRows, passRows: passRows)
        // The hashed context must drop the same rows, or every later token's
        // n-gram ids are computed one position out.
        let discard = min(passRows - acceptedRows, pleContext.count)
        pleContext.removeFirst(discard)
    }

    /// Clears the state that spans a completion: the convolution history and
    /// the token context it hashes. Leaving either in place would let one
    /// prompt's trailing n-grams open the next one.
    func resetPLEState() {
        pleContext.removeAll(keepingCapacity: true)
        pleBlock?.resetState()
    }

    /// Encodes the n-gram block on the layers that carry one.
    func encodePLEDecode(
        commandBuffer: MTLCommandBuffer,
        layer: Int, position: Int, eps: Float
    ) throws {
        guard let ple = pleBlock,
            cfg.ple.layerIndices.contains(layer)
        else { return }
        if activationDumpActive(position: position) {
            dumpActivation("L\(layer)_ple_pre", hidden, count: residualWidth, position: position)
        }
        let weights = PLEBlock.Weights(
            keyProj: projection(try model.pleKeyProj(layer: layer)),
            valueProj: projection(try model.pleValueProj(layer: layer)),
            normKey: vector(try model.pleNormKey(layer: layer)),
            normQuery: vector(try model.pleNormQuery(layer: layer)),
            normConv: vector(try model.pleNormConv(layer: layer)),
            conv1d: vector(try model.pleConv(layer: layer)))
        try ple.encodeDecode(
            commandBuffer: commandBuffer,
            streamsBuffer: hidden,
            weights: weights, eps: eps)
    }

    /// Gathers a whole prefill chunk's n-gram rows.
    ///
    /// Unlike decode, the context for row `i` is the chunk's own tokens plus
    /// whatever preceded the chunk, so this walks the chunk in order and
    /// leaves `pleContext` positioned for the next one.
    func gatherPLERowsPrefill(tokens: ArraySlice<Int32>) throws {
        guard let ple = pleBlock, let hash = pleHash, let table = ngramTable
        else { return }
        // The gather writes `tokens.count` rows into `ple.embedding`, which is
        // `embedDim * ple.maxRows`. The scratch guard at the chunk entry bounds
        // the caller's chunk, not this buffer: `PrefillRuntimeConfig.production`
        // accepts any allowed chunk independently of the runner's own
        // `prefillChunkTokens`, so a mismatched caller could ask for more rows
        // than the block allocated (AUD2-036).
        try PLEPrefillGather.validate(rows: tokens.count, maxRows: ple.maxRows)
        let rowBytes = hash.headCount * table.rowBytes
        // Two passes. Resolving a position's rows consumes the hash window, so
        // the addresses are inherently ordered even though the reads are not.
        var batches = [[UInt32]]()
        batches.reserveCapacity(tokens.count)
        for token in tokens {
            pleContext.insert(token, at: 0)
            if pleContext.count > hash.ngramSize {
                pleContext.removeLast(pleContext.count - hash.ngramSize)
            }
            batches.append(hash.rows(context: pleContext))
        }
        // Each row is a 320 B random read, so the cost of the gather is latency
        // and independent positions can be fetched concurrently: 660 ms serial
        // against 109 ms at the default depth for a 412-token prompt.
        let tGather = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try table.gatherConcurrently(
            batches: batches,
            chunkStride: rowBytes,
            into: ple.embedding.contents())
        // Counted in `gather_ms` alongside decode's gather: before this the
        // prefill side was invisible, so its cost could only be extrapolated
        // from a micro-benchmark rather than read off a run.
        totalGatherNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tGather
    }

    /// Encodes the n-gram block over a whole prefill chunk.
    func encodePLEPrefill(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer,
        layer: Int, tokens: Int, eps: Float
    ) throws {
        guard let ple = pleBlock,
            cfg.ple.layerIndices.contains(layer)
        else { return }
        let weights = PLEBlock.Weights(
            keyProj: projection(try model.pleKeyProj(layer: layer)),
            valueProj: projection(try model.pleValueProj(layer: layer)),
            normKey: vector(try model.pleNormKey(layer: layer)),
            normQuery: vector(try model.pleNormQuery(layer: layer)),
            normConv: vector(try model.pleNormConv(layer: layer)),
            conv1d: vector(try model.pleConv(layer: layer)))
        if decodeOrderVerify {
            // A wide verify pass reproduces decode's per-row GEMV.
            try ple.encodeRows(
                commandBuffer: commandBuffer,
                streamsBuffer: hidden, weights: weights,
                tokens: tokens, eps: eps, project: ple.decodeOrderProjection)
            return
        }
        try ple.encodeRows(
            commandBuffer: commandBuffer,
            streamsBuffer: hidden, weights: weights,
            tokens: tokens, eps: eps
        ) {
            [self] cb, proj, x, y, rows, columns, count in
            // Same reason as the hyper-connection gate: a promoted projection
            // has no scales or biases for the batched QMM to read.
            if proj.isBF16 {
                let halfBytes = MemoryLayout<Float16>.stride
                for row in 0..<count {
                    try bf16Projection.encode(
                        commandBuffer: cb,
                        weights: proj.weights,
                        weightsOffset: proj.weightsOffset,
                        x: x, xOffset: row * columns * halfBytes,
                        y: y, yOffset: row * rows * halfBytes,
                        m: UInt32(rows), n: UInt32(columns))
                }
                return
            }
            try prefillQMM.encode(
                commandBuffer: cb,
                weights: proj.weights,
                weightsOffset: proj.weightsOffset,
                scales: proj.scales,
                scalesOffset: proj.scalesOffset,
                biases: proj.biases,
                biasesOffset: proj.biasesOffset,
                x: x, y: y,
                t: count, n: rows, k: columns)
        }
    }

    private func projection(_ view: TensorView) -> PLEBlock.Projection {
        PLEBlock.Projection(
            weights: view.buffer,
            weightsOffset: Int(view.offset),
            scales: view.buffer,
            scalesOffset: Int(view.scaleOffset),
            biases: view.buffer,
            biasesOffset: Int(view.biasOffset),
            isBF16: view.dtype == 1)
    }

    private func vector(_ view: TensorView) -> PLEBlock.Vector {
        PLEBlock.Vector(buffer: view.buffer, offset: Int(view.offset))
    }
}
