import Foundation
import Metal

/// How a family moves data between the residual stream and a sublayer.
///
/// The Qwen3.5-MoE families use plain pre-norm: normalize the residual into a
/// block's input, add the block's output back. Every layer does this twice,
/// once around attention and once around the MLP.
///
/// Qwen3.8-Flash-Next replaces both halves. Its residual is four parallel
/// 2560-wide streams, read through a learned gate that collapses them to one
/// vector and written through another that injects the block's output into
/// every stream with its own weight (`HyperConnection`). The shape of the
/// layer loop is unchanged — still normalize-in, combine-out — so the
/// difference belongs at this seam rather than branching the loop.
///
/// These four call sites are the entire coupling. Keeping them in one file
/// means adding the new family's behaviour touches this file and not the
/// decode or prefill bodies, matching how `TensorSchema` isolates naming.
/// Which sublayer a residual entry/exit pair brackets. A hyper-connection
/// family has separate gate weights for each; a pre-norm family ignores it.
enum ResidualSublayer {
    case attention
    case mlp
}

extension RealForwardRunner {
    /// Residual streams this model carries. One for pre-norm families; the
    /// hyper-connection families carry `hc_count`.
    var residualStreamCount: Int {
        cfg.hyperConnections.enabled ? cfg.hyperConnections.count : 1
    }

    /// Width of the residual buffer in elements.
    var residualWidth: Int { cfg.hiddenSize * residualStreamCount }

    func gateWeightsPublic(_ view: TensorView) -> HyperConnection.Weights {
        gateWeights(view)
    }

    private func gateWeights(_ view: TensorView) -> HyperConnection.Weights {
        HyperConnection.Weights(
            weights: view.buffer,
            weightsOffset: Int(view.offset),
            scales: view.buffer,
            scalesOffset: Int(view.scaleOffset),
            biases: view.buffer,
            biasesOffset: Int(view.biasOffset),
            // dtype 1 is BF16; a promoted tensor carries no
            // scales or biases and must not be unpacked.
            isBF16: view.dtype == 1)
    }

    /// Fused hyper-connection gates. Default off until the golden has proven
    /// the fused kernels bit-identical; TINYTITAN_HC_FUSED=1 turns them on.
    var hcFusedEnabled: Bool { profile.hcFused }

    /// Residual -> block input, for one decode token.
    func encodeResidualEntryDecode(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer,
        hiddenOffset: Int = 0,
        norm: TensorView,
        out: MTLBuffer,
        outOffset: Int = 0,
        sublayer: ResidualSublayer,
        layer: Int,
        eps: Float
    ) throws {
        if let hc = hyperConnection {
            precondition(
                hiddenOffset == 0,
                "hyper-connections require a caller-owned row buffer")
            let down =
                sublayer == .attention
                ? try model.hcAttnMixDown(layer: layer)
                : try model.hcMlpMixDown(layer: layer)
            let up =
                sublayer == .attention
                ? try model.hcAttnMixUp(layer: layer)
                : try model.hcMlpMixUp(layer: layer)
            if !ablated("hcread") {
                let dw = gateWeights(down)
                let uw = gateWeights(up)
                if hcFusedEnabled, hc.canFuseRead(down: dw, up: uw) {
                    try hc.encodeReadFused(
                        commandBuffer: commandBuffer,
                        streamsBuffer: hidden,
                        hcNorm: norm.buffer,
                        hcNormOffset: Int(norm.offset),
                        down: dw, up: uw,
                        blockInput: out, eps: eps)
                } else {
                    try hc.encodeRead(
                        commandBuffer: commandBuffer,
                        streamsBuffer: hidden,
                        hcNorm: norm.buffer,
                        hcNormOffset: Int(norm.offset),
                        down: dw, up: uw,
                        blockInput: out, eps: eps)
                }
            }
            return
        }
        if !ablated("norm") {
            try rms.encodeBF16W(
                commandBuffer: commandBuffer,
                x: hidden, xOffset: hiddenOffset,
                weight: norm.buffer, weightOffset: Int(norm.offset),
                out: out, outOffset: outOffset,
                d: UInt32(cfg.hiddenSize), eps: eps)
        }
    }

    /// Block output -> residual, for one decode token.
    ///
    /// Consumes the `normed` the matching entry left inside the
    /// `HyperConnection`, so the two must bracket exactly one block on the
    /// same command buffer.
    func encodeResidualExitDecode(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer,
        hiddenOffset: Int = 0,
        delta: MTLBuffer,
        deltaOffset: Int = 0,
        sublayer: ResidualSublayer,
        layer: Int
    ) throws {
        if let hc = hyperConnection {
            precondition(
                hiddenOffset == 0 && deltaOffset == 0,
                "hyper-connections require a caller-owned row buffer")
            let inject =
                sublayer == .attention
                ? try model.hcAttnInject(layer: layer)
                : try model.hcMlpInject(layer: layer)
            let iw = gateWeights(inject)
            if hcFusedEnabled, hc.canFuseWrite(inject: iw) {
                try hc.encodeWriteFused(
                    commandBuffer: commandBuffer,
                    streamsBuffer: hidden,
                    inject: iw,
                    blockOut: delta)
            } else {
                try hc.encodeWrite(
                    commandBuffer: commandBuffer,
                    streamsBuffer: hidden,
                    inject: iw,
                    blockOut: delta)
            }
            return
        }
        try requireElementwise().encodeResidualAdd(
            commandBuffer: commandBuffer,
            hidden: hidden,
            hiddenOffset: hiddenOffset,
            delta: delta,
            deltaOffset: deltaOffset,
            count: cfg.hiddenSize)
    }

    /// The batched projection the hyper-connection gates hand their GEMMs to.
    /// Routing them through the runner's own prefill dispatch keeps one
    /// policy for which kernel serves which shape.
    var prefillGateProjection: HyperConnection.BatchedProjection {
        { [self] commandBuffer, weights, x, y, rows, columns, tokens in
            // A promoted gate has no scales or biases, so the batched QMM
            // cannot read it. One GEMV per row instead; block_inject is four
            // rows, so there is little batching to lose.
            if weights.isBF16 {
                let halfBytes = MemoryLayout<Float16>.stride
                for row in 0..<tokens {
                    try bf16Projection.encode(
                        commandBuffer: commandBuffer,
                        weights: weights.weights,
                        weightsOffset: weights.weightsOffset,
                        x: x, xOffset: row * columns * halfBytes,
                        y: y, yOffset: row * rows * halfBytes,
                        m: UInt32(rows), n: UInt32(columns))
                }
                return
            }
            try prefillQMM.encode(
                commandBuffer: commandBuffer,
                weights: weights.weights,
                weightsOffset: weights.weightsOffset,
                scales: weights.scales,
                scalesOffset: weights.scalesOffset,
                biases: weights.biases,
                biasesOffset: weights.biasesOffset,
                x: x, y: y,
                t: tokens, n: rows, k: columns)
        }
    }

    /// Residual -> block input, for a prefill chunk of `tokens` rows.
    func encodeResidualEntryPrefill(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer,
        norm: TensorView,
        out: MTLBuffer,
        sublayer: ResidualSublayer,
        layer: Int,
        tokens: Int,
        eps: Float
    ) throws {
        if let hc = hyperConnection {
            let down =
                sublayer == .attention
                ? try model.hcAttnMixDown(layer: layer)
                : try model.hcMlpMixDown(layer: layer)
            let up =
                sublayer == .attention
                ? try model.hcAttnMixUp(layer: layer)
                : try model.hcMlpMixUp(layer: layer)
            // A wide verify pass must reproduce decode, whose read is the
            // composed path (the fused read cannot build; see canFuseRead):
            // the same norm, silu and reduce, with decode's GEMV per row.
            if decodeOrderVerify, hcFusedEnabled,
                hc.canFuseRead(down: gateWeightsPublic(down), up: gateWeightsPublic(up))
            {
                throw PrefillError.chunkedUnsupported(
                    "decode reads hyper-connections fused; the verify pass has no fused multi-row read"
                )
            }
            try hc.encodeReadRows(
                commandBuffer: commandBuffer,
                streamsBuffer: hidden,
                hcNorm: norm.buffer,
                hcNormOffset: Int(norm.offset),
                down: gateWeightsPublic(down),
                up: gateWeightsPublic(up),
                blockInput: out,
                tokens: tokens, eps: eps,
                project: decodeOrderVerify ? hc.decodeOrderProjection : prefillGateProjection)
            return
        }
        try prefillRMS.encodeBF16W(
            commandBuffer: commandBuffer,
            x: hidden,
            weight: norm.buffer,
            weightOffset: Int(norm.offset),
            out: out,
            t: UInt32(tokens),
            d: UInt32(cfg.hiddenSize),
            eps: eps)
    }

    /// Block output -> residual, for a prefill chunk of `tokens` rows.
    func encodeResidualExitPrefill(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer,
        delta: MTLBuffer,
        sublayer: ResidualSublayer,
        layer: Int,
        tokens: Int
    ) throws {
        if let hc = hyperConnection {
            let inject =
                sublayer == .attention
                ? try model.hcAttnInject(layer: layer)
                : try model.hcMlpInject(layer: layer)
            let weights = gateWeightsPublic(inject)
            // A wide verify pass writes each row the way decode does: the
            // fused inject when decode would use it, at that row's offsets.
            if decodeOrderVerify, hcFusedEnabled, hc.canFuseWrite(inject: weights) {
                let half = MemoryLayout<Float16>.stride
                for row in 0..<tokens {
                    try hc.encodeWriteFused(
                        commandBuffer: commandBuffer,
                        streamsBuffer: hidden, streamsOffset: row * residualWidth * half,
                        inject: weights,
                        blockOut: delta, blockOutOffset: row * cfg.hiddenSize * half,
                        normedOffset: row * residualWidth * half)
                }
                return
            }
            try hc.encodeWriteRows(
                commandBuffer: commandBuffer,
                streamsBuffer: hidden,
                inject: weights,
                blockOut: delta,
                tokens: tokens,
                project: decodeOrderVerify ? hc.decodeOrderProjection : prefillGateProjection)
            return
        }
        try requireElementwise().encodeResidualAdd(
            commandBuffer: commandBuffer,
            hidden: hidden,
            delta: delta,
            count: tokens * cfg.hiddenSize)
    }
}

extension RealForwardRunner {
    /// The shared expert's scalar gate, whichever way it is stored.
    ///
    /// The gate is one row of D and its error does not average over anything,
    /// which is why the 8-bit build promotes it to the checkpoint's bf16. The
    /// dtype travels on the tensor, not the slot, so the choice is made here
    /// rather than when the kernels are built.
    func encodeScalarGate(
        commandBuffer: MTLCommandBuffer,
        view: TensorView,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        n: UInt32
    ) throws {
        if view.dtype == 1 {
            try requireBF16ScalarGate().encode(
                commandBuffer: commandBuffer,
                weights: view.buffer,
                weightsOffset: Int(view.offset),
                x: x, xOffset: xOffset,
                y: y, yOffset: yOffset,
                m: 1, n: n)
        } else {
            try requireInt8ScalarGate().encode(
                commandBuffer: commandBuffer,
                weights: view.buffer,
                weightsOffset: Int(view.offset),
                scales: view.buffer,
                scalesOffset: Int(view.scaleOffset),
                biases: view.buffer,
                biasesOffset: Int(view.biasOffset),
                x: x, xOffset: xOffset,
                y: y, yOffset: yOffset,
                m: 1, n: n)
        }
    }
}
