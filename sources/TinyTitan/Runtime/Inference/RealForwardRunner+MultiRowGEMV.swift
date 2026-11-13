import Foundation
import Metal

/// The multi-row form of decode's projection dispatch, for the wide verify's
/// decode-order pass (docs/plan-wide-verify.md, task 7).
///
/// Each helper names the `MultiRowGEMV.Format` -- that is, the decode kernel
/// whose per-row arithmetic the multi-row dispatch reproduces -- that the
/// matching single-row dispatcher would pick for the same tensor. The choice
/// must follow decode's exactly: a projection read at another width, or
/// through another body, is a verify row that differs from scalar decode.
extension RealForwardRunner {
    /// A multi-row GEMV over this runner's library. Holds only the context;
    /// pipelines are cached by `MetalContext`.
    var multiRowGEMV: MultiRowGEMV { MultiRowGEMV(context: ctx) }

    /// What `encodeRoleGEMV` reads `projection` with at `weightBits`.
    func roleGEMVFormat(_ projection: TensorView, weightBits: Int) throws -> MultiRowGEMV.Format {
        if projection.dtype == 1 { return .bf16 }
        if weightBits == 4 { return .int4 }
        guard affineByWidth[weightBits] != nil else {
            throw ModelError.unsupportedArchitecture(
                detail: "no \(weightBits)-bit GEMV is built for a multi-row projection")
        }
        return .affine(bits: weightBits)
    }

    /// What `encodePrimaryGEMV(projection:)` reads `projection` with.
    func primaryGEMVFormat(_ projection: TensorView) -> MultiRowGEMV.Format {
        if projection.dtype == 1 { return .bf16 }
        if let affine { return .affine(bits: affine.weightBits) }
        return .int4
    }

    /// The GDN input projections as `encodeLinearAttentionDecode` reads them:
    /// the fused int4 kernel when all four qualify, else each at the GDN role's
    /// width.
    func gdnInputProjectionFormats(_ views: [TensorView]) throws -> [MultiRowGEMV.Format] {
        let fused = FusedKernelPolicy.mayUseFusedGDNInputProjections(
            attentionWeightBits: model.attentionWeightBits,
            dTypes: views.map { Int($0.dtype) })
        if fused { return views.map { _ in .int4 } }
        return try views.map {
            try roleGEMVFormat($0, weightBits: model.gdnProjectionWeightBits)
        }
    }

    /// The gated full attention's q/k/v as `encodeGatedFullQKVProjection`
    /// reads them: the fused int4 kernel when all three qualify, else the
    /// primary GEMV per projection.
    func gatedQKVFormats(q: TensorView, k: TensorView, v: TensorView) -> [MultiRowGEMV.Format] {
        let fused = FusedKernelPolicy.mayUseFusedQKV(
            qoProjectionWeightBits: model.qoProjectionWeightBits,
            kvProjectionWeightBits: model.kvProjectionWeightBits,
            qDType: Int(q.dtype), kDType: Int(k.dtype), vDType: Int(v.dtype))
        if fused { return [.int4, .int4, .int4] }
        return [q, k, v].map(primaryGEMVFormat)
    }

    /// `encodePrimaryGEMV(projection:)` for `rows` activation rows, weights
    /// read once: row `r` reads `x[r * n]` and writes `y[r * yRowStride]`.
    func encodePrimaryGEMVRows(
        commandBuffer cb: MTLCommandBuffer,
        projection p: TensorView,
        x: MTLBuffer, xRowStride: Int,
        y: MTLBuffer, yRowStride: Int,
        m: Int, n: Int, rows: Int
    ) throws {
        try multiRowGEMV.encode(
            commandBuffer: cb,
            segments: [
                .init(
                    view: p, format: primaryGEMVFormat(p), rows: m,
                    y: y, yRowStride: yRowStride)
            ],
            x: x, xRowStride: xRowStride, activationRows: rows, columns: n)
    }

    func encodeRowCopy(
        _ cb: MTLCommandBuffer,
        from source: MTLBuffer, offset sourceOffset: Int,
        to destination: MTLBuffer, offset destinationOffset: Int,
        bytes: Int
    ) throws {
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        blit.copy(
            from: source, sourceOffset: sourceOffset,
            to: destination, destinationOffset: destinationOffset,
            size: bytes)
        blit.endEncoding()
    }
}
