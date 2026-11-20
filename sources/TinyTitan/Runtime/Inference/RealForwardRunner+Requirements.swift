import Foundation
import Metal
import TinyTitanCore

// Checked accessors for the optional kernels and buffers a decode or prefill
// path needs. Split out of RealForwardRunner.swift (2026-09-28) under the
// 500-line-per-file rule as pure code motion.

extension RealForwardRunner {
    /// The elementwise kernel bundle, or a thrown error when this model's
    /// profile did not create one. Call sites that need it used to force
    /// unwrap the optional, so a profile mismatch crashed instead of failing.
    func requireElementwise() throws -> Elementwise {
        guard let elementwise else {
            throw ModelError.internalInconsistency(
                detail: "elementwise kernels are required by this path but the model profile "
                    + "did not enable them")
        }
        return elementwise
    }

    // Checked accessors for the other optional kernels and buffers a path
    // needs. Each is nil exactly when the model's profile does not dispatch
    // that path; the call sites used to force-unwrap them, so a profile
    // mismatch crashed instead of naming what was missing.

    func requireAffine() throws -> AffineQuantGEMV {
        guard let affine else {
            throw ModelError.internalInconsistency(
                detail: "the affine GEMV kernel is required here but the profile did not create one"
            )
        }
        return affine
    }

    func requireOnesPerExpertScale() throws -> MTLBuffer {
        try requireBuffer(onesPerExpertScale, "expert per-expert scale buffer")
    }

    func requireBF16ScalarGate() throws -> BF16GEMV {
        guard let bf16ScalarGate else {
            throw ModelError.internalInconsistency(
                detail:
                    "the bf16 scalar-gate kernel is required here but the profile did not create one"
            )
        }
        return bf16ScalarGate
    }

    func requireInt8ScalarGate() throws -> DequantInt8GEMV {
        guard let int8ScalarGate else {
            throw ModelError.internalInconsistency(
                detail:
                    "the int8 scalar-gate kernel is required here but the profile did not create one"
            )
        }
        return int8ScalarGate
    }

    func requireBuffer(_ buffer: MTLBuffer?, _ what: String) throws -> MTLBuffer {
        guard let buffer else {
            throw ModelError.internalInconsistency(
                detail: "the \(what) is required here but the model profile did not create it")
        }
        return buffer
    }

    func requireTensorView(_ view: TensorView?, _ what: String) throws -> TensorView {
        guard let view else {
            throw ModelError.internalInconsistency(
                detail: "the \(what) is required here but this layer does not carry it")
        }
        return view
    }
}
