import Foundation

/// Which width each runtime role gets in the converted install.
///
/// The policy is deliberately small and declared, because it has to agree with
/// the manifest's quant slots (`GGUFManifestBuilder` writes 4 for
/// embedding/shared experts, 8 for attention and router) — the manifest is a
/// claim about bytes, and this is the rule that makes the claim true.
///
/// Two constraints override the slot:
///
/// - **A tensor the affine format cannot describe stays bf16.** Affine 4/8-bit
///   packs rows in groups of 64, so a 1-D norm, a 4-wide inject vector or any
///   row whose width is not a multiple of 64 must be bf16. That is a property of
///   the format, not a quality choice.
/// - **The hyper-connection pairs stay bf16.** They are small (a rank-320
///   bottleneck over the hyper streams) and the manifest has no slot for them;
///   quantising them would be an unforced error the KLD check would have to
///   justify later.
package enum GGUFResidentWidthPolicy {
    package static func width(role: String, columns: UInt64) -> GGUFResidentWidth {
        let affine = affineBits(role: role)
        // Affine needs whole groups; anything else is the checkpoint's bf16.
        guard affine > 0, columns % UInt64(Quantization.groupSize) == 0 else {
            return .bf16
        }
        return affine == 4 ? .affine4 : .affine8
    }

    /// The width the manifest's slots declare for this role, or 0 for bf16.
    package static func affineBits(role: String) -> Int {
        switch role {
        case "embedding", "lmHead", "pleTable":
            return 4
        case "sharedExpertGate", "sharedExpertUp", "sharedExpertDown":
            return 4
        case "router":
            return 8
        case "qProj", "kProj", "vProj", "oProj",
            "gdnQKV", "gdnZ", "gdnOut",
            "indexerQ", "indexerK":
            return 8
        default:
            // Norms, gates, per-head scalars, PLE projections and the
            // hyper-connection pairs.
            return 0
        }
    }

    /// Bytes an affine tensor occupies: packed weights plus one BF16 scale and
    /// one BF16 bias per group of 64.
    package static func affineBytes(
        rows: UInt64, columns: UInt64, bits: Int
    ) -> UInt64 {
        let weights = rows * columns * UInt64(bits) / 8
        let groups = columns / UInt64(Quantization.groupSize)
        return weights + rows * groups * 4
    }

    package static func bf16Bytes(rows: UInt64, columns: UInt64) -> UInt64 {
        rows * columns * 2
    }
}
