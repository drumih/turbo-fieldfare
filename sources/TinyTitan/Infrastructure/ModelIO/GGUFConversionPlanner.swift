import Foundation

package struct GGUFResidentTensorPlan: Sendable, Equatable {
    package let name: String
    package let role: String
    /// 4, 8 or 16 (bf16).
    package let bits: Int
    package let rows: UInt64
    package let columns: UInt64
    package let bytes: UInt64
}

/// What a full conversion would write, in bytes: the routed experts, every
/// resident tensor at the width the policy assigns it, and the totals.
///
/// No bytes are written and no file is created. This is the plan a full run is
/// budgeted against, and the place the widths are checked against the shapes
/// before a 100 GB write starts.
/// A tensor the policy wanted to quantize but the format could not describe:
/// its row width is not a multiple of the kernel's group, so it fell back to
/// bf16. The PLE table is the one that matters — at 160 wide it cannot be
/// group-64 affine, and bf16 costs 102.4 GB instead of ~32 GB at group 32.
package struct GGUFFormatFallback: Sendable, Equatable {
    package let name: String
    package let role: String
    package let columns: UInt64
    package let intendedBits: Int
    package let bf16Bytes: UInt64
    /// What it would cost at the same width with a group of 32, which is the
    /// group the checkpoint's own quants use along this axis.
    package let group32Bytes: UInt64
}

package struct GGUFConversionPlan: Sendable {
    package let resident: [GGUFResidentTensorPlan]
    package let residentBytes: UInt64
    package let bytesByBits: [Int: UInt64]
    package let expertBytes: UInt64
    package let formatFallbacks: [GGUFFormatFallback]
    package var totalBytes: UInt64 { residentBytes + expertBytes }
}

package enum GGUFConversionPlanError: Error, Equatable, CustomStringConvertible {
    case notGroupAligned(name: String, role: String, columns: UInt64)

    package var description: String {
        switch self {
        case .notGroupAligned(let name, let role, let columns):
            return
                "\(name) (\(role)) is \(columns) wide but was assigned an affine width"
        }
    }
}

package enum GGUFConversionPlanner {
    package static func plan(
        from model: GGUFModel, arch: ArchConfig = .qwen38FlashNext
    ) throws -> GGUFConversionPlan {
        var resident: [GGUFResidentTensorPlan] = []
        var fallbacks: [GGUFFormatFallback] = []
        var byBits: [Int: UInt64] = [:]
        var residentBytes: UInt64 = 0
        for (name, location) in model.tensors.sorted(by: { $0.key < $1.key }) {
            if name.hasSuffix("_exps.weight") { continue }
            let role = try GGUFResidentRoleMap.role(for: name)
            let dimensions = location.tensor.dimensions
            let columns = dimensions[0]
            let rows = dimensions.count > 1 ? dimensions[1] : 1
            let width = GGUFResidentWidthPolicy.width(role: role, columns: columns)
            let bytes: UInt64
            switch width {
            case .affine4, .affine8:
                guard columns % UInt64(Quantization.groupSize) == 0 else {
                    throw GGUFConversionPlanError.notGroupAligned(
                        name: name, role: role, columns: columns)
                }
                bytes = GGUFResidentWidthPolicy.affineBytes(
                    rows: rows, columns: columns, bits: width.bits)
            case .bf16:
                bytes = GGUFResidentWidthPolicy.bf16Bytes(rows: rows, columns: columns)
                let intended = GGUFResidentWidthPolicy.affineBits(role: role)
                if intended > 0, columns % UInt64(Quantization.groupSize) != 0 {
                    let groups32 = columns / 32
                    fallbacks.append(
                        GGUFFormatFallback(
                            name: name, role: role, columns: columns,
                            intendedBits: intended,
                            bf16Bytes: bytes,
                            group32Bytes: rows * columns * UInt64(intended) / 8
                                + rows * groups32 * 4))
                }
            }
            resident.append(
                GGUFResidentTensorPlan(
                    name: name, role: role, bits: width.bits,
                    rows: rows, columns: columns, bytes: bytes))
            residentBytes += bytes
            byBits[width.bits, default: 0] += bytes
        }

        // Experts are uniform in shape, so one layer's padded stride times the
        // expert count times the layer count is exact; each layer is asked for
        // its own stride rather than assuming they all match.
        var expertBytes: UInt64 = 0
        let experts = try GGUFExpertLayout.expertsPerLayer(in: model)
        for layer in 0..<arch.numLayers {
            let layout = try GGUFExpertBlobWriter.expertLayout(layer: layer, model: model)
            expertBytes += layout.stride * UInt64(experts)
        }

        return GGUFConversionPlan(
            resident: resident, residentBytes: residentBytes,
            bytesByBits: byBits, expertBytes: expertBytes,
            formatFallbacks: fallbacks)
    }
}
