import Foundation

package enum GGUFLinearAttentionConfigError: Error, Equatable, CustomStringConvertible {
    case missingKey(String)
    case inconsistent(String)

    package var description: String {
        switch self {
        case .missingKey(let key):
            return "the checkpoint declares no \(key)"
        case .inconsistent(let detail):
            return "the checkpoint's linear-attention geometry is inconsistent: \(detail)"
        }
    }
}

/// Linear-attention (GDN) geometry taken from the checkpoint's own metadata.
///
/// The runtime preset for this family happens to agree with this checkpoint
/// (`ssm.time_step_rank` 48, `ssm.inner_size` 6144 = 48 x 128), but a family
/// preset is not evidence about a particular file: another variant of the same
/// family can ship different head counts, and the GDN kernels would then read a
/// shorter qkv and value projection than the file stores - wrong numbers, no
/// error. So the loader reads the geometry from the file and checks the derived
/// widths against the tensors it is about to read.
package struct GGUFLinearAttentionConfig {
    package let numKHeads: Int
    package let numVHeads: Int
    package let keyHeadDim: Int
    package let valueHeadDim: Int
    package let convKernelSize: Int

    package var qkvDim: Int { 2 * numKHeads * keyHeadDim + numVHeads * valueHeadDim }
    package var valueDim: Int { numVHeads * valueHeadDim }

    package static func read(
        from model: GGUFModel
    ) throws -> GGUFLinearAttentionConfig {
        func value(_ key: String) throws -> Int {
            guard case .uint32(let value)? = model.metadata["qwen4exp.\(key)"] else {
                throw GGUFLinearAttentionConfigError.missingKey("qwen4exp.\(key)")
            }
            return Int(value)
        }
        let groupCount = try value("ssm.group_count")
        let timeStepRank = try value("ssm.time_step_rank")
        let innerSize = try value("ssm.inner_size")
        let stateSize = try value("ssm.state_size")
        let convKernel = try value("ssm.conv_kernel")
        guard groupCount > 0, timeStepRank > 0, stateSize > 0, convKernel > 0 else {
            throw GGUFLinearAttentionConfigError.inconsistent("non-positive dimension")
        }
        guard innerSize % timeStepRank == 0 else {
            throw GGUFLinearAttentionConfigError.inconsistent(
                "ssm.inner_size \(innerSize) is not a multiple of time_step_rank \(timeStepRank)")
        }
        let config = GGUFLinearAttentionConfig(
            numKHeads: groupCount, numVHeads: timeStepRank,
            keyHeadDim: stateSize, valueHeadDim: innerSize / timeStepRank,
            convKernelSize: convKernel)
        // The derived widths must be the ones the tensors actually are.
        if let qkv = model.tensor(named: "blk.0.attn_qkv.weight") {
            guard Int(qkv.tensor.dimensions[1]) == config.qkvDim else {
                throw GGUFLinearAttentionConfigError.inconsistent(
                    "attn_qkv has \(qkv.tensor.dimensions[1]) rows, derived \(config.qkvDim)")
            }
        }
        if let gate = model.tensor(named: "blk.0.attn_gate.weight") {
            guard Int(gate.tensor.dimensions[1]) == config.valueDim else {
                throw GGUFLinearAttentionConfigError.inconsistent(
                    "attn_gate has \(gate.tensor.dimensions[1]) rows, derived \(config.valueDim)")
            }
        }
        return config
    }
}
