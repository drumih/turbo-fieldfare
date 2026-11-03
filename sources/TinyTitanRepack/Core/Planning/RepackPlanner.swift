import Foundation

/// On-disk page alignment unit for `.ssdai` files. Fixed at 16 KB regardless
/// of host page size — the format is the contract, not the kernel.

extension RepackPlanner {
    /// The largest single non-tensor file the repacker will carry, and therefore
    /// the smallest per-file cap a receipt verifier may use.
    ///
    /// `VerifiedInstallTool.payloadMaxBytes` is that cap and used to be pinned at
    /// 128 GiB while this table admitted 256 GiB for `ngram_table.bin`: an install
    /// the repacker had just written by design was one `--verify-install`
    /// refused, with "manifest size … exceeds the per-file cap" naming that file.
    /// Both read this constant now, so they cannot drift apart again.
    static let maximumPassthroughFileBytes: UInt64 = 256 * 1024 * 1024 * 1024

    /// Non-tensor files this family needs copied into the install. Sizes are
    /// resolved from the remote before planning, because they are standalone
    /// files rather than entries in the safetensors index.
    static func passthroughRequirements(
        family: RepackModelFamily
    ) -> [(name: String, required: Bool, capBytes: UInt64)] {
        switch family {
        case .qwen36, .qwen36MTP, .qwen38flashMTP, .qwen4flashMTP:
            // The draft needs no sidecar files of its own: it has no n-gram
            // block, and its embedding and head are the target's.
            return []
        case .qwen35Dense, .k2Horizon, .laguna:
            // A dense Qwen 3.5 has no hashed n-gram block and no PLE
            // constants; its tokenizer travels with the payload like every
            // other family's. K2-Horizon likewise carries only its tokenizer
            // and template.
            return []
        case .qwen38flash, .qwen4flash:
            return [
                // The PLE hash constants: multipliers, per-head offsets and
                // prime vocabulary sizes. Without them the n-gram ids cannot
                // be computed at all, so this one is required.
                ("ple_constants.json", true, 1 * 1024 * 1024),
                // The table itself. Optional by design: the upstream runtime
                // skips the PLE block and stays coherent without it, so a
                // 102 GB download should not be forced on someone who wants
                // the backbone first.
                ("ngram_table.bin", false, maximumPassthroughFileBytes),
            ]
        }
    }
}

// MARK: - Planner

enum RepackPlanner {

    /// Classify a tensor name. Routed-expert tensors split off the LM bucket.
    enum Bucket: Equatable {
        case lmResident
        case routedExpert(role: String, layer: Int)  // role = "gate"|"up"|"down"
        case excludedMultimodal
        /// Belongs to a sidecar installed separately, not to this model's
        /// payload. Qwen3.8-Flash-Next ships its MTP draft inside the target's
        /// index; the draft carries its own 512-expert set and is installed as
        /// its own directory, exactly like the Ornith MTP sidecar.
        case excludedSidecar
        case unknown
    }

    static func classify(
        _ name: String, numLayers: Int,
        family: RepackModelFamily
    ) -> Bucket {
        if family == .qwen36MTP {
            if name.hasPrefix("layers.") {
                if let role = routedExpertRole(in: name),
                    let layer = layerIndex(in: name),
                    layer >= 0 && layer < numLayers
                {
                    return .routedExpert(role: role, layer: layer)
                }
                return .lmResident
            }
            if name == "norm.weight" || name.hasPrefix("fc.")
                || name.hasPrefix("pre_fc_norm_")
            {
                return .lmResident
            }
            return .unknown
        }
        if family == .qwen38flashMTP || family == .qwen4flashMTP {
            // Mirror image of the target's rule: this install is only the
            // `mtp.*` namespace, and everything else belongs to the model it
            // drafts for. The prefix is stripped so the sidecar reads as the
            // one-layer model it is.
            guard name.hasPrefix("mtp.") else { return .excludedSidecar }
            let stripped = String(name.dropFirst("mtp.".count))
            if let role = routedExpertRole(in: stripped),
                let layer = layerIndex(in: stripped),
                layer >= 0 && layer < numLayers
            {
                return .routedExpert(role: role, layer: layer)
            }
            return .lmResident
        }
        if family == .qwen38flash || family == .qwen4flash {
            // The MTP draft rides in the target's index but is a separate
            // install; skip it here rather than folding a second model's
            // experts into this payload.
            if name.hasPrefix("mtp.") { return .excludedSidecar }
            if isMultimodalTensorName(name) { return .excludedMultimodal }
            // `lm_head.*` sits at the top level in this family, not under the
            // language-model prefix the way qwen36 spells it.
            if name.hasPrefix("lm_head.") { return .lmResident }
            if name.hasPrefix("model.language_model.") {
                if let role = routedExpertRole(in: name),
                    let layer = layerIndex(in: name),
                    layer >= 0 && layer < numLayers
                {
                    return .routedExpert(role: role, layer: layer)
                }
                return .lmResident
            }
            return .unknown
        }
        if family == .laguna {
            // Laguna's snapshot keeps the checkpoint's names, with the 256
            // experts already fused into `mlp.switch_mlp.*` by the converter.
            if isMultimodalTensorName(name) { return .excludedMultimodal }
            if let role = routedExpertRole(in: name),
                let layer = layerIndex(in: name),
                layer >= 0, layer < numLayers
            {
                return .routedExpert(role: role, layer: layer)
            }
            if name.hasPrefix("model.") || name.hasPrefix("lm_head.") {
                return .lmResident
            }
            return .unknown
        }
        if family == .k2Horizon {
            // K2's snapshot keeps the checkpoint's names: the MoE experts are
            // fused into the repacker's `switch_mlp` spelling by the
            // converter, and everything else -- attention, the attention gate,
            // the value router, the 64 MoVA experts, the norms, the embedding
            // and the head -- is resident.
            if isMultimodalTensorName(name) { return .excludedMultimodal }
            if let role = routedExpertRole(in: name),
                let layer = layerIndex(in: name),
                layer >= 0, layer < numLayers
            {
                return .routedExpert(role: role, layer: layer)
            }
            if name.hasPrefix("model.") || name.hasPrefix("lm_head.") {
                return .lmResident
            }
            return .unknown
        }
        if name.hasPrefix("language_model.") {
            // Routed expert?
            if let role = routedExpertRole(in: name),
                let layer = layerIndex(in: name),
                layer >= 0 && layer < numLayers
            {
                return .routedExpert(role: role, layer: layer)
            }
            return .lmResident
        }
        if isMultimodalTensorName(name) {
            return .excludedMultimodal
        }
        return .unknown
    }

}
