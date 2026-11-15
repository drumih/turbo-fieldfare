//
//  Model+SchemaValidation.swift
//  TinyTitan
//
//  The per-tensor schema validation: role uniformity, family quant support,
//  layer tensor shapes and the routed-expert layout cross-check.
//
import Darwin
import Foundation
import Metal
import TinyTitanFormat
import TinyTitanCore

extension Model {

    /// limit.
    ///
    /// **Naming the slot's width is honoured, not refused.** A manifest that
    /// carries an explicit `in_proj_a` at the slot's 4 bits describes the width
    /// the kernel already uses; refusing it broke a `qwen38flash` install
    /// (issue #16) on load, before any weight was read.
    static func validateRoleUniformity(
        overrides: [String: Int],
        family: ModelFamily,
        attentionBits: Int
    ) throws {
        guard !overrides.isEmpty else { return }
        for (stem, bits) in overrides.sorted(by: { $0.key < $1.key })
        where stem.hasSuffix(".linear_attn.in_proj_a")
            || stem.hasSuffix(".linear_attn.in_proj_b")
        {
            guard bits == 16 || bits == attentionBits else {
                throw ModelError.unsupportedArchitecture(
                    detail: "\(family.rawValue) declares \(stem) at \(bits) bits; the GDN "
                        + "a/b kernel reads that pair at the attention slot's width "
                        + "(\(attentionBits)) or as bf16, so that override is not honoured")
            }
        }
        // The runtime's roles, not one suffix per tensor: q/o share a
        // dispatcher, as do k/v, the three FFN projections and the three GDN
        // ones.
        let roles: [(name: String, suffixes: [String])] = [
            ("qo", [".self_attn.q_proj", ".self_attn.o_proj"]),
            ("kv", [".self_attn.k_proj", ".self_attn.v_proj"]),
            ("ffn", [".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj"]),
            (
                "gdn",
                [
                    ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
                    ".linear_attn.out_proj",
                ]
            ),
            // The three families whose weights read the attention slot until a
            // manifest overrides them. Each is one kernel instance for the
            // whole model, so a manifest that promoted one layer's gate and not
            // the next would read half of them at the wrong width.
            (
                "hyperGate",
                [
                    ".attn_hyper_connection.block_inject_weight",
                    ".mlp_hyper_connection.block_inject_weight",
                ]
            ),
            ("pleKey", [".ple.key_proj"]),
            (
                "qsaIndexer",
                [
                    ".self_attn.indexer.index_q_proj",
                    ".self_attn.indexer.index_k_proj",
                ]
            ),
            ("head", [".lm_head"]),
        ]
        for role in roles {
            var seen: (bits: Int, stem: String)?
            for (stem, bits) in overrides.sorted(by: { $0.key < $1.key })
            where role.suffixes.contains(where: { stem.hasSuffix($0) }) {
                if let seen, seen.bits != bits {
                    throw ModelError.unsupportedArchitecture(
                        detail: "\(family.rawValue) declares \(stem) at \(bits) bits and "
                            + "\(seen.stem) at \(seen.bits); the runtime builds one kernel "
                            + "for the \(role.name) role, so it has to be uniform")
                }
                seen = (bits, stem)
            }
        }
    }

    /// Refuse a width no kernel on the path can execute.
    ///
    /// `HyperConnection`, `PLEBlock` and `QSAIndexer` read weights whose width
    /// is the attention slot's unless the manifest overrides the tensor. They
    /// took a `DequantInt4GEMV` unconditionally until `SlotGEMV` gave them both
    /// paths, and an 8-bit install then read half the bytes of every gate as
    /// nibbles -- no error, no noise, just a model that answered " Paris" and
    /// degenerated.
    ///
    /// They now dispatch on the resolved width, so 4 and 8 are both executable
    /// and only a width neither GEMV implements is refused. The check is on
    /// every one of them rather than on the slot, because a per-tensor override
    /// is exactly how those ~10 MB are promoted without taking the whole
    /// attention block -- 61% of the active parameters -- to 8 bits.
    static func validateFamilyQuantSupport(
        config: ArchConfig,
        quant: ManifestQuant,
        overrides: [String: Int] = [:]
    ) throws {
        guard config.hyperConnections.enabled else { return }
        /// One suffix per kernel that reads through `SlotGEMV`. The
        /// hyper-connection one carries no leading dot: its stems are
        /// `attn_hyper_connection.…` and `mlp_hyper_connection.…`.
        let families: [(name: String, suffix: String)] = [
            ("hyper-connection", "hyper_connection.block_inject_weight"),
            ("PLE", ".ple.key_proj"),
            ("QSA-indexer", ".self_attn.indexer.index_q_proj"),
        ]
        for family in families {
            let declared =
                overrides.first { $0.key.hasSuffix(family.suffix) }?.value
                ?? quant.attention.weightBits
            guard [4, 8].contains(declared) else {
                throw ModelError.unsupportedArchitecture(
                    detail: "\(config.family) runs its \(family.name) projections "
                        + "through SlotGEMV, which implements 4- and 8-bit; this "
                        + "install declares \(declared)-bit.")
            }
        }
    }

    /// Per-layer norms, router, shared expert, attention and GDN tensors.
    static func validateLayerTensors(
        checks: RuntimeSchemaChecks,
        config: ArchConfig,
        quant: ManifestQuant,
        overrides: [String: Int]
    ) throws {
        // Names resolve through the family's schema; only shapes are spelled
        // here. A family whose per-sublayer norm is the hyper-connection's
        // spans the whole residual rather than one stream.
        let schema = TensorSchema.schema(for: config.modelFamily)
        // A dense model has no router and no shared expert: its `mlp.*` FFN is
        // what the schema's shared-expert roles name, and the routed half of
        // the layer does not exist.
        let denseFFN = config.numExperts == 0
        let blockNormWidth =
            config.hyperConnections.enabled
            ? config.hiddenSize * config.hyperConnections.count
            : config.hiddenSize
        for layer in 0..<config.numLayers {
            try checks.requireBF16(schema.inputNorm(layer), count: blockNormWidth)
            try checks.requireBF16(schema.postAttnNorm(layer), count: blockNormWidth)
            if !denseFFN {
                try checks.requireAffineOrBF16(
                    schema.router(layer),
                    rows: config.numExperts, columns: config.hiddenSize,
                    slot: quant.router)
                // The shared-expert scalar gate is quantized at the ROUTER's bit
                // width (8-bit on the target checkpoint, 4-bit on the MTP
                // sidecar), independent of the sharedExpert slot.
                try checks.requireAffineOrBF16(
                    schema.sharedExpertScalarGate(layer),
                    rows: 1, columns: config.hiddenSize,
                    slot: quant.router)
            }
            // Each projection resolves its own width: a dense install declares
            // `mlp.*` per tensor (4-bit) while the sharedExpert slot says 8, and
            // reading the slot there is a silently wrong model, not an error.
            func ffnSlot(_ name: String) -> ManifestQuantSlot {
                quant.slot(
                    forTensorNamed: name, overrides: overrides,
                    fallback: quant.sharedExpert)
            }
            try checks.requireAffineOrBF16(
                schema.sharedExpertGate(layer),
                rows: config.intermediateSize, columns: config.hiddenSize,
                slot: ffnSlot(schema.sharedExpertGate(layer)))
            try checks.requireAffineOrBF16(
                schema.sharedExpertUp(layer),
                rows: config.intermediateSize, columns: config.hiddenSize,
                slot: ffnSlot(schema.sharedExpertUp(layer)))
            try checks.requireAffineOrBF16(
                schema.sharedExpertDown(layer),
                rows: config.hiddenSize, columns: config.intermediateSize,
                slot: ffnSlot(schema.sharedExpertDown(layer)))

            // Each projection resolves its own width: a dense install keeps
            // k/v at 8 bits while the attention slot says 4, and validating
            // against the slot would refuse a correct install (or, worse, pass
            // one whose bytes are later read at the wrong width).
            func roleSlot(_ name: String, _ fallback: ManifestQuantSlot) -> ManifestQuantSlot {
                quant.slot(forTensorNamed: name, overrides: overrides, fallback: fallback)
            }
            if config.layerIsFull(layer) {
                // Gate-packed [query ; gate] q_proj: 2 * heads * headDim rows.
                let queryDimension = try checks.checkedIntMultiply(
                    2 * config.numHeads, config.fullHeadDim,
                    field: "layer \(layer) query")
                let kvDimension = try checks.checkedIntMultiply(
                    config.numFullKVHeads, config.fullHeadDim,
                    field: "layer \(layer) key/value")
                try checks.requireBF16(
                    schema.qNorm(layer),
                    count: config.fullHeadDim)
                try checks.requireBF16(
                    schema.kNorm(layer),
                    count: config.fullHeadDim)
                // A promoted attention projection is legal: the decode path
                // dispatches on the tensor's own dtype (`encodeRoleGEMV`), so a
                // bf16 q/k/v/o runs through `BF16GEMV` with no scales or biases.
                try checks.requireAffineOrBF16(
                    schema.qProj(layer),
                    rows: queryDimension, columns: config.hiddenSize,
                    slot: roleSlot(schema.qProj(layer), quant.attention))
                try checks.requireAffineOrBF16(
                    schema.kProj(layer),
                    rows: kvDimension, columns: config.hiddenSize,
                    slot: roleSlot(schema.kProj(layer), quant.attention))
                try checks.requireAffineOrBF16(
                    schema.vProj(layer),
                    rows: kvDimension, columns: config.hiddenSize,
                    slot: roleSlot(schema.vProj(layer), quant.attention))
                try checks.requireAffineOrBF16(
                    schema.oProj(layer),
                    rows: config.hiddenSize,
                    columns: config.numHeads * config.fullHeadDim,
                    slot: roleSlot(schema.oProj(layer), quant.attention))
            } else if config.layerIsLinear(layer) {
                let la = config.linearAttention
                // Same rule as the full-attention projections: the decode
                // path picks the fused kernel only when every input projection
                // is affine, so a promoted one is served by `BF16GEMV`.
                try checks.requireAffineOrBF16(
                    schema.gdnQKV(layer),
                    rows: la.qkvDim, columns: config.hiddenSize,
                    slot: roleSlot(schema.gdnQKV(layer), quant.attention))
                try checks.requireAffineOrBF16(
                    schema.gdnZ(layer),
                    rows: la.valueDim, columns: config.hiddenSize,
                    slot: roleSlot(schema.gdnZ(layer), quant.attention))
                try checks.requireAffineOrBF16(
                    schema.gdnA(layer),
                    rows: la.numVHeads, columns: config.hiddenSize,
                    slot: quant.attention)
                try checks.requireAffineOrBF16(
                    schema.gdnB(layer),
                    rows: la.numVHeads, columns: config.hiddenSize,
                    slot: quant.attention)
                try checks.requireAffineOrBF16(
                    schema.gdnOut(layer),
                    rows: config.hiddenSize, columns: la.valueDim,
                    slot: roleSlot(schema.gdnOut(layer), quant.attention))
                try checks.requireBF16(
                    schema.gdnConv(layer),
                    count: la.qkvDim * la.convKernelSize)
                try checks.requireBF16OrFP32(schema.gdnALog(layer), count: la.numVHeads)
                try checks.requireBF16OrFP32(schema.gdnDtBias(layer), count: la.numVHeads)
                try checks.requireBF16OrFP32(
                    schema.gdnNorm(layer),
                    count: la.valueHeadDim)
            }
        }

    }

    /// Routed-expert tensor shapes cross-checked against the packed layout.
    static func validateRoutedExpertLayout(
        checks: RuntimeSchemaChecks,
        layout: PackedExpertsLayout,
        config: ArchConfig,
        quant: ManifestQuant
    ) throws {
        let routedShapes: [(String, Int, Int)] = [
            ("gate", config.moeIntermediateSize, config.hiddenSize),
            ("up", config.moeIntermediateSize, config.hiddenSize),
            ("down", config.hiddenSize, config.moeIntermediateSize),
        ]
        for layer in layout.layers {
            // K2's mixed install: its dense layers 0-2 carry no packed experts,
            // and that is the layout's honest shape, not a missing blob. A
            // layer the architecture calls dense must stay empty, and a routed
            // layer must not be empty -- either way round the two claims
            // disagree and the install is wrong.
            let dense = config.denseLayerIndices.contains(layer.layer)
            if layer.experts.isEmpty {
                guard dense else {
                    throw ModelError.indexCorrupt(
                        detail: "routed layer \(layer.layer) has no experts")
                }
                continue
            }
            guard !dense else {
                throw ModelError.indexCorrupt(
                    detail: "dense layer \(layer.layer) carries routed experts")
            }
            for (role, rows, columns) in routedShapes {
                let sizes = try checks.affineSizes(
                    rows: rows, columns: columns,
                    slot: quant.routedExpert,
                    field: "routed layer \(layer.layer) \(role)")
                let expectedRoles: [(String, String, [UInt32], Int?, UInt64, UInt64)] = [
                    (
                        role, "U32", [sizes.shape.0, sizes.shape.1],
                        quant.routedExpert.weightBits, sizes.weight,
                        UInt64(MemoryLayout<UInt32>.alignment)
                    ),
                    (
                        "\(role)_scales", "BF16",
                        [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                        nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)
                    ),
                    (
                        "\(role)_biases", "BF16",
                        [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                        nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)
                    ),
                ]
                for (name, dtype, shape, bits, _, alignment) in expectedRoles {
                    for expert in layer.experts {
                        guard let tensor = expert.subTensors[name] else {
                            throw ModelError.indexCorrupt(
                                detail:
                                    "routed layer \(layer.layer) expert \(expert.expert) is missing role \(name)"
                            )
                        }
                        // A tiered layer keeps hot experts at the wide width and
                        // cold ones at 2-bit, so the byte extent follows the
                        // tensor's own declared width rather than expert 0's.
                        // What still has to hold: the dtype and shape agree with
                        // the architecture, the offset is aligned, the extent is
                        // exactly shape * width, and it fits inside the blob.
                        let width = tensor.bits ?? bits ?? 0
                        let expectedSize: UInt64 =
                            dtype == "U32"
                            ? UInt64(shape[0]) * UInt64(shape[1]) * UInt64(width) / 8
                            : UInt64(shape[0]) * UInt64(shape[1]) * 2
                        let (end, overflow) = tensor.offset.addingReportingOverflow(
                            tensor.size)
                        guard tensor.dtype == dtype,
                            tensor.shape == shape,
                            tensor.offset % alignment == 0,
                            tensor.size == expectedSize,
                            !overflow,
                            end <= expert.size,
                            end <= UInt64(UInt32.max) + 1
                        else {
                            throw ModelError.indexCorrupt(
                                detail:
                                    "routed layer \(layer.layer) expert \(expert.expert) role \(name) does not match the required schema (width \(width), size \(tensor.size), expected \(expectedSize))"
                            )
                        }
                    }
                }
            }
        }
    }
}
