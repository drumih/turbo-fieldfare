import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// Every tensor name this family's schema produces is checked against the
/// pinned checkpoint's real index. A schema that merely compiles proves
/// nothing: the failure mode here is a plausible-looking name that does not
/// exist, which surfaces only as a load error against a 174 GB install.
@Suite("Qwen38Flash tensor schema")
struct Qwen38FlashSchemaTests {
    static let names: Set<String> = {
        guard
            let url = Bundle.module.url(
                forResource: "qwen38_tensor_names",
                withExtension: "txt"),
            let text = try? String(contentsOf: url, encoding: .utf8)
        else { return [] }
        return Set(text.split(separator: "\n").map(String.init))
    }()

    /// Quantized tensors are stored as a `.weight` base plus companions; the
    /// schema names the base, which is what the resident index keys on.
    private func exists(_ name: String) -> Bool {
        Self.names.contains(name)
    }

    @Test("The fixture loaded")
    func fixturePresent() {
        #expect(Self.names.count == 3164)
    }

    @Test("Model-level roles resolve to real tensors")
    func modelLevelRoles() {
        let s = TensorSchema.schema(for: .qwen38flash)
        #expect(exists(s.embedding), "\(s.embedding)")
        #expect(exists(s.lmHead), "\(s.lmHead)")
        // No `model.norm` in this family: the final norm is the mixer's.
        #expect(exists(s.finalNorm), "\(s.finalNorm)")
    }

    @Test("Attention roles resolve on every full-attention layer")
    func attentionRoles() {
        let s = TensorSchema.schema(for: .qwen38flash)
        let full = stride(from: 3, to: 48, by: 4)
        for l in full {
            #expect(exists(s.qProj(l)), "\(s.qProj(l))")
            #expect(exists(s.kProj(l)), "\(s.kProj(l))")
            #expect(exists(s.vProj(l)), "\(s.vProj(l))")
            #expect(exists(s.oProj(l)), "\(s.oProj(l))")
            #expect(exists(Qwen38FlashTensors.qNorm(l)))
            #expect(exists(Qwen38FlashTensors.kNorm(l)))
            #expect(exists(Qwen38FlashTensors.indexerQProj(l)))
            #expect(exists(Qwen38FlashTensors.indexerKProj(l)))
            #expect(exists(Qwen38FlashTensors.indexerQNorm(l)))
            #expect(exists(Qwen38FlashTensors.indexerKNorm(l)))
        }
    }

    @Test("MoE roles resolve on every layer")
    func moeRoles() {
        let s = TensorSchema.schema(for: .qwen38flash)
        for l in 0..<48 {
            #expect(exists(s.router(l)), "\(s.router(l))")
            #expect(exists(s.sharedExpertGate(l)))
            #expect(exists(s.sharedExpertUp(l)))
            #expect(exists(s.sharedExpertDown(l)))
            #expect(exists(s.sharedExpertScalarGate(l)))
        }
    }

    @Test("The norm roles point at hyper-connection norms, which exist")
    func normRolesAreHyperConnectionNorms() {
        let s = TensorSchema.schema(for: .qwen38flash)
        for l in 0..<48 {
            #expect(exists(s.inputNorm(l)), "\(s.inputNorm(l))")
            #expect(exists(s.postAttnNorm(l)), "\(s.postAttnNorm(l))")
        }
        // And the layer norms a plain transformer would have really are absent,
        // so pointing these roles anywhere else would have failed at load.
        #expect(
            !Self.names.contains(
                "model.language_model.layers.0.input_layernorm.weight"))
        #expect(
            !Self.names.contains(
                "model.language_model.layers.0.post_attention_layernorm.weight"))
        #expect(!Self.names.contains("model.language_model.norm.weight"))
    }

    @Test("Hyper-connection tensors resolve on every layer")
    func hyperConnectionRoles() {
        for l in 0..<48 {
            #expect(exists(Qwen38FlashTensors.attnMixDown(l)))
            #expect(exists(Qwen38FlashTensors.attnMixUp(l)))
            #expect(exists(Qwen38FlashTensors.attnInject(l)))
            #expect(exists(Qwen38FlashTensors.mlpMixDown(l)))
            #expect(exists(Qwen38FlashTensors.mlpMixUp(l)))
            #expect(exists(Qwen38FlashTensors.mlpInject(l)))
        }
        #expect(exists(Qwen38FlashTensors.mixerDown))
        #expect(exists(Qwen38FlashTensors.mixerUp))
    }

    @Test("GDN roles resolve on linear-attention layers only")
    func gdnRoles() {
        let linear = (0..<48).filter { ($0 + 1) % 4 != 0 }
        #expect(linear.count == 36)
        for l in linear {
            #expect(exists(Qwen38FlashTensors.gdnQKV(l)))
            #expect(exists(Qwen38FlashTensors.gdnZ(l)))
            #expect(exists(Qwen38FlashTensors.gdnA(l)))
            #expect(exists(Qwen38FlashTensors.gdnB(l)))
            #expect(exists(Qwen38FlashTensors.gdnConv(l)))
            #expect(exists(Qwen38FlashTensors.gdnALog(l)))
            #expect(exists(Qwen38FlashTensors.gdnDtBias(l)))
            #expect(exists(Qwen38FlashTensors.gdnNorm(l)))
            #expect(exists(Qwen38FlashTensors.gdnOut(l)))
        }
        // A GDN tensor must NOT exist on a full-attention layer.
        #expect(!Self.names.contains(Qwen38FlashTensors.gdnQKV(3)))
    }

    @Test("The PLE block exists on exactly the configured layer")
    func pleRole() {
        let cfg = ArchConfig.qwen38FlashNext
        #expect(cfg.ple.layerIndices == [1])
        for l in cfg.ple.layerIndices {
            #expect(
                exists(Qwen38FlashTensors.pleConv(l)),
                "\(Qwen38FlashTensors.pleConv(l))")
        }
        // And nowhere else -- the table is one layer's, not every layer's.
        for l in 0..<48 where !cfg.ple.layerIndices.contains(l) {
            #expect(!Self.names.contains(Qwen38FlashTensors.pleConv(l)))
        }
    }

    @Test("The schema no longer aliases qwen36")
    func notAliasingQwen36() {
        let flash = TensorSchema.schema(for: .qwen38flash)
        let qwen = TensorSchema.schema(for: .qwen36)
        #expect(flash.embedding != qwen.embedding)
        #expect(flash.lmHead != qwen.lmHead)
        #expect(flash.qProj(3) != qwen.qProj(3))
        // The qwen36 names must not resolve against this checkpoint at all.
        #expect(!Self.names.contains(qwen.embedding))
        #expect(!Self.names.contains(qwen.qProj(3)))
    }
}

/// The residual's width, which is the one scratch buffer a hyper-connection
/// family sizes differently. Every other buffer stays `D`, because the gated
/// read collapses the streams before any block sees them.
@Suite("Residual width")
struct ResidualWidthTests {
    @Test("Hyper-connection families carry hc_count streams")
    func flashCarriesFourStreams() {
        let cfg = ArchConfig.qwen38FlashNext
        #expect(cfg.hyperConnections.enabled)
        #expect(cfg.hyperConnections.count == 4)
        #expect(cfg.hiddenSize * cfg.hyperConnections.count == 10_240)
    }

    @Test("Families without them are unchanged at D")
    func qwen36IsUnchanged() {
        let cfg = ArchConfig.qwen36_35B_A3B
        #expect(!cfg.hyperConnections.enabled)
        // The allocation keys off `enabled`, so this family must keep taking
        // exactly D -- a widened residual here would change every downstream
        // offset and break the goldens.
        #expect(cfg.hyperConnections.count == 0)
    }

    static func slot(_ bits: Int) -> ManifestQuantSlot {
        ManifestQuantSlot(
            weightBits: bits, scheme: "affine",
            scaleType: "BF16", biasType: "BF16", groupSize: 64)
    }

    static func quant(attention: Int, routed: Int = 4) -> ManifestQuant {
        ManifestQuant(
            embedding: slot(8), attention: slot(attention),
            router: slot(8), sharedExpert: slot(routed),
            routedExpert: slot(routed))
    }

    /// These three kernels took a `DequantInt4GEMV` unconditionally, so an
    /// 8-bit install read half the bytes of every gate as nibbles: no error,
    /// no noise, a model that answered " Paris" and degenerated. `SlotGEMV`
    /// gives them both paths, so both widths are now executable.
    @Test("Both supported widths are accepted for hyper-connection families")
    func acceptsBothWidths() throws {
        for bits in [4, 8] {
            try Model.validateFamilyQuantSupport(
                config: .qwen38FlashNext, quant: Self.quant(attention: bits))
        }
    }

    /// The guard stays because the failure it catches is silent: a width no
    /// GEMV implements would otherwise reach these kernels and be misread
    /// rather than refused.
    @Test("An unimplemented width is still refused")
    func refusesUnimplementedWidth() {
        #expect(throws: ModelError.self) {
            try Model.validateFamilyQuantSupport(
                config: .qwen38FlashNext, quant: Self.quant(attention: 6))
        }
    }

    @Test("Families without hyper-connections are unaffected")
    func qwen36AcceptsEightBit() throws {
        try Model.validateFamilyQuantSupport(
            config: .qwen36_35B_A3B, quant: Self.quant(attention: 8, routed: 8))
    }

    @Test("Only the residual widens; block-facing buffers stay D")
    func onlyResidualWidens() {
        let cfg = ArchConfig.qwen38FlashNext
        // The gated read produces one D-wide vector, so attention, the MoE and
        // every scratch after them see 2560 regardless of stream count.
        #expect(cfg.hiddenSize == 2560)
        #expect(cfg.moeIntermediateSize == 640)
    }
}
