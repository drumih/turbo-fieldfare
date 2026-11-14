import Testing

@testable import TinyTitan
import TinyTitanCore

/// The family tensor schemas are the single source of on-disk names; these
/// pins keep a schema edit from silently un-mapping an installed `.ssdai`.
@Suite struct FamilySchemaTests {
    @Test func qwen36SchemaMatchesTheRepackedNames() {
        let schema = TensorSchema.schema(for: .qwen36)
        #expect(schema.embedding == "language_model.model.embed_tokens.weight")
        #expect(schema.lmHead == "language_model.lm_head.weight")
        #expect(schema.finalNorm == "language_model.model.norm.weight")
        #expect(
            schema.qProj(7)
                == "language_model.model.layers.7.self_attn.q_proj.weight")
        #expect(
            schema.router(0)
                == "language_model.model.layers.0.mlp.gate.weight")
        #expect(
            schema.sharedExpertScalarGate(39)
                == "language_model.model.layers.39.mlp.shared_expert_gate.weight")
        #expect(
            schema.inputNorm(3)
                == "language_model.model.layers.3.input_layernorm.weight")
    }

    @Test func familiesShareTheQwenNamingWhereVerified() {
        // The MTP sidecar reuses the layer naming, and Qwen3.8-Flash-Next's
        // shared roles keep the Qwen3.5 names.
        let mtp = TensorSchema.schema(for: .qwen36MTP)
        #expect(mtp.qProj(3) == TensorSchema.schema(for: .qwen36).qProj(3))
        #expect(mtp.embedding == TensorSchema.schema(for: .qwen36).embedding)
    }

    /// Qwen3.8-Flash-Next deliberately does NOT share the naming. It aliased
    /// qwen36 while it was a placeholder; now that the real schema is in
    /// place, asserting the alias would assert a bug. Its names are checked
    /// against the real checkpoint index in Qwen38FlashSchemaTests.
    @Test func qwen38FlashUsesItsOwnNaming() {
        let flash = TensorSchema.schema(for: .qwen38flash)
        let qwen = TensorSchema.schema(for: .qwen36)
        // Mirror-image prefixes: model.language_model vs language_model.model.
        #expect(flash.qProj(3) != qwen.qProj(3))
        #expect(flash.embedding != qwen.embedding)
        #expect(flash.lmHead != qwen.lmHead)
        #expect(flash.embedding.hasPrefix("model.language_model."))
        #expect(flash.lmHead == "lm_head.weight")
    }

    /// The dense family is served by the CPU engine, which resolves every
    /// tensor through this table, so a typo here is a load failure on a real
    /// install rather than a test failure. The GDN bundle and the two stems
    /// without a `.weight` suffix (`A_log`, `dt_bias`) are the parts worth
    /// pinning: they are the only names in the file that break the pattern.
    @Test func qwen35DenseSchemaMatchesTheRepackedNames() {
        let dense = TensorSchema.schema(for: .qwen35Dense)
        #expect(dense.embedding == "language_model.model.embed_tokens.weight")
        #expect(dense.lmHead == "language_model.lm_head.weight")
        #expect(dense.finalNorm == "language_model.model.norm.weight")
        let l = 5
        let p = "language_model.model.layers.\(l)."
        #expect(dense.qProj(l) == "\(p)self_attn.q_proj.weight")
        #expect(dense.kProj(l) == "\(p)self_attn.k_proj.weight")
        #expect(dense.vProj(l) == "\(p)self_attn.v_proj.weight")
        #expect(dense.oProj(l) == "\(p)self_attn.o_proj.weight")
        #expect(dense.qNorm(l) == "\(p)self_attn.q_norm.weight")
        #expect(dense.kNorm(l) == "\(p)self_attn.k_norm.weight")
        #expect(dense.inputNorm(l) == "\(p)input_layernorm.weight")
        #expect(dense.postAttnNorm(l) == "\(p)post_attention_layernorm.weight")
        #expect(dense.gdnQKV(l) == "\(p)linear_attn.in_proj_qkv.weight")
        #expect(dense.gdnZ(l) == "\(p)linear_attn.in_proj_z.weight")
        #expect(dense.gdnA(l) == "\(p)linear_attn.in_proj_a.weight")
        #expect(dense.gdnB(l) == "\(p)linear_attn.in_proj_b.weight")
        #expect(dense.gdnOut(l) == "\(p)linear_attn.out_proj.weight")
        #expect(dense.gdnConv(l) == "\(p)linear_attn.conv1d.weight")
        #expect(dense.gdnALog(l) == "\(p)linear_attn.A_log")
        #expect(dense.gdnDtBias(l) == "\(p)linear_attn.dt_bias")
        #expect(dense.gdnNorm(l) == "\(p)linear_attn.norm.weight")
    }

    /// The dense FFN is a plain SwiGLU spelled through the shared-expert roles,
    /// which is the stage the runner already has. The name it must NOT be
    /// confused with is the MoE router's, whose suffix is one character away.
    @Test func denseSpellsItsFFNThroughTheSharedExpertRoles() {
        let dense = TensorSchema.schema(for: .qwen35Dense)
        let moe = TensorSchema.schema(for: .qwen36)
        #expect(dense.sharedExpertGate(2) == "language_model.model.layers.2.mlp.gate_proj.weight")
        #expect(dense.sharedExpertUp(2) == "language_model.model.layers.2.mlp.up_proj.weight")
        #expect(dense.sharedExpertDown(2) == "language_model.model.layers.2.mlp.down_proj.weight")
        // `mlp.gate.weight` is the router. A dense schema that pointed the
        // shared-expert gate there would load an FFN as a router.
        #expect(moe.router(2) == "language_model.model.layers.2.mlp.gate.weight")
        #expect(dense.sharedExpertGate(2) != moe.router(2))
        #expect(dense.sharedExpertGate(2) != dense.router(2))
    }

    /// A role the family does not have is pointed at a name that cannot exist,
    /// on purpose: the lookup then fails and names itself, while a
    /// plausible-looking name would read the wrong tensor silently.
    @Test func denseRolesThatDoNotExistNameThemselvesAsImpossible() {
        let dense = TensorSchema.schema(for: .qwen35Dense)
        for role in [dense.router, dense.sharedExpertScalarGate] {
            // Constant across layers, and outside the layer namespace entirely.
            #expect(role(0) == role(39))
            #expect(role(0).contains("layers.") == false)
            #expect(role(0).isEmpty == false)
        }
        // No two roles resolve to the same tensor for one layer, which is how a
        // copied line in the table would show up.
        let names = Self.layerRoles(of: dense).map { $0(7) }
        #expect(Set(names).count == names.count)
    }

    /// Every per-layer role except the two impossible ones has to depend on its
    /// argument: a closure that ignores it would read layer 0's tensor for all
    /// 48 layers and still find a tensor, so nothing downstream would complain.
    @Test func everyDenseLayerRoleDependsOnTheLayer() {
        let dense = TensorSchema.schema(for: .qwen35Dense)
        let impossible = Set([dense.router(0), dense.sharedExpertScalarGate(0)])
        for role in Self.layerRoles(of: dense) {
            let name = role(3)
            if impossible.contains(name) { continue }
            #expect(name.contains(".3."))
            #expect(role(4) != name)
        }
    }

    private static func layerRoles(
        of schema: TensorSchema
    ) -> [@Sendable (Int) -> String] {
        [
            schema.qProj, schema.kProj, schema.vProj, schema.oProj, schema.router,
            schema.sharedExpertGate, schema.sharedExpertUp, schema.sharedExpertDown,
            schema.sharedExpertScalarGate, schema.inputNorm, schema.postAttnNorm,
            schema.qNorm, schema.kNorm, schema.gdnQKV, schema.gdnZ, schema.gdnA,
            schema.gdnB, schema.gdnOut, schema.gdnConv, schema.gdnALog,
            schema.gdnDtBias, schema.gdnNorm,
        ]
    }
}
