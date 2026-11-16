import Foundation
import Testing
@testable import TinyTitanCore

@testable import TinyTitan

/// A manifest's per-tensor widths, and the two limits the runtime puts on them.
///
/// Per-tensor widths are how a dense install keeps `mlp.*` at 4 bits against an
/// 8-bit slot and full-attention `k_proj`/`v_proj` at 8 against a 4-bit one, so
/// most of the path is exercised by real installs. These pin the limits: a role
/// has to agree with itself, and the one pair the per-tensor path cannot reach
/// is named rather than left to fail a size check that reads like corruption.
///
/// The `attentionBits` argument is the rule issue #16 turned on: the GDN `a`/`b`
/// kernel reads at the attention slot's width *or* bf16, so an override naming
/// the slot's width is honoured, and only a width the kernel cannot read is a
/// defect.
@Suite struct RoleUniformityTests {
    private let layer = "language_model.model.layers.0"

    @Test func aUniformRoleOverrideIsAccepted() throws {
        try Model.validateRoleUniformity(
            overrides: [
                "\(layer).self_attn.q_proj": 8,
                "\(layer).self_attn.o_proj": 8,
            ],
            family: .qwen36, attentionBits: 4)
    }

    @Test func aRoleThatDisagreesWithItselfIsRefused() {
        #expect(throws: ModelError.self) {
            try Model.validateRoleUniformity(
                overrides: [
                    "\(layer).self_attn.k_proj": 8,
                    "\(layer).self_attn.v_proj": 4,
                ],
                family: .qwen36, attentionBits: 4)
        }
    }

    @Test func aQuantizedOverrideOnTheGdnABPairIsNamed() {
        do {
            // 8 against a 4-bit slot is a width the kernel cannot read.
            try Model.validateRoleUniformity(
                overrides: ["\(layer).linear_attn.in_proj_a": 8],
                family: .qwen36, attentionBits: 4)
            Issue.record("an override the kernel cannot read must be refused")
        } catch let error as ModelError {
            // The message has to say what it is: the size mismatch this used to
            // produce ("in_proj_a.weight size N does not match expected M")
            // reads as corruption rather than as a limit.
            let text = "\(error)"
            #expect(text.contains("in_proj_a"))
            #expect(text.contains("not honoured"))
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }

    /// Issue #16: a `qwen38flash` install whose manifest names `in_proj_a` at
    /// the attention slot's own 4 bits was refused on load, though the kernel
    /// already reads the pair at that width.
    @Test func anOverrideAtTheAttentionSlotWidthIsAccepted() throws {
        try Model.validateRoleUniformity(
            overrides: [
                "\(layer).linear_attn.in_proj_a": 4,
                "\(layer).linear_attn.in_proj_b": 4,
            ],
            family: .qwen38flash, attentionBits: 4)
    }

    @Test func anOverrideThatDiffersFromTheSlotIsRefused() {
        // 4 against an 8-bit slot, and 8 against a 4-bit one, are both widths
        // the bf16-or-slot kernel will not read.
        #expect(throws: ModelError.self) {
            try Model.validateRoleUniformity(
                overrides: ["\(layer).linear_attn.in_proj_a": 4],
                family: .qwen38flash, attentionBits: 8)
        }
        #expect(throws: ModelError.self) {
            try Model.validateRoleUniformity(
                overrides: ["\(layer).linear_attn.in_proj_b": 8],
                family: .qwen38flash, attentionBits: 4)
        }
    }

    @Test func aBf16OverrideOnThePairIsAccepted() throws {
        // 16 is the promoted width the kernel can read for this pair.
        try Model.validateRoleUniformity(
            overrides: ["\(layer).linear_attn.in_proj_b": 16],
            family: .qwen36, attentionBits: 4)
    }

    @Test func anEmptyOverrideMapIsFine() throws {
        try Model.validateRoleUniformity(
            overrides: [:], family: .qwen36,
            attentionBits: 4)
    }

    @Test func theThreeSlotReadingFamiliesHaveRoles() throws {
        // Hyper-connection gates, the PLE key projection and the sparse
        // indexer's keys read the attention slot until a manifest overrides
        // them; each is one kernel instance for the whole model, so a uniform
        // override across the family is what makes promoting the ~10 MB
        // possible without taking the whole attention block to 8 bits.
        try Model.validateRoleUniformity(
            overrides: [
                "\(layer).attn_hyper_connection.block_inject_weight": 8,
                "\(layer).mlp_hyper_connection.block_inject_weight": 8,
                "\(layer).ple.key_proj": 8,
                "\(layer).self_attn.indexer.index_q_proj": 8,
                "\(layer).self_attn.indexer.index_k_proj": 8,
            ],
            family: .qwen38flash, attentionBits: 4)
    }

    @Test func aHyperGateThatDiffersBetweenSublayersIsRefused() {
        #expect(throws: ModelError.self) {
            try Model.validateRoleUniformity(
                overrides: [
                    "\(layer).attn_hyper_connection.block_inject_weight": 8,
                    "\(layer).mlp_hyper_connection.block_inject_weight": 4,
                ],
                family: .qwen38flash, attentionBits: 4)
        }
    }

    @Test func anIndexerWhoseKeysDisagreeIsRefused() {
        #expect(throws: ModelError.self) {
            try Model.validateRoleUniformity(
                overrides: [
                    "\(layer).self_attn.indexer.index_q_proj": 8,
                    "\(layer).self_attn.indexer.index_k_proj": 4,
                ],
                family: .qwen38flash, attentionBits: 4)
        }
    }
}
