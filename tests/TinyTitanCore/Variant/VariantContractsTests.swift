import Testing
import Foundation
@testable import TinyTitanCore

/// The 7.0 EngineVariant contracts (plan task 4, tracker row TT-075). These pin
/// the validation logic the variants rely on: variant-id parsing, the 6-bit
/// refusal, quant-spec resolution order, and geometry-contract validation.
struct VariantContractsTests {

    // MARK: VariantID

    @Test func validVariantIDParsesAndExposesBits() throws {
        let id = try VariantID("qwen35-2b-4bit")
        #expect(id.raw == "qwen35-2b-4bit")
        #expect(id.bits == 4)
        #expect(try VariantID("ornith-1.5-122b-8bit").bits == 8)
        #expect(try VariantID("some-model-2bit").bits == 2)
    }

    @Test func sixBitVariantIDIsRefusedByName() {
        #expect(throws: (any Error).self) { try VariantID("qwen35-2b-6bit") }
        do {
            _ = try VariantID("qwen35-2b-6bit")
            Issue.record("6-bit variant id was accepted")
        } catch {
            #expect("\(error)".contains("6"))
        }
    }

    @Test func malformedVariantIDsAreRefused() {
        #expect(throws: (any Error).self) { try VariantID("Qwen35-2B-4bit") }  // uppercase
        #expect(throws: (any Error).self) { try VariantID("qwen35-2b") }  // no width suffix
        #expect(throws: (any Error).self) { try VariantID("qwen35-2b-4") }  // no "bit"
        #expect(throws: (any Error).self) { try VariantID("") }
    }

    @Test func variantIDsAreComparableAndHashable() throws {
        let a = try VariantID("a-model-4bit")
        let b = try VariantID("b-model-4bit")
        #expect(a < b)
        #expect(Set([a, b, a]).count == 2)
    }

    // MARK: QuantSpec

    @Test func quantSpecAcceptsTwoFourEight() throws {
        for bits in [2, 4, 8] {
            let spec = try QuantSpec(bits: bits, group: 64)
            #expect(spec.bits == bits)
            #expect(spec.group == 64)
            #expect(spec.scaleIsBF16)
            #expect(spec.biasIsBF16)
        }
    }

    @Test func quantSpecRefusesSixByName() {
        do {
            _ = try QuantSpec(bits: 6, group: 64)
            Issue.record("6-bit quant spec was accepted")
        } catch {
            #expect("\(error)".contains("6"))
        }
    }

    @Test func quantSpecRefusesOtherWidths() {
        #expect(throws: (any Error).self) { try QuantSpec(bits: 3, group: 64) }
        #expect(throws: (any Error).self) { try QuantSpec(bits: 16, group: 64) }
    }

    // MARK: VariantQuant resolution

    @Test func quantResolutionOrderIsPerTensorThenRoleThenDefault() throws {
        let def = try QuantSpec(bits: 4, group: 64)
        let roleSpec = try QuantSpec(bits: 8, group: 64)
        let tensorSpec = try QuantSpec(bits: 2, group: 32)
        let stem = TensorStem("model.embed_tokens.weight")
        let quant = VariantQuant(
            defaultSpec: def,
            roles: [.embedding: roleSpec],
            perTensor: [stem: tensorSpec])

        // perTensor wins over the role and the default.
        #expect(quant.spec(forRole: .embedding, stem: stem).bits == 2)
        // role wins over the default for a different stem.
        #expect(quant.spec(forRole: .embedding, stem: TensorStem("other")).bits == 8)
        // default for an unlisted role.
        #expect(quant.spec(forRole: .router, stem: TensorStem("other")).bits == 4)
    }

    // MARK: GeometryContract

    @Test func geometryAcceptsAConformingManifest() throws {
        let contract = GeometryContract(required: [
            "hidden_size": .int(2048),
            "model_type": .string("qwen3_5_dense"),
        ])
        try contract.validate(["hidden_size": 2048, "model_type": "qwen3_5_dense", "extra": 1])
    }

    @Test func geometryRefusesAMissingKeyAndNamesIt() {
        let contract = GeometryContract(required: ["hidden_size": .int(2048)])
        do {
            try contract.validate(["model_type": "x"])
            Issue.record("missing key was accepted")
        } catch {
            #expect("\(error)".contains("hidden_size"))
        }
    }

    @Test func geometryRefusesAWrongIntAndNamesTheKey() {
        let contract = GeometryContract(required: ["hidden_size": .int(2048)])
        do {
            try contract.validate(["hidden_size": 4096])
            Issue.record("wrong int was accepted")
        } catch {
            #expect("\(error)".contains("hidden_size"))
        }
    }

    @Test func geometryRefusesAWrongType() {
        let contract = GeometryContract(required: ["hidden_size": .int(2048)])
        #expect(throws: (any Error).self) {
            try contract.validate(["hidden_size": "2048"])
        }
    }

    // MARK: Profiles and records construct

    @Test func profilesAndQualificationConstruct() throws {
        let perf = PerformanceProfile(
            expertCacheBytes: 1 << 30, prefetchDepth: 4, prefillChunk: 64,
            routerTopKSimd: true, attentionSimdPartial: false, keepCacheWired: true)
        #expect(perf.prefillChunk == 64)
        let sampling = SamplingProfile(
            temperature: 0.6, topK: 20, topP: 0.95, presencePenalty: 0,
            thinking: ThinkingDefaults(enabledByDefault: false))
        #expect(sampling.topK == 20)
        let q = Qualification(
            goldenSHA256: String(repeating: "a", count: 64),
            decodeTokS: 44.4, prefillTokS: 100, ttftS: 0.5,
            memoryCeilingBytes: 1 << 30, terminalBench: nil,
            hardware: "M3 MacBook Pro 24 GB", date: "2026-11-14")
        #expect(q.decodeTokS == 44.4)
        let recipe = ModelRecipe(
            variantID: try VariantID("qwen35-2b-4bit"),
            renames: [:], keepBF16Stems: [], headBits: nil, embeddingBits: nil)
        #expect(recipe.variantID.bits == 4)
    }
}
