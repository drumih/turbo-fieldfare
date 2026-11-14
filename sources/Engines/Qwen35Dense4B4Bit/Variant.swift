// Provenance: scaffold created 2026-11-14 against docs/plan-tinytitan-7-engine.md §3
// (7.0 dense family). Statics grounded in tools/prepare_qwen35.py SIZES (hidden
// 2560, 32 layers, tied output) and the dense family sampling row. The CPU+GPU
// graph is ported in the dense-port task; until then this target refuses to serve.
import Foundation
import TinyTitanCore

/// The Qwen 3.5 4B model at 4-bit, as a 7.0 engine variant (CPU + GPU paths).
public struct Qwen35Dense4B4BitVariant: EngineVariant {
    public static let id: VariantID = {
        do { return try VariantID("qwen35-4b-4bit") } catch {
            fatalError("invalid variant id literal: \(error)")
        }
    }()

    public static let source = SourcePin(
        repo: "Qwen/Qwen3.5-4B", sha: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")

    public static let geometry = GeometryContract(required: [
        "family": .string("qwen3_5_dense"),
        "hiddenSize": .int(2560),
    ])

    public static let quant: VariantQuant = {
        do { return VariantQuant(defaultSpec: try QuantSpec(bits: 4, group: 64)) } catch {
            fatalError("invalid quant spec literal: \(error)")
        }
    }()

    public static let performance = PerformanceProfile(
        expertCacheBytes: 0, prefetchDepth: 0, prefillChunk: 128,
        routerTopKSimd: false, attentionSimdPartial: false, keepCacheWired: false)

    public static let sampling = SamplingProfile(
        temperature: 0.6, topK: 20, topP: 0.95, presencePenalty: 0, thinking: nil)

    public static let recipe = ModelRecipe(
        variantID: id, renames: [:], keepBF16Stems: [],
        headBits: nil, embeddingBits: nil)

    public init(context: EngineContext, model: LoadedModelHandle) throws {
        throw VariantError.notPorted(Self.id.raw)
    }
    public func prefill(_ tokens: [Int], state: inout SessionState) throws {
        throw VariantError.notPorted(Self.id.raw)
    }
    public func decode(_ state: inout SessionState) throws -> Int {
        throw VariantError.notPorted(Self.id.raw)
    }
    public static func qualify() throws -> Qualification {
        throw VariantError.notPorted(id.raw)
    }
}
