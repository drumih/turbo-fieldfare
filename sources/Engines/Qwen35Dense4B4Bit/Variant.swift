// Provenance: 7.0 dense variant (plan task 5/7). The CPU forward is copied from
// sources/TinyTitan/CPUEngine; the model-specific reasoning/sampling subsystem
// is copied with it (a variant owns its model knowledge, spec §1.2). The loader
// reads through the Core SSDAIManifestReader. Statics grounded in
// tools/prepare_qwen35.py SIZES["4b"] (hidden 2560, 32 layers, tied output).
import Foundation
import TinyTitanCore

/// The Qwen 3.5 4B at 4-bit, as a 7.0 engine variant serving the dense CPU path.
/// unchecked-invariant: the CPUQwen35 instance and its last-logits buffer are
/// confined to a single generation session and accessed serially by that
/// session's prefill/decode loop; no two sessions share one variant instance.
public final class Qwen35Dense4B4BitVariant: EngineVariant, @unchecked Sendable {
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

    private let engine: CPUQwen35
    private var lastLogits: [Float] = []

    public init(context: EngineContext, model: LoadedModelHandle) throws {
        guard let directory = model.directoryURL else {
            throw VariantError.notPorted(Self.id.raw)
        }
        let snapshot = try AffineSnapshot(ssdai: directory)
        engine = try CPUQwen35(snapshot: snapshot)
    }

    public func prefill(_ tokens: [Int], state: inout SessionState) throws {
        lastLogits = try engine.prefill(tokens: tokens)
        state.position = tokens.count
        state.tokens = tokens
    }

    public func decode(_ state: inout SessionState) throws -> Int {
        guard !lastLogits.isEmpty else {
            throw VariantError.notPorted(Self.id.raw)
        }
        var best = 0
        for index in lastLogits.indices where lastLogits[index] > lastLogits[best] {
            best = index
        }
        lastLogits = try engine.step(token: best)
        state.position += 1
        state.tokens.append(best)
        return best
    }

    public static func qualify() throws -> Qualification {
        throw VariantError.notPorted(id.raw)
    }
}
