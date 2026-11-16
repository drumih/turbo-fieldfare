// Provenance: template created 2026-11-14 against docs/plan-tinytitan-7-engine.md §3
// (plan task 5, tracker row TT-076). The graph ported in task 7 is copied from
// sources/TinyTitan/CPUEngine (commit recorded at copy time). Statics are grounded
// in tools/prepare_qwen35.py SIZES["2b"] (hidden 2048, 24 layers, tied output) and
// the dense family sampling row (GenerationDefaults.forFamily(.qwen35Dense)); the
// dense trio has no ModelProfile row (tracker TT-072), so family defaults apply.
import Foundation
import TinyTitanCore

/// The Qwen 3.5 2B at 4-bit, as a 7.0 engine variant serving the dense CPU path.
/// unchecked-invariant: the CPUQwen35 instance and its last-logits buffer are
/// confined to a single generation session and accessed serially by that
/// session's prefill/decode loop; no two sessions share one variant instance.
public final class Qwen35Dense2B4BitVariant: EngineVariant, @unchecked Sendable {
    public static let id: VariantID = {
        do { return try VariantID("qwen35-2b-4bit") } catch {
            fatalError("invalid variant id literal: \(error)")
        }
    }()

    public static let source = SourcePin(
        repo: "Qwen/Qwen3.5-2B",
        sha: "15852e8c16360a2fea060d615a32b45270f8a8fc")

    /// The manifest contract over the install's `arch` sub-dictionary. A preset
    /// is a validation contract, not a runtime geometry (spec §12.2).
    public static let geometry = GeometryContract(required: [
        "family": .string("qwen3_5_dense"),
        "hiddenSize": .int(2048),
    ])

    public static let quant: VariantQuant = {
        do {
            let spec = try QuantSpec(bits: 4, group: 64)
            return VariantQuant(defaultSpec: spec)
        } catch { fatalError("invalid quant spec literal: \(error)") }
    }()

    /// Dense: no routed experts, so no expert cache or prefetch. The prefill
    /// chunk is the runtime default; task 7 sets the measured value.
    public static let performance = PerformanceProfile(
        expertCacheBytes: 0, prefetchDepth: 0, prefillChunk: 128,
        routerTopKSimd: false, attentionSimdPartial: false, keepCacheWired: false)

    /// The dense family's single sampling row (0.6 / top-k 20 / top-p 0.95);
    /// Qwen 3.5 publishes no separate thinking row.
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
