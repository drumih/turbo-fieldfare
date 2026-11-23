import Foundation
import Inkling2GGUFQ4
import Metal
import TinyTitan
import TinyTitanCore

/// Loads an Inkling 2 Q4 `.ssdai` install (TT-121 phase 2) for the Inkling
/// backend: the engine variant's decoder with its experts streamed into slots
/// sized by `--ram-budget` (or `--expert-cache-slots`), and the tokenizer from
/// the GGUF header the install keeps.
package enum Inkling2Q4ServerLoad {
    /// Whether `url` is an Inkling 2 Q4 install.
    package static func recognizes(_ url: URL) -> Bool {
        InklingQ4Install.recognizes(url)
    }

    /// The session, and the plan it was sized by.
    package static func session(
        directory: URL, requestedContext: Int, ramBudget: Int?, explicitSlots: Int?,
        maxChunk: Int = 256
    ) async throws -> (session: Inkling2GGUFSession, plan: InklingQ4LoadPlan) {
        let workingSet = Int(MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0)
        let plan = try InklingQ4LoadPlan.make(
            geometry: try InklingQ4LoadPlan.Geometry(directory: directory),
            requested: requestedContext, ramBudget: ramBudget, explicitSlots: explicitSlots,
            workingSet: workingSet)
        let decoder = try InklingQ4Decoder(
            directory: directory, maxContext: plan.context,
            maxChunk: min(maxChunk, plan.context), experts: .slots(plan.slots))
        let session = try await Inkling2GGUFSession(
            decoder: Adapter(decoder),
            ggufHeader: directory.appendingPathComponent(InklingQ4Install.ggufHeaderFile))
        FileHandle.standardError.write(
            Data(
                ("Inkling 2 Q4: \(plan.slots) expert slots per layer "
                    + "(\((decoder.expertCache?.residentBytes ?? 0) >> 20) MiB), context "
                    + "\(plan.context) tokens (asked \(requestedContext)), planned "
                    + "\(plan.plannedBytes >> 20) MiB\n").utf8))
        return (session, plan)
    }

    /// The variant's decoder as the session's decoder.
    final class Adapter: Inkling2Decoding {
        let decoder: InklingQ4Decoder

        init(_ decoder: InklingQ4Decoder) { self.decoder = decoder }

        var maxContext: Int { decoder.state.maxContext }

        func resetSequence() { decoder.state.reset() }

        func prefill(_ prompt: [Int], position: Int, returnLogits: Bool) throws
            -> (next: Int, logits: [Float]?)
        {
            try decoder.prefill(prompt, position: position, returnLogits: returnLogits)
        }

        func step(token: Int, position: Int, returnLogits: Bool) throws
            -> (next: Int, logits: [Float]?)
        {
            try decoder.step(token: token, position: position, returnLogits: returnLogits)
        }
    }
}
