import Foundation
import TurboFieldfare

/// Whether the service may attempt a load at the requested context.
///
/// Pure and separate from the command loop so the decision is testable without
/// a socket, a model or a machine of the size in question. The service is the
/// process that would be killed: a 262,144-token KV is charged in full the
/// moment a command buffer binds it, so an unbacked context is not a load that
/// runs slowly — it is a jetsam kill with no event on the wire, which the app
/// can only report as a lost connection.
enum DecodeLoadAdmission {
    /// Diagnostics only. Set it to run a context this rule refuses and find out
    /// what actually happens on the machine, which is the measurement the rule
    /// itself came from.
    static let overrideEnvironmentKey = "TURBO_FIELDFARE_ALLOW_UNBACKED_CONTEXT"

    /// The message to answer the load with, or nil to proceed.
    static func refusal(maxContextTokens: Int,
                        hostMemoryBytes: UInt64,
                        environment: [String: String], expertCacheSlots: Int = 16) -> String? {
        let config = ArchConfig.gemma4_26B_A4B
        // The ceiling is the checkpoint's, not the host's, so no override
        // reaches it: memory cannot make a position the model was never
        // trained to address valid.
        guard maxContextTokens <= config.maxPositionEmbeddings else {
            return ModelError.contextExceedsModel(
                requested: maxContextTokens,
                maximum: config.maxPositionEmbeddings).description
        }
        guard environment[overrideEnvironmentKey] != "1" else { return nil }
        guard case .needsMemory = ContextAdmission.availability(
            config: config,
            maxContext: maxContextTokens,
            hostMemoryBytes: hostMemoryBytes, expertCacheSlots: expertCacheSlots) else { return nil }
        return ContextAdmission.needDescription(config: config,
                                                maxContext: maxContextTokens,
                                                hostMemoryBytes: hostMemoryBytes, expertCacheSlots: expertCacheSlots)
    }
}
