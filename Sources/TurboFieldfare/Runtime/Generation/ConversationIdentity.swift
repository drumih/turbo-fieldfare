import Foundation

public enum ConversationIdentityError: Error, CustomStringConvertible {
    case snapshotHashUnavailable

    public var description: String {
        switch self {
        case .snapshotHashUnavailable:
            "the loaded model's manifest carries no source snapshot hash, so a "
                + "conversation recorded against it could not be told apart from "
                + "one recorded against a different checkpoint"
        }
    }
}

/// Everything that has to be the same for a stored conversation's token IDs to
/// mean what they meant when they were recorded.
///
/// Not a cache key: a conversation owns its lineage and never has to match
/// anything. It is a refusal test. Token IDs index a vocabulary, sit at
/// positions a template chose, and stand in for images a preprocessor produced;
/// change any of those and the same integers describe a different conversation.
/// The four fields are the four ways that can happen — a different checkpoint,
/// a different model, a different chat framing, or different pixels out of the
/// same file — and a mismatch in any of them leaves the transcript readable and
/// says so rather than continuing on a lineage the model never had.
public struct ConversationIdentity: Sendable, Equatable, Codable {
    public let modelID: String
    public let sourceSnapshotHash: String
    public let templateIdentity: String
    public let imageProcessingVersion: Int

    public init(modelID: String,
                sourceSnapshotHash: String,
                templateIdentity: String,
                imageProcessingVersion: Int) {
        self.modelID = modelID
        self.sourceSnapshotHash = sourceSnapshotHash
        self.templateIdentity = templateIdentity
        self.imageProcessingVersion = imageProcessingVersion
    }

    /// The identity of a loaded model. The chat framing is the tokenizer's
    /// declared template identity, which is a type-level constant because the
    /// renderer is hand-written Swift with no template file to hash.
    ///
    /// Throws rather than substituting a placeholder for a missing snapshot
    /// hash: an identity that cannot be established has to refuse replay, and a
    /// blank field would compare equal to the next blank field and admit it.
    public init(model: Model) throws {
        guard let snapshot = model.sourceSnapshotHash, !snapshot.isEmpty else {
            throw ConversationIdentityError.snapshotHashUnavailable
        }
        self.init(modelID: model.modelID,
                  sourceSnapshotHash: snapshot,
                  templateIdentity: GFTokenizer.chatTemplateIdentity,
                  imageProcessingVersion: VisionImageProcessing.version)
    }

    /// The identity of an installed model, from its manifest alone.
    ///
    /// The app process needs this and never loads the model: the decode service
    /// does. Reading the manifest costs a few kilobytes and stays well inside
    /// the rule that no path holds model-derived data in the heap.
    public static func forModelDirectory(_ directoryURL: URL) throws -> ConversationIdentity {
        let manifest = try ManifestReader.load(
            directoryURL: directoryURL, expecting: .gemma4_26B_A4B)
        guard let snapshot = manifest.sourceSnapshotHash, !snapshot.isEmpty else {
            throw ConversationIdentityError.snapshotHashUnavailable
        }
        return ConversationIdentity(
            modelID: manifest.modelID,
            sourceSnapshotHash: snapshot,
            templateIdentity: GFTokenizer.chatTemplateIdentity,
            imageProcessingVersion: VisionImageProcessing.version)
    }
}
