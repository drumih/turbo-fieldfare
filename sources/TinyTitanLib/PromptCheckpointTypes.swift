import CryptoKit
import Foundation
import TinyTitanCore

/// What boundary an interior checkpoint sits on.
enum PromptCheckpointKind: String, Codable, Sendable, Equatable, CaseIterable {
    /// End of the system message and tool schema.
    case systemEnd = "system-end"
    /// The same boundary as `systemEnd`, captured so it can be reused by a
    /// *different* conversation ("warm subagents"). The kind marks a record
    /// whose only valid cross-conversation use is this system-only prefix;
    /// same-conversation reuse still requires a matching conversation key.
    case sharedSystem = "shared-system"
    /// First token of the latest user turn.
    case lastTurnStart = "last-turn-start"
    /// End of a completed assistant turn (the existing prompt-cache point).
    case turnEnd = "turn-end"
    /// A fixed token-grid point. Off by default; the T24 benchmark enables it
    /// through `TINYTITAN_CHECKPOINT_GRID` to price coarse placement against
    /// the message boundaries.
    case grid = "grid"
}

/// Content address of a checkpoint: the hash of the domain plus the exact
/// prefix token IDs. Two checkpoints with different prefixes can never share
/// a key, and the key is the lookup for a restore.
struct PromptCheckpointKey: Codable, Hashable, Sendable, Equatable {
    let contentHash: String

    static func make(
        domain: ServerPromptCacheDomain,
        prefixTokenIDs: [Int32]
    ) -> PromptCheckpointKey {
        var data = Data()
        data.append(0x01)  // key version
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        // The domain is a small Codable value; a failed encode is a programmer
        // error in a type that only holds strings and ints.
        if let domainData = try? encoder.encode(domain) {
            data.append(domainData)
        }
        data.append(0x00)
        for token in prefixTokenIDs {
            var littleEndian = UInt32(bitPattern: token).littleEndian
            withUnsafeBytes(of: &littleEndian) { raw in
                data.append(contentsOf: raw)
            }
        }
        return PromptCheckpointKey(
            contentHash: Sha256Verifier.hashData(data))
    }

    /// Short form for logs; never used as the lookup.
    var short: String { String(contentHash.prefix(12)) }
}

/// Everything needed to find and validate a checkpoint without its payload.
struct PromptCheckpointMetadata: Codable, Equatable, Sendable {
    static let currentVersion = 1

    let version: Int
    let key: PromptCheckpointKey
    let domain: ServerPromptCacheDomain
    let kind: PromptCheckpointKind
    /// Groups the points of one conversation for eviction.
    let conversationKey: String
    let prefixTokenIDs: [Int32]
    let descriptor: InferenceStateSnapshotDescriptor
    let createdAt: Date

    init(
        key: PromptCheckpointKey,
        domain: ServerPromptCacheDomain,
        kind: PromptCheckpointKind,
        conversationKey: String,
        prefixTokenIDs: [Int32],
        descriptor: InferenceStateSnapshotDescriptor,
        createdAt: Date = Date(),
        version: Int = PromptCheckpointMetadata.currentVersion
    ) {
        self.version = version
        self.key = key
        self.domain = domain
        self.kind = kind
        self.conversationKey = conversationKey
        self.prefixTokenIDs = prefixTokenIDs
        self.descriptor = descriptor
        self.createdAt = createdAt
    }

    var position: Int { descriptor.position }

    /// Rejects a record whose state and prefix disagree.
    func validate() throws {
        guard version == Self.currentVersion else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "unsupported checkpoint version \(version)")
        }
        guard !conversationKey.isEmpty else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "conversationKey is empty")
        }
        guard descriptor.position > 0 else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "position must be positive")
        }
        guard descriptor.position == prefixTokenIDs.count else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "position \(descriptor.position) != prefix token count "
                    + "\(prefixTokenIDs.count)")
        }
        let recomputed = PromptCheckpointKey.make(
            domain: domain, prefixTokenIDs: prefixTokenIDs)
        guard recomputed == key else {
            throw PromptCheckpointStoreError.invalidMetadata(
                "content hash does not match the prefix")
        }
    }
}

enum PromptCheckpointStoreError: Error, CustomStringConvertible, Equatable {
    case invalidMetadata(String)
    case keyMismatch
    case domainMismatch
    case missing(PromptCheckpointKey)
    case corrupt(PromptCheckpointKey, String)
    case exceedsLimit(bytes: Int, limit: Int)
    case diskUnavailable

    var description: String {
        switch self {
        case .invalidMetadata(let detail):
            "invalid prompt checkpoint: \(detail)"
        case .keyMismatch:
            "prompt checkpoint key does not match its metadata"
        case .domainMismatch:
            "prompt checkpoint belongs to a different runtime domain"
        case .missing(let key):
            "prompt checkpoint \(key.short) is unavailable"
        case .corrupt(let key, let reason):
            "prompt checkpoint \(key.short) is corrupt: \(reason)"
        case .exceedsLimit(let bytes, let limit):
            "prompt checkpoint is \(bytes) bytes, over the \(limit)-byte limit"
        case .diskUnavailable:
            "prompt checkpoint disk store is not configured"
        }
    }
}
