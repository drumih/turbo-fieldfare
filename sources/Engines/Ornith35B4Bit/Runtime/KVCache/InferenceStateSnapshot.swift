import Foundation

/// Serializable description of every persistent inference-state buffer needed
/// to resume a prompt without replaying its cached tokens.
public struct InferenceStateSnapshotDescriptor: Codable, Equatable, Sendable {
    public static let currentVersion = 1

    public let version: Int
    public let position: Int
    public let kvSegmentLengths: [Int]
    public let gdnSegmentLengths: [Int]
    public let payloadBytes: Int

    public init(
        version: Int = currentVersion,
        position: Int,
        kvSegmentLengths: [Int],
        gdnSegmentLengths: [Int],
        payloadBytes: Int
    ) {
        self.version = version
        self.position = position
        self.kvSegmentLengths = kvSegmentLengths
        self.gdnSegmentLengths = gdnSegmentLengths
        self.payloadBytes = payloadBytes
    }
}

/// In-memory inference-state checkpoint. The payload is a concatenation of the
/// K/V segments followed by Qwen gated-DeltaNet recurrent-state segments.
public struct InferenceStateSnapshot: Equatable, Sendable {
    public let descriptor: InferenceStateSnapshotDescriptor
    public let payload: Data

    public init(descriptor: InferenceStateSnapshotDescriptor, payload: Data) {
        self.descriptor = descriptor
        self.payload = payload
    }
}

public enum InferenceStateSnapshotError: Error, Equatable, CustomStringConvertible {
    case unsupportedVersion(Int)
    case invalidPosition(Int)
    case invalidLayout
    case invalidPayloadSize(expected: Int, actual: Int)
    case exceedsLimit(bytes: Int, limit: Int)
    case integerOverflow
    /// A live subsystem's state is not carried by the snapshot, so a restore
    /// would leave the previous conversation's buffers mixed into this prefix.
    case stateNotInSnapshot(String)

    public var description: String {
        switch self {
        case .unsupportedVersion(let version):
            "unsupported inference-state snapshot version: \(version)"
        case .invalidPosition(let position):
            "invalid inference-state snapshot position: \(position)"
        case .invalidLayout:
            "inference-state snapshot layout does not match the loaded runtime"
        case .invalidPayloadSize(let expected, let actual):
            "inference-state snapshot payload is \(actual) bytes; expected \(expected)"
        case .exceedsLimit(let bytes, let limit):
            "inference-state snapshot requires \(bytes) bytes; cache limit is \(limit)"
        case .integerOverflow:
            "inference-state snapshot size overflow"
        case .stateNotInSnapshot(let feature):
            "inference-state snapshot does not carry \(feature) state, and restoring "
                + "would leave the previous conversation's buffers in place"
        }
    }
}

extension InferenceStateSnapshotDescriptor {
    public func validatedPayloadBytes() throws -> Int {
        var total = 0
        for length in kvSegmentLengths + gdnSegmentLengths {
            guard length >= 0 else { throw InferenceStateSnapshotError.invalidLayout }
            let (next, overflow) = total.addingReportingOverflow(length)
            guard !overflow else { throw InferenceStateSnapshotError.integerOverflow }
            total = next
        }
        guard total == payloadBytes else {
            throw InferenceStateSnapshotError.invalidPayloadSize(
                expected: total,
                actual: payloadBytes)
        }
        return total
    }
}
