import Foundation

/// Per-sequence state an `InferenceStateSnapshot` does not carry.
///
/// The KV and Gated-DeltaNet payload is enough for the plain families. A
/// prefix that also runs the sparse indexer owns that indexer's state (raw
/// tail keys and pooled blocks, both per layer), one PLE convolution window,
/// and the token context the PLE n-gram hash reads (version 2; version 1
/// omitted it, so the first `ngramSize - 1` tokens after every restore hashed
/// the wrong n-grams — found by the wide-verify identity bisect,
/// docs/plan-wide-verify.md). Without those, a restore would attend to a
/// subset of keys chosen from the previous conversation's n-grams, which is
/// fluent and wrong. This payload travels beside the snapshot rather than
/// inside it so the snapshot format and every existing record stay valid.
public struct InferenceAuxiliaryState: Sendable, Equatable {
    public static let currentVersion = 2

    public let version: Int
    /// KV position the payload was captured at; must equal the snapshot's.
    public let position: Int
    public let payload: Data

    public init(
        version: Int = currentVersion,
        position: Int,
        payload: Data
    ) {
        self.version = version
        self.position = position
        self.payload = payload
    }
}

public enum InferenceAuxiliaryStateError: Error, Equatable, CustomStringConvertible {
    case unsupportedVersion(Int)
    case invalidPosition(expected: Int, actual: Int)
    case truncated(String)
    case malformed(String)
    case missingState(String)
    case unexpectedState(String)

    public var description: String {
        switch self {
        case .unsupportedVersion(let version):
            "unsupported auxiliary inference-state version: \(version)"
        case .invalidPosition(let expected, let actual):
            "auxiliary state position \(actual) does not match snapshot \(expected)"
        case .truncated(let detail):
            "auxiliary inference state is truncated: \(detail)"
        case .malformed(let detail):
            "auxiliary inference state is malformed: \(detail)"
        case .missingState(let feature):
            "auxiliary state does not carry \(feature)"
        case .unexpectedState(let feature):
            "auxiliary state carries \(feature), but the loaded runner does not"
        }
    }
}

/// Little-endian cursor helpers for the auxiliary payload. `package` because
/// the engine modules' own snapshot capture/restore code writes with them.
package enum AuxiliaryCoding {
    package static func appendUInt32(_ value: Int, to data: inout Data) {
        var littleEndian = UInt32(value).littleEndian
        withUnsafeBytes(of: &littleEndian) { raw in
            data.append(contentsOf: raw)
        }
    }

    package static func appendBytes(_ bytes: UnsafeRawPointer, count: Int, to data: inout Data) {
        data.append(Data(bytes: bytes, count: count))
    }

    package static func readUInt32(
        _ bytes: UnsafeRawBufferPointer,
        offset: inout Int
    ) throws -> Int {
        guard offset + 4 <= bytes.count, let base = bytes.baseAddress else {
            throw InferenceAuxiliaryStateError.truncated("uint32 at \(offset)")
        }
        var value: UInt32 = 0
        withUnsafeMutableBytes(of: &value) { destination in
            destination.copyMemory(
                from: UnsafeRawBufferPointer(
                    start: base.advanced(by: offset), count: 4))
        }
        offset += 4
        return Int(UInt32(littleEndian: value))
    }

    /// A token list: its count, then each token's bits.
    package static func appendTokens(_ tokens: [Int32], to data: inout Data) {
        appendUInt32(tokens.count, to: &data)
        for token in tokens {
            appendUInt32(Int(UInt32(bitPattern: token)), to: &data)
        }
    }

    /// Reads what `appendTokens` wrote, refusing more than `maximumCount`.
    package static func readTokens(
        _ bytes: UnsafeRawBufferPointer,
        offset: inout Int,
        maximumCount: Int
    ) throws -> [Int32] {
        let count = try readUInt32(bytes, offset: &offset)
        guard count <= maximumCount else {
            throw InferenceAuxiliaryStateError.malformed(
                "token context of \(count) exceeds \(maximumCount)")
        }
        var tokens: [Int32] = []
        tokens.reserveCapacity(count)
        for _ in 0..<count {
            tokens.append(Int32(bitPattern: UInt32(try readUInt32(bytes, offset: &offset))))
        }
        return tokens
    }

    package static func require(
        _ bytes: UnsafeRawBufferPointer,
        offset: Int,
        count: Int,
        _ what: String
    ) throws {
        guard count >= 0, offset >= 0, offset + count <= bytes.count else {
            throw InferenceAuxiliaryStateError.truncated(what)
        }
    }
}
