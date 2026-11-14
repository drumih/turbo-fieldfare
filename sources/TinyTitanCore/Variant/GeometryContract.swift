import Foundation

/// A variant's manifest contract (7.0 spec §12.2): the geometry fields it
/// requires, validated against the install's `manifest.json`. A preset is a
/// validation contract, not a runtime geometry — the 6.x "family preset unless
/// the manifest differs" branch disappears; every variant validates its own
/// manifest and a near-miss is refused.
public struct GeometryContract: Sendable {
    /// A required geometry value: an exact integer or an exact string.
    public enum GeometryValue: Sendable, Equatable {
        case int(Int)
        case string(String)
    }

    public let required: [String: GeometryValue]

    public init(required: [String: GeometryValue]) {
        self.required = required
    }

    /// Validates a manifest dictionary against the contract. Throws naming the
    /// first missing or mismatched key. Extra keys are allowed (the manifest
    /// carries more than the contract requires).
    public func validate(_ manifest: [String: Any]) throws {
        // Sorted so the refusal is deterministic regardless of dictionary order.
        for (key, expected) in required.sorted(by: { $0.key < $1.key }) {
            guard let present = manifest[key] else {
                throw GeometryContractError("manifest is missing required key '\(key)'")
            }
            switch expected {
            case .int(let want):
                guard let got = Self.intValue(present), got == want else {
                    throw GeometryContractError(
                        "manifest key '\(key)' must be \(want), got \(present)")
                }
            case .string(let want):
                guard let got = present as? String, got == want else {
                    throw GeometryContractError(
                        "manifest key '\(key)' must be \"\(want)\", got \(present)")
                }
            }
        }
    }

    /// JSON manifests arrive as NSNumber; a native Int also bridges. Accept both.
    private static func intValue(_ value: Any) -> Int? {
        if let int = value as? Int { return int }
        if let number = value as? NSNumber { return number.intValue }
        return nil
    }
}

/// A refused manifest, naming the key that failed.
public struct GeometryContractError: Error, CustomStringConvertible {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}
