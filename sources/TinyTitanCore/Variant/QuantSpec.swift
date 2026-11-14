import Foundation

/// One affine quantization spec (7.0 spec §6). Affine only in 7.0; the width is
/// 2, 4 or 8 — 6 is refused by name. The group size is a function constant of
/// the variant's kernels: a group-32 variant compiles group-32 kernels and does
/// not reinterpret a group-64 buffer.
public struct QuantSpec: Sendable, Equatable {
    public let bits: Int
    public let group: Int
    public let scaleIsBF16: Bool
    public let biasIsBF16: Bool

    public init(bits: Int, group: Int, scaleIsBF16: Bool = true, biasIsBF16: Bool = true) throws {
        guard [2, 4, 8].contains(bits) else {
            throw QuantSpecError(
                "quant width must be 2, 4 or 8 (6-bit is withdrawn): \(bits)")
        }
        guard group > 0 else {
            throw QuantSpecError("quant group must be positive: \(group)")
        }
        self.bits = bits
        self.group = group
        self.scaleIsBF16 = scaleIsBF16
        self.biasIsBF16 = biasIsBF16
    }
}

/// A refused quant spec, carrying the reason.
public struct QuantSpecError: Error, CustomStringConvertible {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}

/// The tensor roles a variant may declare a width for (7.0 spec §6). A variant
/// declares only the roles it actually has; there is no global role-uniformity
/// rule. The MoE-shaped five-slot contract of 6.x is retired.
public enum QuantRole: String, Sendable, CaseIterable {
    case embedding, lmHead, attentionQO, attentionKV, gdnQKV, gdnZ, gdnOut,
        router, sharedExpert, routedExpert, hyperConnection, ple, indexer,
        movaValue
}

/// A tensor stem: the name a variant reads a tensor by, minus any layer index.
public struct TensorStem: Hashable, Sendable, CustomStringConvertible {
    public let raw: String
    public init(_ raw: String) { self.raw = raw }
    public var description: String { raw }
}

/// A variant's full quantization contract: a default spec, per-role overrides,
/// and per-tensor overrides (7.0 spec §6). Resolution is perTensor > roles >
/// default. The variant validates every tensor it reads against this; a width
/// cannot be bypassed because the concrete kernels are internal to the variant.
public struct VariantQuant: Sendable {
    public let defaultSpec: QuantSpec
    public let roles: [QuantRole: QuantSpec]
    public let perTensor: [TensorStem: QuantSpec]

    public init(
        defaultSpec: QuantSpec,
        roles: [QuantRole: QuantSpec] = [:],
        perTensor: [TensorStem: QuantSpec] = [:]
    ) {
        self.defaultSpec = defaultSpec
        self.roles = roles
        self.perTensor = perTensor
    }

    /// The spec a given role/stem reads at: per-tensor override first, then the
    /// role override, then the default.
    public func spec(forRole role: QuantRole, stem: TensorStem) -> QuantSpec {
        if let override = perTensor[stem] { return override }
        if let roleSpec = roles[role] { return roleSpec }
        return defaultSpec
    }
}
