import Foundation
import Metal

/// The context a variant is constructed with (7.0 spec §3, generalized from the
/// spec's MetalContext-only init). CPU-only variants receive `metal == nil`;
/// GPU variants receive the device. The full Metal context lives in the runtime
/// today; a variant that needs more than the device is given it through its own
/// target, not through this generic handle.
public struct EngineContext: Sendable {
    public let metal: MTLDevice?
    public let physicalRAMBytes: UInt64

    public init(metal: MTLDevice?, physicalRAMBytes: UInt64) {
        self.metal = metal
        self.physicalRAMBytes = physicalRAMBytes
    }
}

/// A tensor/residency handle (7.0 spec §8): names, offsets, byte counts and
/// dtypes — not PLE, hyper-connections or QSA. Those features live with the
/// variants that own them. This is the minimal generic shape; the first variant
/// widens it by pure code motion if it needs more.
public struct LoadedModelHandle: Sendable {
    public struct Tensor: Sendable {
        public let name: String
        public let byteOffset: Int
        public let byteCount: Int
        public let dtype: String

        public init(name: String, byteOffset: Int, byteCount: Int, dtype: String) {
            self.name = name
            self.byteOffset = byteOffset
            self.byteCount = byteCount
            self.dtype = dtype
        }
    }

    public let tensors: [String: Tensor]

    public init(tensors: [String: Tensor]) {
        self.tensors = tensors
    }
}

/// The mutable decode state a variant advances. Minimal generic shape; the
/// first variant widens it (KV cache, auxiliary state) by pure code motion.
public struct SessionState: Sendable {
    public var position: Int
    public var tokens: [Int]

    public init(position: Int = 0, tokens: [Int] = []) {
        self.position = position
        self.tokens = tokens
    }
}

/// The unit of code, test, tuning and release (7.0 spec §3). One variant owns
/// its graph, kernels, quant spec, geometry, profiles, recipe and qualification.
/// It has no methods that take another model's name, no optional feature flags,
/// and no runtime family switch. The registry resolves a variant id to a
/// concrete factory once, at load; an unknown id is refused rather than defaulted.
public protocol EngineVariant: Sendable {
    static var id: VariantID { get }
    static var source: SourcePin { get }
    static var geometry: GeometryContract { get }
    static var quant: VariantQuant { get }
    static var performance: PerformanceProfile { get }
    static var sampling: SamplingProfile { get }
    static var recipe: ModelRecipe { get }

    init(context: EngineContext, model: LoadedModelHandle) throws
    func prefill(_ tokens: [Int], state: inout SessionState) throws
    func decode(_ state: inout SessionState) throws -> Int
    static func qualify() throws -> Qualification
}

/// The pinned source a variant is built from: repo + commit sha (7.0 spec §3).
public struct SourcePin: Sendable, Equatable {
    public let repo: String
    public let sha: String

    public init(repo: String, sha: String) {
        self.repo = repo
        self.sha = sha
    }
}
