import Foundation
import Metal

/// Shims for Metal symbols that exist only in the macOS 26 / iOS 26 SDK.
///
/// The package deploys to macOS 15 so it can be built by the Swift 6.1
/// toolchain, whose SDK does not declare `MTLLanguageVersion.version4_0` or
/// `MTLGPUFamily.apple10`. Both are integer-backed `NS_ENUM`s, so we look them
/// up by raw value instead of naming the case: the lookup returns `nil` on the
/// older SDK and the real case on the newer one. That keeps the MPP tensor-ops
/// fast path available to anyone building with Xcode 26 without breaking the
/// older toolchain.

extension MTLLanguageVersion {
    /// MSL 4.0 — the version that enables the MPP tensor-ops kernels.
    /// `nil` when compiled against an SDK that predates it.
    ///
    /// `MTLLanguageVersion` encodes its raw value as `(major << 16) + minor`.
    static var msl4_0: MTLLanguageVersion? { MTLLanguageVersion(rawValue: 4 << 16) }
}

extension MTLGPUFamily {
    /// `MTLGPUFamily.apple10` — the first family with MPP tensor support.
    /// `nil` when compiled against an SDK that predates it.
    static var apple10IfAvailable: MTLGPUFamily? { MTLGPUFamily(rawValue: 1010) }
}

extension MTLDevice {
    /// Whether this device supports the Apple10 MPP tensor operations that back
    /// the TensorOps prefill kernels. Always `false` when the package is built
    /// against an SDK that has no Apple10 family to ask about.
    var supportsApple10TensorOps: Bool {
        guard let family = MTLGPUFamily.apple10IfAvailable else { return false }
        return supportsFamily(family)
    }
}
