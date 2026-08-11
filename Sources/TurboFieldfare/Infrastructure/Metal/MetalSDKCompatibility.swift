import Foundation
import Metal

/// Shims for Metal symbols that exist only in the macOS 26 / iOS 26 SDK.
///
/// The package deploys to macOS 15 so it can be built by the Swift 6.1
/// toolchain, whose SDK does not declare `MTLLanguageVersion.version4_0` or
/// `MTLGPUFamily.apple10`. Both are integer-backed `NS_ENUM`s, so we pass their
/// raw values through instead of naming the cases.
///
/// These import as non-frozen enums, so `init?(rawValue:)` does **not** reject
/// undeclared values — it constructs one, and `MetalSDKCompatibilityTests` pins
/// that. Nothing here can tell you whether the SDK knows a case, so callers must
/// gate on the runtime OS instead:
///
/// - `MTLGPUFamily` is safe to pass through. `supportsFamily` is a runtime
///   query, so a system with no Apple10 answers `false`.
/// - `MTLLanguageVersion` is **not** safe to pass through. Handing MSL 4.0 to
///   the macOS 15 Metal compiler fails the entire shader library, which takes
///   the whole runtime down — so `MetalContext.shaderLanguageVersion` guards it
///   with `#available` and must keep doing so.

extension MTLLanguageVersion {
    /// MSL 4.0 — the version that enables the MPP tensor-ops kernels.
    ///
    /// `MTLLanguageVersion` encodes its raw value as `(major << 16) + minor`.
    /// Never hand this to Metal outside a macOS 26 availability check.
    static var msl4_0: MTLLanguageVersion? { MTLLanguageVersion(rawValue: 4 << 16) }
}

extension MTLDevice {
    /// Whether this device supports the Apple10 MPP tensor operations that back
    /// the TensorOps prefill kernels.
    ///
    /// Safe to ask on any OS: an older system simply reports `false` for a
    /// family it has never heard of.
    var supportsApple10TensorOps: Bool {
        guard let family = MTLGPUFamily(rawValue: 1010) else { return false }
        return supportsFamily(family)
    }
}
