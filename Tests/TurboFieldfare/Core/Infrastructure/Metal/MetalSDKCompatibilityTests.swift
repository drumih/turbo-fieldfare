import Testing
import Metal
@testable import TurboFieldfare

/// Guards the raw-value lookups in `MetalSDKCompatibility.swift`.
///
/// Those values decide whether shaders compile at MSL 4.0. Get the encoding
/// wrong in one direction and a Metal 4 machine silently drops its tensor-ops
/// kernels — everything still builds and produces correct output, just slower.
/// Get it wrong in the other and the entire shader library fails to compile.
@Suite struct MetalSDKCompatibilityTests {

    /// `MTLLanguageVersion` encodes its raw value as `(major << 16) + minor`.
    /// Pin that against cases the macOS 15 SDK declares, so a wrong assumption
    /// is caught on the floor toolchain, which cannot name MSL 4.0 at all.
    @Test func languageVersionRawValueUsesMajorMinorEncoding() {
        #expect(MTLLanguageVersion(rawValue: 3 << 16) == .version3_0)
        #expect(MTLLanguageVersion(rawValue: (3 << 16) + 2) == .version3_2)
    }

    /// The value handed to `MTLCompileOptions` on macOS 26 must be exactly
    /// MSL 4.0. It is built by raw value, so nothing else type-checks it.
    @Test func msl4ShimCarriesTheMSL4RawValue() {
        #expect(MTLLanguageVersion.msl4_0?.rawValue == 4 << 16)
    }

    /// Metal's enums import as non-frozen, so `init?(rawValue:)` constructs
    /// undeclared values instead of rejecting them. That means nil-ness cannot
    /// be used to detect whether the SDK knows a case, which is why
    /// `MetalContext.shaderLanguageVersion` gates on `#available` instead.
    /// If this ever starts failing, that guard could be simplified — until
    /// then, removing it would hand MSL 4.0 to a macOS 15 Metal compiler.
    @Test func unknownRawValuesAreConstructedNotRejected() {
        #expect(MTLLanguageVersion(rawValue: 99 << 16) != nil)
        #expect(MTLGPUFamily(rawValue: 9_999) != nil)
    }
}
