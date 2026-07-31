import Testing
import Metal
@testable import TurboFieldfare

/// Guards the raw-value lookups in `MetalSDKCompatibility.swift`.
///
/// Those shims decide whether shaders compile at MSL 4.0. If they silently
/// answer "no" on a Metal 4 machine, the TensorOps prefill kernels drop out of
/// the library and prefill falls back to the causal-tiled path — everything
/// still builds, still runs, and still produces correct output, just slower.
/// These tests exist so that failure mode is loud instead.
@Suite struct MetalSDKCompatibilityTests {

    /// `MTLLanguageVersion` encodes its raw value as `(major << 16) + minor`.
    /// Pin that against cases every supported SDK declares, so a wrong
    /// assumption is caught even on a toolchain that has never heard of
    /// MSL 4.0 and therefore cannot exercise `msl4_0` directly.
    @Test func languageVersionRawValueUsesMajorMinorEncoding() {
        #expect(MTLLanguageVersion(rawValue: 3 << 16) == .version3_0)
        #expect(MTLLanguageVersion(rawValue: (3 << 16) + 2) == .version3_2)
    }

    /// The shims depend on `init?(rawValue:)` rejecting values the SDK does not
    /// declare. If it ever started returning a constructed value instead, they
    /// would claim Metal 4 support on every SDK, and `shaderLanguageVersion`
    /// would hand `MTLCompileOptions` a version the compiler cannot honor.
    @Test func unknownRawValuesAreRejected() {
        #expect(MTLLanguageVersion(rawValue: 99 << 16) == nil)
        #expect(MTLGPUFamily(rawValue: 9_999) == nil)
    }

    /// `MTLLanguageVersion.version4_0` and `MTLGPUFamily.apple10` both ship in
    /// the macOS 26 SDK, so the two lookups must agree: either this build can
    /// see Metal 4 or it cannot. A split means one raw value is wrong, and the
    /// shims would misreport what the toolchain supports.
    @Test func metal4ShimsResolveTogether() {
        let hasMSL4 = MTLLanguageVersion.msl4_0 != nil
        let hasApple10 = MTLGPUFamily.apple10IfAvailable != nil
        #expect(hasMSL4 == hasApple10, """
            Metal 4 SDK shims disagree: MTLLanguageVersion.msl4_0 \
            \(hasMSL4 ? "resolved" : "was nil") but MTLGPUFamily.apple10IfAvailable \
            \(hasApple10 ? "resolved" : "was nil"). Both are declared in the macOS 26 \
            SDK, so either both resolve or neither does. Check the raw values in \
            MetalSDKCompatibility.swift.
            """)
    }
}
