import Foundation
import Testing
import TinyTitan

@testable import TinyTitanLib
@testable import TinyTitanCore

/// The one integrity decision the facade exposes.
///
/// `.automatic` has to stay the loader's own rule — the default must not change
/// what every existing caller does — and the two explicit policies have to be
/// what they say: one re-reads the payload, the other trusts the installer's
/// receipt (and is strict about it, refusing an install whose receipt is missing
/// or does not bind).
@Suite struct InstallIntegrityTests {
    @Test func automaticLeavesTheResolvingToTheLoader() {
        #expect(InstallIntegrity.automatic.engineValue == nil)
        #expect(EngineConfiguration().integrityPolicy == .automatic)
    }

    @Test func theExplicitPoliciesMapToTheRuntimeOnes() {
        #expect(InstallIntegrity.verifyEveryFile.engineValue == .fullSha256)
        #expect(InstallIntegrity.trustInstallerReceipt.engineValue == .sizeCheckTrustedReceipt)
    }
}
