import Testing
import TurboFieldfare
import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareDecodeService

@Suite struct DecodeRuntimeOptionsTests {
    @Test func omittedPrefetchReachesServiceAsOff() throws {
        let app = try TurboFieldfareDecodeServiceMain.appRuntimeOptions(DecodeRuntimeOptions())
        #expect(try app.resolvedRuntimeConfiguration(forceLogitsHead: false).earlyExpertRead == .off)
    }

    @Test func serviceMapsExpertPrefetchToExistingRuntimeMode() throws {
        for enabled in [false, true] {
            let wire = DecodeRuntimeOptions(expertPrefetchEnabled: enabled)
            let app = try TurboFieldfareDecodeServiceMain.appRuntimeOptions(wire)
            #expect(app.expertPrefetchEnabled == enabled)
            let runtime = try app.resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.earlyExpertRead == (enabled ? .router : .off))
        }
    }
}
