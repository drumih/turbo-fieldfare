import Darwin
import Testing

@testable import TurboFieldfare

@Suite(.serialized) struct InteractivityWatchdogEnvironmentTests {
    private static let key = "AGX_RELAX_CDM_CTXSTORE_TIMEOUT"

    private static func read() -> String? {
        guard let raw = getenv(key) else { return nil }
        return String(cString: raw)
    }

    private static func withCleanEnvironment(_ body: () throws -> Void) rethrows {
        let saved = read()
        unsetenv(key)
        defer {
            if let saved {
                setenv(key, saved, 1)
            } else {
                unsetenv(key)
            }
        }
        try body()
    }

    #if os(macOS)
    @Test func constructingAMetalContextSetsTheWatchdogRelaxationOnMacOS() throws {
        try Self.withCleanEnvironment {
            #expect(Self.read() == nil)
            _ = try MetalContext()
            #expect(Self.read() == "1")
        }
    }

    @Test func anExplicitOperatorOverrideIsNotOverwritten() throws {
        try Self.withCleanEnvironment {
            setenv(Self.key, "0", 1)
            _ = try MetalContext()
            #expect(Self.read() == "0")
        }
    }
    #else
    @Test func constructingAMetalContextDoesNotSetTheMacOSWatchdogRelaxation() throws {
        try Self.withCleanEnvironment {
            #expect(Self.read() == nil)
            _ = try MetalContext()
            #expect(Self.read() == nil)
        }
    }
    #endif
}
