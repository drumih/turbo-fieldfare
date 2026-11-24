import Foundation
import Testing
import TinyTitanCore
import TinyTitanPluginAPI

/// TPP/1: the registry picks the strongest probe, refuses a tie by name and a
/// plugin built against another major version.
@Suite struct PluginRegistryTests {
    enum Exact: EnginePlugin {
        static let pluginID = "test.exact"
        static let protocolVersion = ProtocolVersion.current
        static let displayName = "exact"
        static func probe(_ install: InstallProbe) -> MatchStrength? {
            install.directory.lastPathComponent == "mine" ? .exact : nil
        }
        static func parameters(for install: InstallProbe) -> ParameterSupport { .init() }
        static func reasoningControl(for install: InstallProbe) -> ReasoningControl? { nil }
        static func engineFactory(host: HostServices) -> EngineFactory { Stub.factory }
    }

    enum Twin: EnginePlugin {
        static let pluginID = "test.twin"
        static let protocolVersion = ProtocolVersion.current
        static let displayName = "twin"
        static func probe(_ install: InstallProbe) -> MatchStrength? {
            install.directory.lastPathComponent == "mine" ? .exact : nil
        }
        static func parameters(for install: InstallProbe) -> ParameterSupport { .init() }
        static func reasoningControl(for install: InstallProbe) -> ReasoningControl? { nil }
        static func engineFactory(host: HostServices) -> EngineFactory { Stub.factory }
    }

    enum Fallback: EnginePlugin {
        static let pluginID = "test.fallback"
        static let protocolVersion = ProtocolVersion.current
        static let displayName = "fallback"
        static func probe(_ install: InstallProbe) -> MatchStrength? { .fallback }
        static func parameters(for install: InstallProbe) -> ParameterSupport { .init() }
        static func reasoningControl(for install: InstallProbe) -> ReasoningControl? { nil }
        static func engineFactory(host: HostServices) -> EngineFactory { Stub.factory }
    }

    enum Future: EnginePlugin {
        static let pluginID = "test.future"
        static let protocolVersion = ProtocolVersion(major: 2, minor: 0)
        static let displayName = "future"
        static func probe(_ install: InstallProbe) -> MatchStrength? { .exact }
        static func parameters(for install: InstallProbe) -> ParameterSupport { .init() }
        static func reasoningControl(for install: InstallProbe) -> ReasoningControl? { nil }
        static func engineFactory(host: HostServices) -> EngineFactory { Stub.factory }
    }

    enum Stub {
        static var factory: EngineFactory {
            EngineFactory(
                buildModel: { _, _, _ in throw EngineFactoryError.unsupported("stub") },
                buildRunner: { _, _ in throw EngineFactoryError.unsupported("stub") })
        }
    }

    let mine = URL(fileURLWithPath: "/nonexistent/mine")
    let other = URL(fileURLWithPath: "/nonexistent/other")

    @Test func theStrongestProbeWins() throws {
        let registry = PluginRegistry()
        try registry.register(Fallback.self)
        try registry.register(Exact.self)
        #expect(try registry.resolve(mine).pluginID == "test.exact")
        #expect(try registry.resolve(other).pluginID == "test.fallback")
    }

    @Test func aTieIsAnErrorNamingBothPlugins() throws {
        let registry = PluginRegistry()
        try registry.register(Exact.self)
        try registry.register(Twin.self)
        #expect(
            throws: PluginRegistryError.ambiguous(
                directory: "/nonexistent/mine", pluginIDs: ["test.exact", "test.twin"])
        ) {
            try registry.resolve(mine)
        }
    }

    @Test func nothingClaimsIsAnError() throws {
        let registry = PluginRegistry()
        try registry.register(Exact.self)
        #expect(throws: PluginRegistryError.noPlugin(directory: "/nonexistent/other")) {
            try registry.resolve(other)
        }
    }

    @Test func anotherMajorVersionIsRefused() {
        let registry = PluginRegistry()
        #expect(
            throws: PluginRegistryError.incompatible(
                pluginID: "test.future", version: ProtocolVersion(major: 2, minor: 0))
        ) {
            try registry.register(Future.self)
        }
        #expect(registry.pluginIDs.isEmpty)
    }

    @Test func registrationIsIdempotent() throws {
        let registry = PluginRegistry()
        try registry.register(Exact.self)
        try registry.register(Exact.self)
        #expect(registry.pluginIDs == ["test.exact"])
    }
}

/// TPP/1 §4: a parameter is supported, limited to some values, or refused --
/// an explicit unsupported value is refused by name, an unsupported default is
/// replaced by the engine's own with a notice.
@Suite struct SupportTests {
    let limited = Support<Int>.only([16], reason: "measured")

    @Test func supportedAcceptsEverything() {
        #expect(Support<Int>.supported.refusal(for: 8, flag: "--x", plugin: "P") == nil)
        #expect(Support<Int>.supported.resolveDefault(8, flag: "--x", plugin: "P").notice == nil)
    }

    @Test func anExplicitUnsupportedValueIsRefusedWithTheReason() throws {
        let refusal = try #require(limited.refusal(for: 8, flag: "--x", plugin: "P"))
        #expect(refusal == "--x 8 is not supported by P: measured")
        #expect(limited.refusal(for: 16, flag: "--x", plugin: "P") == nil)
    }

    @Test func anUnsupportedDefaultIsReplacedAndAnnounced() throws {
        let resolved = limited.resolveDefault(8, flag: "--x", plugin: "P")
        #expect(resolved.value == 16)
        let notice = try #require(resolved.notice)
        #expect(notice.contains("does not apply") && notice.contains("measured"))
    }

    @Test func aRefusedParameterNamesThePlugin() throws {
        let refusal = try #require(
            Support<Int>.refused(reason: "no cache").refusal(for: 1, flag: "--y", plugin: "P"))
        #expect(refusal == "--y does not apply to P: no cache")
    }
}
