import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitanPluginAPI

/// Model-free conformance kit tests (TT-131).
@Suite struct PluginConformanceTests {
    private struct UnusedError: Error {}

    private enum GoodPlugin: EnginePlugin {
        static let pluginID = "com.example.good"
        static let protocolVersion = ProtocolVersion.current
        static let displayName = "Good Plugin"
        static func probe(_ install: InstallProbe) -> MatchStrength? {
            install.declaredFamily == "good" ? .exact : nil
        }
        static func parameters(for install: InstallProbe) -> ParameterSupport {
            ParameterSupport(
                kvCachePrecision: .only([.fp16], reason: "fp16 only"),
                multiTokenPrediction: .refused(reason: "no draft head"))
        }
        static func reasoningControl(for install: InstallProbe) -> ReasoningControl? {
            .binaryThinking
        }
        static func engineFactory(host: HostServices) -> EngineFactory {
            EngineFactory(
                prepareLoad: { _ in throw UnusedError() },
                metadata: { _ in throw UnusedError() },
                resolveArchitecture: { _ in throw UnusedError() },
                preview: { _, _ in throw UnusedError() },
                buildModel: { _, _, _ in throw UnusedError() },
                buildRunner: { _, _ in throw UnusedError() },
                runRawCompletion: { _ in throw UnusedError() })
        }
    }

    private enum BadPlugin: EnginePlugin {
        static let pluginID = "bad"
        static let protocolVersion = ProtocolVersion(major: 2, minor: 0)
        static let displayName = "   "
        static func probe(_ install: InstallProbe) -> MatchStrength? { .exact }
        static func parameters(for install: InstallProbe) -> ParameterSupport {
            ParameterSupport(
                kvCachePrecision: .only([], reason: ""),
                multiTokenPrediction: .refused(reason: " "))
        }
        static func reasoningControl(for install: InstallProbe) -> ReasoningControl? { nil }
        static func engineFactory(host: HostServices) -> EngineFactory {
            GoodPlugin.engineFactory(host: host)
        }
    }

    @Test func aCompliantPluginPassesStaticChecks() {
        let report = PluginConformance.checkStatic(GoodPlugin.self)
        #expect(report.passed)
        #expect(report.findings.isEmpty)
    }

    @Test func aBrokenPluginReportsEveryViolation() {
        let report = PluginConformance.checkStatic(BadPlugin.self)
        #expect(!report.passed)
        let checks = Set(report.findings.map(\.check))
        #expect(checks.contains("identity"))
        #expect(checks.contains("protocol"))
        #expect(checks.contains("probe"))
        #expect(checks.contains("reasoning"))
        #expect(checks.contains("parameters"))
    }
}
