import Foundation
import Testing
import TinyTitanCore
import TinyTitanPluginAPI
import TinyTitanServerCore

/// TPP/1 §4 on the server: each flag the operator set is checked against the
/// resolved plugin's table before the model loads.
@Suite struct PluginParameterGateTests {
    /// A plugin that runs 16-bit K/V only, one sequence, no MTP.
    enum Narrow: EnginePlugin {
        static let pluginID = "test.narrow"
        static let protocolVersion = ProtocolVersion.current
        static let displayName = "Narrow"
        static func probe(_ install: InstallProbe) -> MatchStrength? { .exact }
        static func parameters(for install: InstallProbe) -> ParameterSupport {
            ParameterSupport(
                kvCachePrecision: .only([.fp16], reason: "measured"),
                concurrentSequences: .only([1], reason: "one sequence"),
                multiTokenPrediction: .refused(reason: "no draft head"))
        }
        static func reasoningControl(for install: InstallProbe) -> ReasoningControl? { nil }
        static func engineFactory(host: HostServices) -> EngineFactory {
            EngineFactory(
                buildModel: { _, _, _ in throw EngineFactoryError.unsupported("stub") },
                buildRunner: { _, _ in throw EngineFactoryError.unsupported("stub") })
        }
    }

    let plugin = ResolvedPlugin(
        plugin: Narrow.self,
        probe: InstallProbe(
            directory: URL(fileURLWithPath: "/nonexistent"), manifest: nil, declaredFamily: nil),
        strength: .exact)

    func gate(_ flags: [String]) throws -> (refusal: String?, notices: [String]) {
        let arguments = try ServerArguments.parse(["--model", "/nonexistent"] + flags)
        return PluginParameterGate.check(arguments, plugin: plugin)
    }

    @Test func anExplicitUnsupportedKVPrecisionIsRefused() throws {
        #expect(
            try gate(["--kv-bits", "8"]).refusal
                == "--kv-bits 8 is not supported by Narrow: measured")
        #expect(try gate(["--kv-bits", "16"]).refusal == nil)
    }

    @Test func theDefaultKVPrecisionIsReplacedAndAnnounced() throws {
        let result = try gate([])
        #expect(result.refusal == nil)
        #expect(result.notices.count == 1)
        #expect(result.notices.first?.contains("default --kv-bits 8 does not apply") == true)
    }

    @Test func unsupportedMTPAndWidthAreRefused() throws {
        #expect(
            try gate(["--mtp-model", "/nonexistent/mtp"]).refusal
                == "--mtp-model does not apply to Narrow: no draft head")
        #expect(
            try gate(["--max-concurrent-sequences", "2"]).refusal
                == "--max-concurrent-sequences 2 is not supported by Narrow: one sequence")
    }
}
