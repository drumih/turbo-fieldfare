import Foundation
import Testing
import TinyTitanCore
import TinyTitanLegacyEngine
@testable import TinyTitanPluginAPI
import TinyTitanVariants

/// Pre-release conformance gate over every compiled-in production plugin in
/// `EngineFactories.pluginRegistry`: fails CI and release builds if any LLM
/// plugin refuses mandatory TinyTitan parameters (`--kv-bits 4|8|16` or
/// `--max-concurrent-sequences`) or violates TPP/1.1 static invariants.
@Suite("Live plugin registry conformance")
struct PluginRegistryConformanceTests {
    @Test func everyRegisteredPluginPassesMandatoryConformance() {
        let plugins = EngineFactories.builtInPlugins
        #expect(!plugins.isEmpty)
        for plugin in plugins {
            let report = PluginConformance.checkStatic(plugin)
            let summary = report.findings
                .map { "\($0.check): \($0.message)" }
                .joined(separator: "; ")
            #expect(
                report.passed,
                Comment(rawValue: "Plugin \(plugin.pluginID) failed conformance: \(summary)"))
        }
    }

    @Test func everyProductionPluginAcceptsAllKVPrecisionsAndConcurrency() {
        let emptyProbe = InstallProbe(
            directory: URL(fileURLWithPath: "/nonexistent/tinytitan-conformance"),
            manifest: nil, declaredFamily: nil)
        let plugins = EngineFactories.builtInPlugins
        var checked = 0
        for plugin in plugins {
            if plugin.pluginID == "ai.tinytitan.template"
                || plugin.probe(emptyProbe) == .fallback
            {
                continue
            }
            checked += 1
            let params = plugin.parameters(for: emptyProbe)
            for precision in KVCachePrecision.allCases {
                #expect(
                    params.kvCachePrecision.accepts(precision),
                    Comment(
                        rawValue: "\(plugin.pluginID) must accept --kv-bits \(precision.rawValue)"))
            }
            #expect(
                params.concurrentSequences.accepts(1) && params.concurrentSequences.accepts(2),
                Comment(rawValue: "\(plugin.pluginID) must support --max-concurrent-sequences"))
        }
        #expect(checked >= 4)
    }
}
