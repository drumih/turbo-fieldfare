import Foundation
import Testing
import TinyTitanLegacyEngine
import TinyTitanPluginAPI
import TinyTitanVariants

/// TT-130: every install on disk resolves through the plugin registry to the
/// engine it ran on before the registry existed. Manifest reads only; enabled
/// with `TINYTITAN_PLUGIN_MODELS_DIR` (normally `models`).
@Suite(.enabled(if: pluginModelsDirectory() != nil))
struct PluginResolutionTests {
    /// The engine each installed family ran on before TT-130.
    static func expected(_ name: String) -> String? {
        if name.hasPrefix("inkling-2-mini_GGUF-Q4") { return "ai.tinytitan.inkling2-gguf-q4" }
        if name.hasPrefix("ornith-1.5_35B_A3B_4Bit") {
            return "ai.tinytitan.ornith-1.5-35b-a3b-4bit"
        }
        if name.hasPrefix("qwen3.5_") { return "ai.tinytitan.adapter.qwen3.5-dense" }
        return nil
    }

    @Test func everyInstallResolvesToItsEngine() throws {
        let directory = try #require(pluginModelsDirectory())
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            .filter { !$0.hasSuffix(".lock") }
            .sorted()
        var checked = 0
        for name in names {
            guard let want = Self.expected(name) else { continue }
            let resolved = try EngineFactories.pluginRegistry.resolve(
                directory.appendingPathComponent(name))
            #expect(resolved.pluginID == want, "\(name)")
            checked += 1
        }
        #expect(checked > 0, "no known install under \(directory.path)")
    }

    @Test func builtInPluginIDsAreUnique() {
        let ids = EngineFactories.builtInPlugins.map { $0.pluginID }
        #expect(Set(ids).count == ids.count)
        #expect(EngineFactories.pluginRegistry.pluginIDs.starts(with: ids))
    }
}

/// `TINYTITAN_PLUGIN_MODELS_DIR`, when set.
func pluginModelsDirectory() -> URL? {
    ProcessInfo.processInfo.environment["TINYTITAN_PLUGIN_MODELS_DIR"].map {
        URL(fileURLWithPath: $0)
    }
}
