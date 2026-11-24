import Foundation
import DenseGPU
import Inkling2GGUFQ4
import Inkling2NVFP4
import Ornith35B4Bit
import TemplatePlugin
import TinyTitan
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanPluginAPI

extension EngineFactories {
    /// The engine plugins compiled into this build (TPP/1, TT-130), strongest
    /// first only for readability: resolution is by probe strength.
    ///
    /// `TemplatePlugin`, `Inkling2GGUFQ4Plugin`, `Ornith35B4BitPlugin` and
    /// `DenseQwen35Plugin` are standalone plugins. `LegacyEngineAdapter` wraps the
    /// 6.x engine, whose families still borrow its stages and
    /// are migrated into plugins of their own one by one (TT-132); each adapter
    /// is removed when its plugin lands.
    public static let builtInPlugins: [any EnginePlugin.Type] = [
        TemplatePlugin.self,
        Inkling2GGUFQ4Plugin.self,
        Inkling2NVFP4Plugin.self,
        Ornith35B4BitPlugin.self,
        DenseQwen35Plugin.self,
        LegacyEngineAdapter.self,
    ]

    /// The process's registry with the built-in plugins registered. An
    /// embedder registers its own plugins on the same registry.
    public static var pluginRegistry: PluginRegistry {
        _ = registerBuiltIns
        return .shared
    }

    private static let registerBuiltIns: Void = {
        for plugin in builtInPlugins {
            // Every built-in is built against this protocol version.
            try? PluginRegistry.shared.register(plugin)
        }
    }()
}

/// Adapter (TT-132 migrates it, family by family): the 6.x engine, which
/// serves every install no plugin claims more specifically.
public enum LegacyEngineAdapter: EnginePlugin {
    public static let pluginID = "ai.tinytitan.adapter.legacy-6x"
    public static let protocolVersion = ProtocolVersion.current
    public static let displayName = "TinyTitan 6.x engine (adapter)"

    public static func probe(_ install: InstallProbe) -> MatchStrength? { .fallback }

    public static func parameters(for install: InstallProbe) -> ParameterSupport {
        ParameterSupport()
    }

    public static func reasoningControl(for install: InstallProbe) -> ReasoningControl? { nil }

    public static func engineFactory(host: HostServices) -> EngineFactory {
        EngineFactories.sixXGPU()
    }
}
