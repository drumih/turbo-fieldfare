import Foundation
import Inkling2GGUFQ4
import TemplatePlugin
import TinyTitan
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanPluginAPI

extension EngineFactories {
    /// The engine plugins compiled into this build (TPP/1, TT-130), strongest
    /// first only for readability: resolution is by probe strength.
    ///
    /// `TemplatePlugin` and `Inkling2GGUFQ4Plugin` are standalone plugins. The
    /// three `…Adapter` entries wrap engines that still borrow 6.x stages and
    /// are migrated into plugins of their own one by one (TT-132); each adapter
    /// is removed when its plugin lands.
    public static let builtInPlugins: [any EnginePlugin.Type] = [
        TemplatePlugin.self,
        Inkling2GGUFQ4Plugin.self,
        Ornith35B4BitAdapter.self,
        DenseQwen35Adapter.self,
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

    /// The manifest identity the adapters probe with (the 6.x reader).
    fileprivate static func identity(_ install: InstallProbe)
        -> (family: TinyTitan.ModelFamily, modelID: String, bits: Int)?
    {
        guard install.manifest != nil,
            let identity = try? TinyTitan.ManifestReader.peekIdentity(
                directoryURL: install.directory)
        else { return nil }
        return (identity.family, identity.modelID, identity.weightBits)
    }

    fileprivate static func selection(_ install: InstallProbe) -> GPUVariantSelection? {
        identity(install).map {
            gpuSelection(
                family: $0.family, manifestModelID: $0.modelID, routedExpertWeightBits: $0.bits)
        }
    }
}

/// Adapter (TT-132 migrates it): the shipped Ornith 1.5 35B-A3B 4-bit variant.
public enum Ornith35B4BitAdapter: EnginePlugin {
    public static let pluginID = "ai.tinytitan.adapter.ornith-1.5-35b-a3b-4bit"
    public static let protocolVersion = ProtocolVersion.current
    public static let displayName = "Ornith 1.5 35B-A3B 4-bit (adapter)"

    public static func probe(_ install: InstallProbe) -> MatchStrength? {
        EngineFactories.selection(install) == .ornith35B4Bit ? .exact : nil
    }

    /// The adapter keeps the flags' existing handling, which the 6.x stages own.
    public static func parameters(for install: InstallProbe) -> ParameterSupport {
        ParameterSupport()
    }

    public static func reasoningControl(for install: InstallProbe) -> ReasoningControl? { nil }

    public static func engineFactory(host: HostServices) -> EngineFactory {
        EngineFactories.variantGPUOrnith()
    }
}

/// Adapter (TT-132 migrates it): the shared dense-GPU engine for every dense
/// Qwen 3.5 install.
public enum DenseQwen35Adapter: EnginePlugin {
    public static let pluginID = "ai.tinytitan.adapter.qwen3.5-dense"
    public static let protocolVersion = ProtocolVersion.current
    public static let displayName = "Qwen 3.5 dense (adapter)"

    public static func probe(_ install: InstallProbe) -> MatchStrength? {
        EngineFactories.selection(install) == .qwen35DenseGPU ? .exact : nil
    }

    public static func parameters(for install: InstallProbe) -> ParameterSupport {
        ParameterSupport()
    }

    public static func reasoningControl(for install: InstallProbe) -> ReasoningControl? { nil }

    public static func engineFactory(host: HostServices) -> EngineFactory {
        EngineFactories.variantDenseGPU()
    }
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
