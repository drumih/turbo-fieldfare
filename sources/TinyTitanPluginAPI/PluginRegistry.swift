import Foundation
import TinyTitanCore

/// The plugins this process knows, and the one rule for choosing among them.
///
/// Built-in plugins are registered by the build (`TinyTitanVariants`);
/// an embedder registers its own the same way. Every front end resolves an
/// install only through `resolve(_:)`.
///
/// unchecked-invariant: `plugins` is read and written only under `lock`.
public final class PluginRegistry: @unchecked Sendable {
    public static let shared = PluginRegistry()

    private let lock = NSLock()
    private var plugins: [any EnginePlugin.Type] = []

    public init() {}

    /// Adds `plugin`. Idempotent by `pluginID`; a plugin built against another
    /// major protocol version is refused.
    public func register(_ plugin: any EnginePlugin.Type) throws {
        guard plugin.protocolVersion.major == ProtocolVersion.current.major else {
            throw PluginRegistryError.incompatible(
                pluginID: plugin.pluginID, version: plugin.protocolVersion)
        }
        lock.withLock {
            if !plugins.contains(where: { $0.pluginID == plugin.pluginID }) {
                plugins.append(plugin)
            }
        }
    }

    /// The registered plugin ids, in registration order.
    public var pluginIDs: [String] {
        lock.withLock { plugins.map { $0.pluginID } }
    }

    /// The plugin that serves `directory`: the strongest probe. A tie at the
    /// top is an error naming the plugins, never a silent pick.
    public func resolve(_ directory: URL) throws -> ResolvedPlugin {
        let probe = InstallProbe.read(directory)
        let candidates = lock.withLock { plugins }
        let matches = candidates.compactMap { plugin in
            plugin.probe(probe).map { (plugin: plugin, strength: $0) }
        }
        guard let best = matches.map(\.strength).max() else {
            throw PluginRegistryError.noPlugin(directory: probe.directory.path)
        }
        let top = matches.filter { $0.strength == best }
        guard top.count == 1, let chosen = top.first else {
            throw PluginRegistryError.ambiguous(
                directory: probe.directory.path, pluginIDs: top.map { $0.plugin.pluginID })
        }
        return ResolvedPlugin(plugin: chosen.plugin, probe: probe, strength: best)
    }
}

/// An install and the plugin chosen for it.
public struct ResolvedPlugin: Sendable {
    public let plugin: any EnginePlugin.Type
    public let probe: InstallProbe
    public let strength: MatchStrength

    public init(plugin: any EnginePlugin.Type, probe: InstallProbe, strength: MatchStrength) {
        self.plugin = plugin
        self.probe = probe
        self.strength = strength
    }

    public var pluginID: String { plugin.pluginID }
    public var parameters: ParameterSupport { plugin.parameters(for: probe) }
    public var reasoningControl: ReasoningControl? { plugin.reasoningControl(for: probe) }
    public var chatDialect: (any ChatDialect)? { plugin.chatDialect(for: probe) }

    public func engineFactory(host: HostServices = .standard) -> EngineFactory {
        plugin.engineFactory(host: host)
    }
}

public enum PluginRegistryError: Error, Equatable, CustomStringConvertible {
    case noPlugin(directory: String)
    case ambiguous(directory: String, pluginIDs: [String])
    case incompatible(pluginID: String, version: ProtocolVersion)

    public var description: String {
        switch self {
        case .noPlugin(let directory):
            "no engine plugin serves \(directory)"
        case .ambiguous(let directory, let ids):
            "more than one engine plugin claims \(directory) equally: \(ids.joined(separator: ", "))"
        case .incompatible(let id, let version):
            "plugin \(id) implements protocol \(version); this TinyTitan implements "
                + "\(ProtocolVersion.current)"
        }
    }
}
