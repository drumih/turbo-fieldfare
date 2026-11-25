import Foundation
import TinyTitanCore

/// TinyTitan Plugin Protocol v1 (TPP/1): how one LLM joins TinyTitan.
///
/// A plugin is a standalone library target, compiled into the build, that
/// depends on `TinyTitanPluginAPI`, `TinyTitanCore` and system frameworks only.
/// Everything model-specific -- recognising the install, its chat format, its
/// load plan, its kernels, its sampling, its tuning -- lives inside the plugin.
/// The host (library, CLI, server, catalog) holds no model code: it resolves an
/// install through `PluginRegistry` and drives the engine the plugin returns,
/// so every TinyTitan feature reaches the model through this protocol.
///
/// The specification is `docs/plugin-api-v1.md`; the golden base to copy is
/// `sources/Plugins/TemplatePlugin`.
public protocol EnginePlugin: Sendable {
    /// Stable reverse-DNS identity, e.g. `ai.tinytitan.inkling2-gguf-q4`.
    static var pluginID: String { get }
    /// The protocol this plugin implements. The host refuses another major.
    static var protocolVersion: ProtocolVersion { get }
    /// A human name for logs and the catalog, e.g. "Inkling 2 Mini Q4".
    static var displayName: String { get }

    /// Whether this plugin serves `install`, from the manifest alone: no
    /// weights, no Metal, no directory walk. nil: not this plugin's install.
    /// The host takes the strongest match; a tie between plugins is an error
    /// naming both, never a silent pick.
    static func probe(_ install: InstallProbe) -> MatchStrength?

    /// Which user-facing parameters the engine honours (§4 of the spec). The
    /// host refuses an explicit unsupported value with the plugin's reason and
    /// replaces an unsupported default with the plugin's, announcing it.
    static func parameters(for install: InstallProbe) -> ParameterSupport

    /// The model's thinking and effort levels. nil only for the 6.x adapter,
    /// whose families still carry their own table; a plugin returns its own.
    static func reasoningControl(for install: InstallProbe) -> ReasoningControl?

    /// The chat dialect this plugin's tokenizer speaks (TPP/1.1, TT-132).
    /// nil defers to the tokenizer's auto-detected dialect (used by the 6.x
    /// adapter and TPP/1.0 plugins).
    static func chatDialect(for install: InstallProbe) -> (any ChatDialect)?

    /// The engine: load planning, model, runner, completion loop and the
    /// tokenizer source, as the shared session consumes them (Core
    /// `EngineFactory`). `host` is the plugin's only door back into TinyTitan.
    static func engineFactory(host: HostServices) -> EngineFactory
}

extension EnginePlugin {
    public static func chatDialect(for install: InstallProbe) -> (any ChatDialect)? { nil }
}

/// The protocol version a plugin was built against.
public struct ProtocolVersion: Sendable, Hashable, Comparable, CustomStringConvertible {
    public let major: Int
    public let minor: Int

    public init(major: Int, minor: Int) {
        self.major = major
        self.minor = minor
    }

    /// TPP/1.1, the version this host implements (adds `chatDialect(for:)`).
    public static let current = ProtocolVersion(major: 1, minor: 1)

    public var description: String { "\(major).\(minor)" }

    public static func < (lhs: Self, rhs: Self) -> Bool {
        (lhs.major, lhs.minor) < (rhs.major, rhs.minor)
    }
}

/// How specifically a plugin claims an install.
public enum MatchStrength: Int, Sendable, Comparable {
    /// Serves anything nobody else claims (the 6.x adapter).
    case fallback = 0
    /// Serves the install's family generically.
    case family = 1
    /// Built for exactly this install (family, scheme and model).
    case exact = 2

    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
}

/// What a plugin may read about an install before it is chosen: the parsed
/// `.ssdai` manifest when the directory has a readable one. Built by the host
/// with a bounded read; a plugin never walks the directory to probe.
public struct InstallProbe: Sendable {
    public let directory: URL
    /// nil: no readable `.ssdai` manifest (a snapshot, a GGUF directory).
    public let manifest: Manifest?
    /// The manifest's `arch.family`, when it declares one.
    public let declaredFamily: String?

    public init(directory: URL, manifest: Manifest?, declaredFamily: String?) {
        self.directory = directory
        self.manifest = manifest
        self.declaredFamily = declaredFamily
    }

    /// Reads `directory`'s manifest, if any. Never throws: an unreadable
    /// manifest is a probe with no manifest.
    public static func read(_ directory: URL) -> InstallProbe {
        let url = directory.standardizedFileURL
        return InstallProbe(
            directory: url,
            manifest: try? SSDAIManifestReader.read(directoryURL: url),
            declaredFamily: (try? SSDAIManifestReader.declaredFamily(directoryURL: url)) ?? nil)
    }
}

/// The host services a plugin may use. Logging goes to stderr through the
/// host, so a plugin never writes to stdout (the library's stdout belongs to
/// the embedding program).
public struct HostServices: Sendable {
    public let log: @Sendable (String) -> Void
    /// Where a plugin may keep derived files (an exported tokenizer).
    public let cacheDirectory: URL

    public init(log: @escaping @Sendable (String) -> Void, cacheDirectory: URL) {
        self.log = log
        self.cacheDirectory = cacheDirectory
    }

    /// stderr and the user's cache directory under `TinyTitan/`.
    public static let standard = HostServices(
        log: { FileHandle.standardError.write(Data(($0 + "\n").utf8)) },
        cacheDirectory: (FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
            .first ?? FileManager.default.temporaryDirectory)
            .appendingPathComponent("TinyTitan", isDirectory: true))
}
