import Foundation
import TinyTitanCore

/// One finding from the TPP/1 plugin conformance checker (TT-131).
package struct PluginCheckFinding: Sendable, Equatable {
    package let check: String
    package let message: String

    package init(check: String, message: String) {
        self.check = check
        self.message = message
    }
}

/// Conformance result for one `EnginePlugin` (TT-131).
package struct PluginCheckReport: Sendable, Equatable {
    package let pluginID: String
    package let displayName: String
    package let protocolVersion: ProtocolVersion
    package let matchStrength: MatchStrength?
    package let findings: [PluginCheckFinding]

    package var passed: Bool { findings.isEmpty }

    package init(
        pluginID: String,
        displayName: String,
        protocolVersion: ProtocolVersion,
        matchStrength: MatchStrength? = nil,
        findings: [PluginCheckFinding]
    ) {
        self.pluginID = pluginID
        self.displayName = displayName
        self.protocolVersion = protocolVersion
        self.matchStrength = matchStrength
        self.findings = findings
    }
}

/// Model-free and install-level conformance checks for TPP/1 plugins (TT-131,
/// `docs/plugin-api-v1.md` §7).
package enum PluginConformance {
    /// Checks a plugin's static metadata, probe hygiene, parameter table
    /// honesty, reasoning declaration, and factory construction without loading
    /// weights or touching Metal.
    package static func checkStatic(
        _ plugin: any EnginePlugin.Type,
        probe: InstallProbe? = nil
    ) -> PluginCheckReport {
        var findings: [PluginCheckFinding] = []
        let emptyProbe =
            probe
            ?? InstallProbe(
                directory: URL(fileURLWithPath: "/nonexistent/tinytitan-plugin-check"),
                manifest: nil, declaredFamily: nil)
        let isFallback = plugin.probe(emptyProbe) == .fallback

        if plugin.pluginID.isEmpty || !plugin.pluginID.contains(".") {
            findings.append(
                PluginCheckFinding(
                    check: "identity",
                    message: "pluginID must be a non-empty reverse-DNS identifier"))
        }
        if plugin.displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            findings.append(
                PluginCheckFinding(
                    check: "identity",
                    message: "displayName must be non-empty"))
        }
        if plugin.protocolVersion.major != ProtocolVersion.current.major {
            findings.append(
                PluginCheckFinding(
                    check: "protocol",
                    message: "protocolVersion \(plugin.protocolVersion) does not match host major "
                        + "\(ProtocolVersion.current.major)"))
        }
        if !isFallback, let unexpected = plugin.probe(emptyProbe), probe == nil {
            findings.append(
                PluginCheckFinding(
                    check: "probe",
                    message: "claimed empty directory with strength \(unexpected)"))
        }
        if !isFallback, plugin.reasoningControl(for: emptyProbe) == nil {
            findings.append(
                PluginCheckFinding(
                    check: "reasoning",
                    message: "non-fallback plugin must declare a ReasoningControl"))
        }

        let params = plugin.parameters(for: emptyProbe)
        checkSupport(params.kvCachePrecision, flag: "--kv-bits", into: &findings)
        checkSupport(params.ropeScaling, flag: "--rope-scaling", into: &findings)
        checkSupport(params.hostKVWindow, flag: "--host-kv-window", into: &findings)
        checkSupport(
            params.concurrentSequences, flag: "--max-concurrent-sequences", into: &findings)
        checkSupport(params.multiTokenPrediction, flag: "--mtp-model", into: &findings)
        checkSupport(params.expertStreaming, flag: "--ram-budget", into: &findings)

        // Mandatory engine parameters (TPP/1.1): every production LLM plugin
        // must honour --kv-bits (4, 8, 16) and --max-concurrent-sequences.
        if !isFallback, plugin.pluginID != "ai.tinytitan.template" {
            for precision in KVCachePrecision.allCases
            where !params.kvCachePrecision.accepts(precision) {
                findings.append(
                    PluginCheckFinding(
                        check: "mandatory-parameter",
                        message: "--kv-bits \(precision.rawValue) is required by the TinyTitan "
                            + "plugin protocol; plugin refused or restricted it"))
            }
            if !params.concurrentSequences.accepts(1) || !params.concurrentSequences.accepts(2) {
                findings.append(
                    PluginCheckFinding(
                        check: "mandatory-parameter",
                        message: "--max-concurrent-sequences is required by the TinyTitan "
                            + "plugin protocol; plugin refused or restricted it"))
            }
        }

        _ = plugin.engineFactory(host: .standard)

        return PluginCheckReport(
            pluginID: plugin.pluginID,
            displayName: plugin.displayName,
            protocolVersion: plugin.protocolVersion,
            matchStrength: plugin.probe(emptyProbe),
            findings: findings)
    }

    /// Resolves `directory` through `registry`, runs static checks on the
    /// winning plugin, and verifies metadata, architecture resolution, and
    /// tokenizer discovery without loading weights.
    package static func checkInstall(
        _ directory: URL,
        registry: PluginRegistry
    ) -> PluginCheckReport {
        let resolved: ResolvedPlugin
        do {
            resolved = try registry.resolve(directory)
        } catch {
            return PluginCheckReport(
                pluginID: "<unresolved>",
                displayName: directory.lastPathComponent,
                protocolVersion: .current,
                matchStrength: nil,
                findings: [PluginCheckFinding(check: "resolve", message: "\(error)")])
        }

        let probe = InstallProbe.read(directory)
        var findings = checkStatic(resolved.plugin, probe: probe).findings
        let factory = resolved.engineFactory()

        do {
            let metadata = try factory.metadata(directory)
            if metadata.modelID.isEmpty {
                findings.append(
                    PluginCheckFinding(check: "metadata", message: "modelID is empty"))
            }
            if metadata.family.isEmpty {
                findings.append(
                    PluginCheckFinding(check: "metadata", message: "family is empty"))
            }
        } catch {
            findings.append(
                PluginCheckFinding(check: "metadata", message: "metadata failed: \(error)"))
        }

        do {
            let arch = try factory.resolveArchitecture(directory)
            if arch.vocabSize <= 0 || arch.numLayers <= 0 || arch.hiddenSize <= 0 {
                findings.append(
                    PluginCheckFinding(
                        check: "architecture",
                        message: "invalid geometry (layers=\(arch.numLayers), "
                            + "hidden=\(arch.hiddenSize), vocab=\(arch.vocabSize))"))
            }
        } catch {
            findings.append(
                PluginCheckFinding(
                    check: "architecture", message: "resolveArchitecture failed: \(error)"))
        }

        do {
            let folder =
                try factory.tokenizerFolder?(directory)
                ?? GFTokenizer.tokenizerFolder(forModelDirectory: directory)
            if folder == nil {
                findings.append(
                    PluginCheckFinding(
                        check: "tokenizer",
                        message: "no tokenizer/ folder or plugin tokenizerFolder hook"))
            }
        } catch {
            findings.append(
                PluginCheckFinding(
                    check: "tokenizer", message: "tokenizerFolder failed: \(error)"))
        }

        return PluginCheckReport(
            pluginID: resolved.pluginID,
            displayName: resolved.plugin.displayName,
            protocolVersion: resolved.plugin.protocolVersion,
            matchStrength: resolved.strength,
            findings: findings)
    }

    private static func checkSupport<V: Sendable & Equatable>(
        _ support: Support<V>,
        flag: String,
        into findings: inout [PluginCheckFinding]
    ) {
        switch support {
        case .supported:
            break
        case .only(let values, let reason):
            if values.isEmpty {
                findings.append(
                    PluginCheckFinding(
                        check: "parameters",
                        message: "\(flag): .only([]) must list at least one value"))
            }
            if reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                findings.append(
                    PluginCheckFinding(
                        check: "parameters",
                        message: "\(flag): .only reason must be non-empty"))
            }
        case .refused(let reason):
            if reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                findings.append(
                    PluginCheckFinding(
                        check: "parameters",
                        message: "\(flag): .refused reason must be non-empty"))
            }
        }
    }
}
