import Foundation
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanPluginAPI
import TinyTitanVariants

/// CLI options for `TinyTitanCLI plugin-check` (TT-131).
public struct PluginCheckOptions: Equatable, Sendable {
    public var installPath: String?
    public var pluginID: String?
    public var json: Bool
    public var help: Bool

    public init(
        installPath: String? = nil,
        pluginID: String? = nil,
        json: Bool = false,
        help: Bool = false
    ) {
        self.installPath = installPath
        self.pluginID = pluginID
        self.json = json
        self.help = help
    }

    public static let usage = """
        usage: TinyTitanCLI plugin-check [--install <dir>] [--plugin <id>] [--json]

        Check compiled-in TPP/1 engine plugins against the protocol contract
        (docs/plugin-api-v1.md §7). Without --install, checks every built-in
        plugin's static contract; with --install, resolves that directory and
        verifies metadata, architecture, and tokenizer discovery without
        loading weights.
        """

    public static func parse(_ argv: [String]) throws -> PluginCheckOptions {
        var options = PluginCheckOptions()
        var index = 0
        while index < argv.count {
            let arg = argv[index]
            switch arg {
            case "-h", "--help":
                options.help = true
            case "--json":
                options.json = true
            case "--install":
                index += 1
                guard index < argv.count else {
                    throw PluginCheckOptionError.missingValue("--install")
                }
                options.installPath = argv[index]
            case "--plugin":
                index += 1
                guard index < argv.count else {
                    throw PluginCheckOptionError.missingValue("--plugin")
                }
                options.pluginID = argv[index]
            default:
                throw PluginCheckOptionError.unknownFlag(arg)
            }
            index += 1
        }
        return options
    }
}

public enum PluginCheckOptionError: Error, CustomStringConvertible, Equatable {
    case missingValue(String)
    case unknownFlag(String)

    public var description: String {
        switch self {
        case .missingValue(let flag): "\(flag) requires a value"
        case .unknownFlag(let flag): "unknown option: \(flag)"
        }
    }
}

/// Runs TPP/1 conformance checks from `TinyTitanCLI plugin-check` (TT-131).
public enum PluginCheckCommand {
    public static func run(
        _ options: PluginCheckOptions,
        stdout: FileHandle = .standardOutput,
        stderr: FileHandle = .standardError
    ) -> Int32 {
        if options.help {
            stdout.write(Data((PluginCheckOptions.usage + "\n").utf8))
            return 0
        }

        let reports: [PluginCheckReport]
        if let installPath = options.installPath {
            let url = URL(fileURLWithPath: installPath).standardizedFileURL
            reports = [
                PluginConformance.checkInstall(url, registry: EngineFactories.pluginRegistry)
            ]
        } else {
            let plugins: [any EnginePlugin.Type]
            if let wanted = options.pluginID {
                plugins = EngineFactories.builtInPlugins.filter { $0.pluginID == wanted }
                if plugins.isEmpty {
                    stderr.write(Data("error: no built-in plugin with id \(wanted)\n".utf8))
                    return 1
                }
            } else {
                plugins = EngineFactories.builtInPlugins
            }
            reports = plugins.map { PluginConformance.checkStatic($0) }
        }

        if options.json {
            stdout.write(Data((renderJSON(reports) + "\n").utf8))
        } else {
            stdout.write(Data(renderText(reports).utf8))
        }
        return reports.allSatisfy(\.passed) ? 0 : 1
    }

    private static func renderText(_ reports: [PluginCheckReport]) -> String {
        var lines: [String] = []
        for report in reports {
            let status = report.passed ? "PASS" : "FAIL"
            lines.append(
                "[\(status)] \(report.pluginID) — \(report.displayName) "
                    + "(TPP/\(report.protocolVersion))")
            for finding in report.findings {
                lines.append("  - [\(finding.check)] \(finding.message)")
            }
        }
        let passed = reports.filter(\.passed).count
        lines.append("\(passed)/\(reports.count) plugins passed conformance checks")
        return lines.joined(separator: "\n") + "\n"
    }

    private static func renderJSON(_ reports: [PluginCheckReport]) -> String {
        let rows: [[String: Any]] = reports.map { report in
            [
                "pluginID": report.pluginID,
                "displayName": report.displayName,
                "protocolVersion": "\(report.protocolVersion)",
                "passed": report.passed,
                "findings": report.findings.map {
                    ["check": $0.check, "message": $0.message]
                },
            ]
        }
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: ["plugins": rows], options: [.prettyPrinted, .sortedKeys]),
            let text = String(data: data, encoding: .utf8)
        else { return "{\"plugins\":[]}" }
        return text
    }
}
