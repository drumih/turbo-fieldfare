import Foundation

/// `tinytitan catalog` (7.0 spec §5, plan task 6): the one registry rendered and
/// checked. `--check` refuses an inconsistent registry ↔ target ↔ qualification
/// state; `--markdown` renders the model list. The registry is data, not code;
/// adding a model is one registry entry plus one target.
package struct CatalogOptions: Equatable, Sendable {
    package var check: Bool = false
    package var markdown: Bool = false
    package var registryPath: String = "registry/registry.json"
    package var help: Bool = false

    package init(
        check: Bool = false, markdown: Bool = false,
        registryPath: String = "registry/registry.json", help: Bool = false
    ) {
        self.check = check
        self.markdown = markdown
        self.registryPath = registryPath
        self.help = help
    }

    package static let usage = """
        usage: TinyTitanCLI catalog [--check] [--markdown] [--registry <path>]

        The one model registry (registry/registry.json): the single source for
        the installer menu, the CLI list, display names and the benchmark tables.

        options:
          --check               Refuse a registry entry naming a missing target,
                                a variant target with no registry entry, or a
                                shipped variant with no qualification record.
          --markdown            Render the model list as a Markdown table.
          --registry <path>     Registry file (default: registry/registry.json).
          --help                Show this message.
        """

    package static func parse(_ arguments: [String]) throws -> CatalogOptions {
        var options = CatalogOptions()
        var index = 0
        while index < arguments.count {
            let arg = arguments[index]
            switch arg {
            case "--check": options.check = true
            case "--markdown": options.markdown = true
            case "--registry":
                index += 1
                guard index < arguments.count else { throw CatalogError.missingValue("--registry") }
                options.registryPath = arguments[index]
            case "--help", "-h": options.help = true
            default: throw CatalogError.unknownArgument(arg)
            }
            index += 1
        }
        return options
    }
}

package enum CatalogError: Error, CustomStringConvertible {
    case missingValue(String)
    case unknownArgument(String)
    case unreadableRegistry(String)

    package var description: String {
        switch self {
        case .missingValue(let flag): return "\(flag) requires a value"
        case .unknownArgument(let arg): return "unknown catalog argument: \(arg)"
        case .unreadableRegistry(let path): return "cannot read registry: \(path)"
        }
    }
}

package enum CatalogCommand {
    /// The pure consistency check: registry ↔ target ↔ qualification. Returns
    /// one finding per inconsistency; empty means consistent. `targets` is every
    /// SwiftPM target name; `variantTargets` is the subset that are engine
    /// variants (the sources/Engines directories).
    package static func findings(
        registry: [String: Any], targets: Set<String>, variantTargets: Set<String>
    ) -> [String] {
        var findings: [String] = []
        var namedTargets: Set<String> = []
        let models = registry["models"] as? [[String: Any]] ?? []
        for model in models {
            let variants = model["variants"] as? [[String: Any]] ?? []
            for variant in variants {
                let id = variant["id"] as? String ?? "<unnamed>"
                if let target = variant["target"] as? String {
                    namedTargets.insert(target)
                    if !targets.contains(target) {
                        findings.append("registry entry '\(id)' names missing target '\(target)'")
                    }
                }
                let shipped = variant["shipped"] as? Bool ?? false
                let qualified = (variant["qualification"] as? [String: Any]) != nil
                if shipped && !qualified {
                    findings.append("shipped variant '\(id)' has no qualification record")
                }
            }
        }
        for variantTarget in variantTargets.sorted() where !namedTargets.contains(variantTarget) {
            findings.append("variant target '\(variantTarget)' has no registry entry")
        }
        return findings
    }

    package static func run(_ options: CatalogOptions) -> Int32 {
        if options.help {
            print(CatalogOptions.usage)
            return 0
        }
        guard let data = FileManager.default.contents(atPath: options.registryPath),
            let registry = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            FileHandle.standardError.write(
                Data("error: \(CatalogError.unreadableRegistry(options.registryPath))\n".utf8))
            return 2
        }

        if options.markdown {
            print(renderMarkdown(registry))
        }

        if options.check {
            let targets = packageTargetNames()
            let variantTargets = engineVariantTargets()
            let problems = findings(
                registry: registry, targets: targets, variantTargets: variantTargets)
            if problems.isEmpty {
                print("catalog: ok (registry ↔ target ↔ qualification consistent)")
                return 0
            }
            for problem in problems {
                FileHandle.standardError.write(Data("catalog: \(problem)\n".utf8))
            }
            return 1
        }
        return 0
    }

    /// Render the model list as a Markdown table (the installer menu and the
    /// README/wiki tables are generated from this in later phases).
    package static func renderMarkdown(_ registry: [String: Any]) -> String {
        var lines = [
            "| Model | Variant | Bits | Engine | Shipped |",
            "| --- | --- | ---: | --- | --- |",
        ]
        let models = registry["models"] as? [[String: Any]] ?? []
        for model in models {
            let name = model["displayName"] as? String ?? model["modelID"] as? String ?? "?"
            for variant in model["variants"] as? [[String: Any]] ?? [] {
                let id = variant["id"] as? String ?? "?"
                let bits = variant["bits"] as? Int ?? 0
                let engine = variant["engine"] as? String ?? "?"
                let shipped = (variant["shipped"] as? Bool ?? false) ? "yes" : "no"
                lines.append("| \(name) | \(id) | \(bits) | \(engine) | \(shipped) |")
            }
        }
        return lines.joined(separator: "\n")
    }

    /// Every SwiftPM target name, from the manifest rather than from artifacts
    /// (RELEASE.md §1.5: guard the plan, not the byproduct).
    private static func packageTargetNames() -> Set<String> {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["swift", "package", "describe", "--type", "json"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return []
        }
        // Drain the pipe before reaping: `swift package describe` emits more
        // than the 64 KB pipe buffer, so waiting for exit first deadlocks the
        // child blocked in write while we block in waitUntilExit.
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let targets = json["targets"] as? [[String: Any]]
        else { return [] }
        return Set(targets.compactMap { $0["name"] as? String })
    }

    /// The engine-variant targets: the directories under sources/Engines.
    private static func engineVariantTargets() -> Set<String> {
        let entries =
            (try? FileManager.default.contentsOfDirectory(atPath: "sources/Engines")) ?? []
        return Set(
            entries.filter {
                var isDir: ObjCBool = false
                let path = "sources/Engines/\($0)"
                return FileManager.default.fileExists(atPath: path, isDirectory: &isDir)
                    && isDir.boolValue
            })
    }
}
