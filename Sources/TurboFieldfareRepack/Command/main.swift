import Foundation
import TurboFieldfareRepackCore

private let usage = """
Usage:
  TurboFieldfareRepack --output <model.gturbo> [--overwrite] [--resume] [--repo-id <owner/repo>] [--revision <ref>]
  TurboFieldfareRepack --discard-partial --output <model.gturbo>
  TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
  TurboFieldfareRepack --help

The installer streams a Gemma 4 checkpoint from Hugging Face and repackages
it without materializing the source checkpoint on disk. By default the
installer uses the curated, pinned source. Use --repo-id and optionally
--revision to target an alternative Hugging Face repo (for example the
QAT variant). Set HF_TOKEN only if Hugging Face requests authentication. A
cancelled or interrupted download can be continued with --resume or removed
with --discard-partial.
"""

private struct Arguments {
    var output: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var inputGTurbo: String?
    /// Optional override to fetch from a different Hugging Face repo (owner/repo)
    var repoID: String?
    /// Optional revision (branch, tag, or commit). If omitted and repoID is set, defaults to "main".
    var revision: String?

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--output", "--input-gturbo", "--repo-id", "--revision":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                let value = values[index + 1]
                switch flag {
                case "--output": parsed.output = value
                case "--input-gturbo": parsed.inputGTurbo = value
                case "--repo-id": parsed.repoID = value
                case "--revision": parsed.revision = value
                default: break
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.verifyInstall else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume else {
                throw ParseError.invalidMode("verification accepts only --input-gturbo")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode("--input-gturbo requires --verify-install")
            }
        }
        return parsed
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
    } catch ParseError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }

    // If the user provided a custom Hugging Face repo or revision, use that.
    // This enables installing alternative checkpoints such as QAT variants.
    let options: RemoteStreamingRepackOptions
    if let repoID = arguments.repoID {
        let revision = arguments.revision ?? "main"
        options = RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: URL(fileURLWithPath: output).path,
            token: ProcessInfo.processInfo.environment["HF_TOKEN"],
            requireKnownSource: false,
            minFreeReserveBytes: SupportedModelSource.reserveBytes,
            overwrite: arguments.overwrite,
            resume: arguments.resume)
    } else {
        options = SupportedModelSource.installOptions(
            outputDirectory: URL(fileURLWithPath: output),
            overwrite: arguments.overwrite,
            token: ProcessInfo.processInfo.environment["HF_TOKEN"],
            resume: arguments.resume)
    }

    do {
        let result = try await RemoteStreamingRepacker(options: options).run()
        let installedName = arguments.repoID == nil ? SupportedModelSource.displayName : "Custom: \(options.repoID)@\(options.revision)"
        print("Installed \(installedName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
