import Foundation
import TurboFieldfareRepackCore

private let usage = """
Usage:
  TurboFieldfareRepack --output <model.gturbo> [--overwrite] [--resume]
  TurboFieldfareRepack --output <model.gturbo> --model-family kimi-k3 [--trunk-quant int4|int8]
  TurboFieldfareRepack --output <existing-k3.gturbo> --model-family kimi-k3 \\
    --trunk-quant int8 --reuse-existing-experts [--resume]
  TurboFieldfareRepack --discard-partial --output <model.gturbo>
  TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
  TurboFieldfareRepack --help

The installer streams the supported Gemma 4 checkpoint from Hugging Face and
repackages it without materializing the source checkpoint on disk. Set HF_TOKEN
only if Hugging Face requests authentication. A cancelled or interrupted
download can be continued with --resume or removed with --discard-partial.

--model-family kimi-k3 installs the pinned moonshotai/Kimi-K3 checkpoint
instead, quantizing the BF16 trunk to affine-g64 (int4 default, or int8) and
copying the MXFP4 routed experts verbatim.

--reuse-existing-experts upgrades an existing verified K3 bundle in place:
its expert files are APFS-cloned into a sibling partial bundle, only the trunk
source ranges are downloaded, and the completed bundle is atomically swapped.
"""

private struct Arguments {
    var output: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var inputGTurbo: String?
    var modelFamily: String?
    var trunkQuant: String?
    var reuseExistingExperts = false

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
            case "--reuse-existing-experts":
                parsed.reuseExistingExperts = true
                index += 1
            case "--output", "--input-gturbo", "--model-family", "--trunk-quant":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                if flag == "--output" {
                    parsed.output = values[index + 1]
                } else if flag == "--input-gturbo" {
                    parsed.inputGTurbo = values[index + 1]
                } else if flag == "--model-family" {
                    parsed.modelFamily = values[index + 1]
                } else {
                    parsed.trunkQuant = values[index + 1]
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        if let family = parsed.modelFamily,
           family != "gemma" && family != "kimi-k3" {
            throw ParseError.invalidMode("unknown model family: \(family)")
        }
        if parsed.trunkQuant != nil, parsed.modelFamily != "kimi-k3" {
            throw ParseError.invalidMode("--trunk-quant requires --model-family kimi-k3")
        }
        if let quant = parsed.trunkQuant, K3TrunkQuant(rawValue: quant) == nil {
            throw ParseError.invalidMode("unknown trunk quantization: \(quant)")
        }
        if parsed.reuseExistingExperts {
            guard parsed.modelFamily == "kimi-k3" else {
                throw ParseError.invalidMode(
                    "--reuse-existing-experts requires --model-family kimi-k3")
            }
            guard parsed.trunkQuant != nil else {
                throw ParseError.invalidMode(
                    "--reuse-existing-experts requires an explicit --trunk-quant")
            }
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.verifyInstall,
                  !parsed.reuseExistingExperts else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume,
                  !parsed.reuseExistingExperts else {
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
    if arguments.modelFamily == "kimi-k3" {
        let trunkQuant = arguments.trunkQuant.flatMap(K3TrunkQuant.init(rawValue:)) ?? .int4
        let options = K3SupportedModelSource.installOptions(
            outputDirectory: URL(fileURLWithPath: output),
            overwrite: arguments.overwrite || arguments.reuseExistingExperts,
            token: ProcessInfo.processInfo.environment["HF_TOKEN"],
            resume: arguments.resume,
            trunkQuant: trunkQuant)
        let effectiveOptions = arguments.reuseExistingExperts
            ? K3RemoteStreamingRepackOptions(
                repoID: options.repoID,
                revision: options.revision,
                outputDir: options.outputDir,
                token: options.token,
                requireKnownSource: options.requireKnownSource,
                trunkQuant: options.trunkQuant,
                copyAuditPath: options.copyAuditPath,
                rangeChunkBytes: options.rangeChunkBytes,
                writeTileBytes: options.writeTileBytes,
                minFreeReserveBytes: options.minFreeReserveBytes,
                overwrite: true,
                resume: options.resume,
                reuseExpertsFrom: output,
                dryRunSpaceCheck: options.dryRunSpaceCheck,
                downloadSession: options.downloadSession,
                baseURL: options.baseURL,
                rangeRetryAttempts: options.rangeRetryAttempts,
                retryBaseDelayNs: options.retryBaseDelayNs)
            : options
        do {
            let result = try await K3RemoteStreamingRepacker(options: effectiveOptions).run()
            print("Installed \(K3SupportedModelSource.displayName)")
            print("Source revision: \(result.resolvedCommit)")
            print("Model: \(result.outputDir)")
            return 0
        } catch {
            printError("install failed: \(error)")
            return 1
        }
    }
    let options = SupportedModelSource.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume)
    do {
        let result = try await RemoteStreamingRepacker(options: options).run()
        print("Installed \(SupportedModelSource.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
