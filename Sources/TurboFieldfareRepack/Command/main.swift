import Foundation
import TurboFieldfareRepackCore

private let usage = """
Usage:
  TurboFieldfareRepack --output <model.gturbo> [--overwrite] [--resume]
  TurboFieldfareRepack --discard-partial --output <model.gturbo>
  TurboFieldfareRepack --install-vision --output <model.gturbo> [--overwrite] [--resume]
  TurboFieldfareRepack --discard-vision-partial --output <model.gturbo>
  TurboFieldfareRepack --verify-vision --input-gturbo <model.gturbo>
  TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
  TurboFieldfareRepack --help

The installer streams the supported Gemma 4 checkpoint from Hugging Face and
repackages it without materializing the source checkpoint on disk. Set HF_TOKEN
only if Hugging Face requests authentication. A cancelled or interrupted
download can be continued with --resume or removed with --discard-partial.
Vision weights are an opt-in sidecar for a completed model. Install them with
--install-vision; their resume and discard state is independent of the model.
"""

private struct Arguments {
    var output: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var installVision = false
    var discardVisionPartial = false
    var verifyVision = false
    var verifyInstall = false
    var inputGTurbo: String?

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
            case "--install-vision":
                parsed.installVision = true
                index += 1
            case "--discard-vision-partial":
                parsed.discardVisionPartial = true
                index += 1
            case "--verify-vision":
                parsed.verifyVision = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--output", "--input-gturbo":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                if flag == "--output" {
                    parsed.output = values[index + 1]
                } else {
                    parsed.inputGTurbo = values[index + 1]
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        let explicitModes = [
            parsed.discardPartial,
            parsed.installVision,
            parsed.discardVisionPartial,
            parsed.verifyVision,
            parsed.verifyInstall,
        ].filter { $0 }.count
        guard explicitModes <= 1 else {
            throw ParseError.invalidMode("installer actions are mutually exclusive")
        }
        guard !(parsed.resume && (parsed.discardPartial || parsed.discardVisionPartial)) else {
            throw ParseError.invalidMode("--resume and discard actions are mutually exclusive")
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
        if parsed.discardVisionPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.resume else {
                throw ParseError.invalidMode(
                    "--discard-vision-partial only accepts --output")
            }
            return parsed
        }
        if parsed.installVision {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode(
                    "--install-vision accepts --output, not --input-gturbo")
            }
            return parsed
        }
        if parsed.verifyVision {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume else {
                throw ParseError.invalidMode(
                    "--verify-vision accepts only --input-gturbo")
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

    if arguments.discardVisionPartial, let output = arguments.output {
        do {
            try RemoteVisionSidecarInstaller.discardPartial(modelDirectory: output)
            print("Discarded saved vision download for \(output)")
            return 0
        } catch {
            printError("vision discard failed: \(error)")
            return 1
        }
    }

    if arguments.verifyVision, let input = arguments.inputGTurbo {
        do {
            let result = try VisionSidecarVerifier.run(modelDirectory: input)
            print("Verified vision weights (\(result.weightsBytesVerified) bytes)")
            print("Entries: \(result.entryCount)")
            print("Manifest: \(result.manifestPath)")
            return 0
        } catch {
            printError("vision verification failed: \(error)")
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
    if arguments.installVision {
        let options = SupportedModelSource.visionSidecarOptions(
            modelDirectory: URL(fileURLWithPath: output),
            overwrite: arguments.overwrite,
            token: ProcessInfo.processInfo.environment["HF_TOKEN"],
            resume: arguments.resume)
        do {
            let result = try await RemoteVisionSidecarInstaller(options: options).run()
            print("Installed vision sidecar for \(SupportedModelSource.displayName)")
            print("Source revision: \(result.resolvedCommit)")
            print("Vision: \(result.visionDirectory)")
            return 0
        } catch {
            printError("vision install failed: \(error)")
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
