import Foundation
import TinyTitanRepackCore

private let supportedModelNames = SupportedModelSource.all.map(\.name).joined(separator: "|")

private let usage = """
    Usage:
      TinyTitanRepack [--model <\(supportedModelNames)>] --output <model.ssdai> [--overwrite] [--resume]
      TinyTitanRepack --input-snapshot <affine-safetensors-dir> --model-id <id>
                      --output <model.ssdai> [--overwrite] [--draft-head] [--share-ngram-table]
                      [--consume-snapshot]
      TinyTitanRepack --input-gguf <inkling2.gguf> --model-id <id> --output <model.ssdai>
                      [--overwrite] [--consume-snapshot]
      TinyTitanRepack --discard-partial --output <model.ssdai>
      TinyTitanRepack --verify-install --input-ssdai <model.ssdai>  (--input-gturbo still accepted)
      TinyTitanRepack --compress-experts --input-ssdai <raw model.ssdai>
                      --output <compressed model.ssdai> [--overwrite]
          Experimental. Saves ~9% of the expert bytes, but measured -42% decode
          on Ornith 4-bit (docs/perf-campaign-2026-10-18.md); for measurement
          only, not a faster install.
      TinyTitanRepack --help

    The installer streams the selected Qwen 3.6 or text-only Ornith 1.5 checkpoint
    (default: Ornith 8-bit) from Hugging Face and repackages it without materializing
    the source checkpoint on disk. Set HF_TOKEN only if Hugging Face requests
    authentication. A cancelled or interrupted download can be continued with
    --resume or removed with --discard-partial.

    --input-snapshot imports a completed local MLX-affine safetensors snapshot.
    It is intended for reproducibly derived sidecars such as Ornith's native MTP
    draft and does not support --resume because no network payload is involved.
    --draft-head imports only the draft head's tensors. --share-ngram-table
    hardlinks the snapshot's n-gram table rather than rebuilding it, which works
    only when the staging directory and the install share a filesystem. Neither
    flag means anything without --input-snapshot, and the repack refuses that
    combination rather than ignoring it. --consume-snapshot deletes each snapshot
    shard once the last range that reads it has been committed, which is what
    lets a 125 GB install grow into the space its own snapshot is releasing on a
    host that cannot hold both. It is one-way: an interrupted run leaves a
    partly consumed snapshot.

    --input-gguf converts an Inkling 2 GGUF on disk into a .ssdai install, every
    tensor kept in its GGUF encoding (routed experts NVFP4/Q8_0, the rest Q8_0,
    F32 or BF16). --consume-snapshot deletes the GGUF once it is converted.
    """

private struct Arguments {
    var model = SupportedModelSource.default
    var modelExplicit = false
    var output: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var inputSSDAI: String?
    var inputSnapshot: String?
    var inputGGUF: String?
    var compressExperts = false
    var codec: CompressExpertsCodec = .frames
    var localModelID: String?
    var draftHead = false
    var shareNgramTable = false
    var consumeSnapshot = false

    /// lint:allow-long a flag table: one `case` per option plus its
    /// validation. Splitting it would scatter the accepted grammar across
    /// helpers and make an option's parse and its refusal impossible to read
    /// together.
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
            case "--compress-experts":
                parsed.compressExperts = true
                index += 1
            case "--codec":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                guard let codec = CompressExpertsCodec(rawValue: values[index + 1]) else {
                    throw ParseError.invalidMode(
                        "unknown codec \"\(values[index + 1])\"; supported: frames, rans")
                }
                parsed.codec = codec
                index += 2
            case "--model":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                guard let source = SupportedModelSource.named(values[index + 1]) else {
                    throw ParseError.invalidMode(
                        "unknown model \"\(values[index + 1])\"; supported: "
                            + SupportedModelSource.all.map(\.name).joined(separator: ", "))
                }
                parsed.model = source
                parsed.modelExplicit = true
                index += 2
            case "--draft-head":
                parsed.draftHead = true
                index += 1
            case "--share-ngram-table":
                parsed.shareNgramTable = true
                index += 1
            case "--consume-snapshot":
                parsed.consumeSnapshot = true
                index += 1
            case "--output", "--input-ssdai", "--input-gturbo", "--input-snapshot", "--model-id",
                "--input-gguf":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                if flag == "--output" {
                    parsed.output = values[index + 1]
                } else if flag == "--input-ssdai" || flag == "--input-gturbo" {
                    parsed.inputSSDAI = values[index + 1]
                } else if flag == "--input-snapshot" {
                    parsed.inputSnapshot = values[index + 1]
                } else if flag == "--input-gguf" {
                    parsed.inputGGUF = values[index + 1]
                } else {
                    parsed.localModelID = values[index + 1]
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        // Both are fields of `LocalSnapshotRepackOptions`, which the download,
        // discard and receipt routes have no parameter for.
        guard !parsed.consumeSnapshot || parsed.inputSnapshot != nil || parsed.inputGGUF != nil
        else {
            throw ParseError.invalidMode(
                "--consume-snapshot applies only to --input-snapshot and --input-gguf imports")
        }
        if parsed.inputGGUF != nil {
            return try parsed.validatingLocalGGUF()
        }
        guard (!parsed.draftHead && !parsed.shareNgramTable) || parsed.inputSnapshot != nil else {
            throw ParseError.invalidMode(
                "--draft-head, --share-ngram-table and --consume-snapshot apply "
                    + "only to --input-snapshot imports")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputSSDAI == nil,
                parsed.inputSnapshot == nil,
                parsed.localModelID == nil,
                !parsed.modelExplicit,
                !parsed.overwrite,
                !parsed.verifyInstall
            else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.inputSnapshot != nil || parsed.localModelID != nil {
            return try parsed.validatingLocalSnapshot()
        }
        if parsed.compressExperts {
            return try parsed.validatingCompressExperts()
        }
        if parsed.verifyInstall {
            guard parsed.inputSSDAI != nil else {
                throw ParseError.missingRequired("--input-ssdai")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume else {
                throw ParseError.invalidMode("verification accepts only --input-ssdai")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputSSDAI == nil else {
                throw ParseError.invalidMode("--input-ssdai requires --verify-install")
            }
        }
        return parsed
    }

    /// A GGUF conversion names its file, model id and output, and takes only
    /// `--overwrite` and `--consume-snapshot` besides.
    private func validatingLocalGGUF() throws -> Arguments {
        guard localModelID != nil else { throw ParseError.missingRequired("--model-id") }
        guard output != nil else { throw ParseError.missingRequired("--output") }
        guard inputSnapshot == nil, inputSSDAI == nil, !modelExplicit, !resume,
            !discardPartial, !verifyInstall, !compressExperts, !draftHead, !shareNgramTable
        else {
            throw ParseError.invalidMode(
                "--input-gguf accepts only --model-id, --output, --overwrite and --consume-snapshot"
            )
        }
        return self
    }

    /// A local snapshot import names its own source and model id, and takes no
    /// flag from the download, discard or verification modes.
    private func validatingLocalSnapshot() throws -> Arguments {
        guard inputSnapshot != nil else {
            throw ParseError.missingRequired("--input-snapshot")
        }
        guard localModelID != nil else {
            throw ParseError.missingRequired("--model-id")
        }
        guard output != nil else {
            throw ParseError.missingRequired("--output")
        }
        guard !modelExplicit, inputSSDAI == nil, !resume, !discardPartial,
            !verifyInstall
        else {
            throw ParseError.invalidMode(
                "local snapshot import accepts only --input-snapshot, --model-id, --output, --draft-head, --share-ngram-table, and --overwrite"
            )
        }
        return self
    }

    /// `--compress-experts` is a standalone mode: it reads an installed model
    /// and writes another one, so every flag that describes a fresh install or a
    /// verification contradicts it.
    private func validatingCompressExperts() throws -> Arguments {
        guard inputSSDAI != nil else {
            throw ParseError.missingRequired("--input-ssdai")
        }
        guard output != nil else {
            throw ParseError.missingRequired("--output")
        }
        guard !verifyInstall, !resume, !discardPartial, !modelExplicit,
            inputSnapshot == nil, localModelID == nil, !draftHead, !shareNgramTable
        else {
            throw ParseError.invalidMode(
                "--compress-experts accepts only --input-ssdai, --output and --overwrite")
        }
        return self
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

/// `--input-gguf`: convert an Inkling 2 GGUF on disk into a `.ssdai` install.
private func runInputGGUF(
    input: String, modelID: String, output: String, arguments: Arguments,
    progressDisplay: InstallProgressDisplay
) async -> Int32 {
    do {
        let result = try await RemoteStreamingRepacker.runLocalGGUF(
            ggufPath: input, outputDir: output, modelID: modelID,
            consumeSource: arguments.consumeSnapshot, overwrite: arguments.overwrite,
            progress: { progress in progressDisplay.handle(progress) })
        print("Converted GGUF")
        print("Source fingerprint: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("GGUF conversion failed: \(error)")
        return 1
    }
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

    if arguments.compressExperts, let input = arguments.inputSSDAI,
        let output = arguments.output
    {
        do {
            let result = try CompressExpertsTool.run(
                options: CompressExpertsOptions(
                    inputSSDAI: input, output: output, overwrite: arguments.overwrite,
                    codec: arguments.codec),
                progress: { progress in
                    let ratio = Double(progress.compressedBytes) / Double(max(progress.rawBytes, 1))
                    print(
                        "[\(progress.layer + 1)/\(progress.layerCount)] \(progress.file): "
                            + "\(progress.rawBytes) -> \(progress.compressedBytes) bytes "
                            + String(format: "(%.1f%%)", ratio * 100)
                            + String(format: " %.1f s", progress.elapsedSeconds))
                })
            print(
                "Compressed \(result.layerCount) expert layers: "
                    + "\(result.rawBytes) -> \(result.compressedBytes) bytes "
                    + "(indices \(result.indexBytes) bytes)")
            print("Model: \(result.outputDirectory)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("compression failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputSSDAI {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputSSDAI: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    let progressDisplay = InstallProgressDisplay()
    if let input = arguments.inputGGUF, let modelID = arguments.localModelID,
        let output = arguments.output
    {
        return await runInputGGUF(
            input: input, modelID: modelID, output: output, arguments: arguments,
            progressDisplay: progressDisplay)
    }
    if let input = arguments.inputSnapshot,
        let modelID = arguments.localModelID,
        let output = arguments.output
    {
        do {
            let result = try await RemoteStreamingRepacker.runLocalSnapshot(
                options: LocalSnapshotRepackOptions(
                    inputSnapshotDir: input,
                    outputDir: output,
                    modelID: modelID,
                    draftHead: arguments.draftHead,
                    shareNgramTable: arguments.shareNgramTable,
                    consumeSnapshot: arguments.consumeSnapshot,
                    overwrite: arguments.overwrite),
                progress: { progress in progressDisplay.handle(progress) })
            print("Imported local snapshot")
            print("Source fingerprint: \(result.resolvedCommit)")
            print("Model: \(result.outputDir)")
            return 0
        } catch {
            printError("local import failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }
    let source = arguments.model
    let options = source.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume)
    do {
        let result = try await RemoteStreamingRepacker(options: options).run(
            progress: { progress in progressDisplay.handle(progress) })
        print("Installed \(source.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
