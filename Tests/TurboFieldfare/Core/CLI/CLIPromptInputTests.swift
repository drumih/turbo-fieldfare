import Testing
import Foundation
import Metal
import TurboFieldfare
@testable import TurboFieldfareCLICore

/// The messages-file image path and the pre-load size check. Both were review
/// findings, and neither had a test: the path resolution was dead code that
/// looked live, and the size check is only worth anything if it runs before the
/// model load.
@Suite struct CLIPromptInputTests {
    private func makeMessagesFile(
        _ json: String, in directory: URL
    ) throws -> URL {
        let url = directory.appendingPathComponent("messages.json")
        try Data(json.utf8).write(to: url)
        return url
    }

    /// A relative `image_file` belongs to the messages file's directory. The old
    /// code branched on `URL(fileURLWithPath:).path`, which resolves against the
    /// *process* directory and is therefore always absolute — so the relative
    /// branch never ran and every relative path silently resolved against
    /// wherever the CLI happened to be started.
    @Test func aRelativeImagePathResolvesAgainstTheMessagesFile() throws {
        let manager = FileManager.default
        let root = manager.temporaryDirectory
            .appendingPathComponent("cli-messages-\(UUID().uuidString)", isDirectory: true)
        let nested = root.appendingPathComponent("pictures", isDirectory: true)
        try manager.createDirectory(at: nested, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: root) }

        let absolute = root.appendingPathComponent("absolute.png")
        let messages = try makeMessagesFile("""
        [{"role":"user","content":[
          {"type":"image_file","path":"pictures/relative.png"},
          {"type":"image_file","path":"\(absolute.path)"},
          {"type":"text","text":"what is this?"}]}]
        """, in: root)

        var args = try Args.parse(["--model", "m.gturbo", "--messages-file", messages.path])
        args.messagesFile = messages.path
        let input = try parseInput(args: args)
        guard case .multimodal(_, let images) = input else {
            Issue.record("expected a multimodal input, got \(input)")
            return
        }

        let paths = Set(images.values.map(\.standardizedFileURL.path))
        #expect(paths.contains(nested.appendingPathComponent("relative.png")
            .standardizedFileURL.path),
                "a relative image path did not resolve against the messages file: \(paths)")
        #expect(paths.contains(absolute.standardizedFileURL.path),
                "an absolute image path was rewritten: \(paths)")
        // Nothing may resolve against the process directory.
        let cwd = manager.currentDirectoryPath
        #expect(!paths.contains { $0.hasPrefix(cwd + "/relative") },
                "an image resolved against the working directory")
    }

    /// A path that merely *contains* a slash is still relative. This is the case
    /// the old `hasPrefix` on a `URL.path` could never distinguish.
    @Test func aDotSlashPathIsTreatedAsRelative() throws {
        let manager = FileManager.default
        let root = manager.temporaryDirectory
            .appendingPathComponent("cli-messages-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: root) }
        let messages = try makeMessagesFile("""
        [{"role":"user","content":[{"type":"image_file","path":"./a.png"}]}]
        """, in: root)

        var args = try Args.parse(["--model", "m.gturbo", "--messages-file", messages.path])
        args.messagesFile = messages.path
        guard case .multimodal(_, let images) = try parseInput(args: args) else {
            Issue.record("expected a multimodal input")
            return
        }
        let resolved = try #require(images.values.first).standardizedFileURL.path
        #expect(resolved == root.appendingPathComponent("a.png").standardizedFileURL.path,
                "./a.png resolved to \(resolved)")
    }

    /// The size half of the prompt check, which the image path calls before the
    /// model load. An oversized image used to cost a full load and a GPU encode
    /// before anything refused it.
    @Test func anOversizedPromptIsRefusedByTheSizeCheckAlone() throws {
        let pipe = Pipe()
        var args = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        args.maxContext = 512
        args.maxNew = 64

        let fits = validatePromptSize(
            tokens: 400, args: args, stderr: pipe.fileHandleForWriting)
        #expect(fits == nil, "a prompt inside the context was refused")

        let overflows = validatePromptSize(
            tokens: 449, args: args, stderr: pipe.fileHandleForWriting)
        #expect(overflows?.exitCode == 2,
                "prompt 449 + maxNew 64 exceeds maxContext 512 and was accepted")

        // The boundary itself fits: 448 + 64 == 512.
        #expect(validatePromptSize(
            tokens: 448, args: args, stderr: pipe.fileHandleForWriting) == nil,
                "an exactly-fitting prompt was refused")

        pipe.fileHandleForWriting.closeFile()
        let message = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(),
                             as: UTF8.self)
        #expect(message.contains("context overflow"),
                "the refusal did not say why: \(message)")
    }

    /// The CLI must reach the same residency default as the runtime and the
    /// conversation, which is why there is one constant rather than four
    /// literals — one of the four had drifted to `keepReady`.
    /// `--image` with no prompt flag used to be reported as a conflict with
    /// `--messages-file`, which the user had not passed. The missing flag is
    /// `--chat-prompt`, and `--help` already says so.
    @Test func aBareImageNamesTheMissingChatPromptRatherThanAConflict() {
        #expect(throws: ArgsError.requiredMissing("--chat-prompt")) {
            _ = try Args.parse(["--model", "m.gturbo", "--image", "/tmp/a.png"])
        }
        // The conflict message is still right when the conflict is real.
        #expect(throws: ArgsError.mutuallyExclusive("--image", "--messages-file")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--image", "/tmp/a.png",
                "--messages-file", "/tmp/m.json",
            ])
        }
    }

    /// The slot pre-flight reads `args.images`, which only `--image` fills, so
    /// an image arriving through `--messages-file` used to survive validation
    /// and fail inside the routed tile scheduler - after the model load and
    /// after every image had been encoded on the GPU.
    @Test func aMessagesFileImageIsRefusedBeforeTheModelLoadToo() throws {
        let manager = FileManager.default
        let directory = manager.temporaryDirectory
            .appendingPathComponent("slots-preflight-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: directory) }
        let image = directory.appendingPathComponent("shot.png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: image)
        let messages = directory.appendingPathComponent("messages.json")
        try Data("""
        [{"role":"user","content":[{"type":"image_file","path":"shot.png"},
                                   {"type":"text","text":"describe"}]}]
        """.utf8).write(to: messages)

        let arguments = try Args.parse([
            "--model", "m.gturbo",
            "--messages-file", messages.path,
            "--expert-cache-slots", "8",
            "--prefill", "off",
        ])
        let input = try parseInput(args: arguments)
        #expect(input.hasImages)
        let need = RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        #expect(throws: ArgsError.imagePromptNeedsExpertCacheSlots(have: 8, need: need)) {
            _ = try arguments.resolvedRuntimeConfiguration(
                forceLogitsHead: false, imagePrompt: input.hasImages)
        }
        // The same arguments without an image stay valid.
        #expect(throws: Never.self) {
            _ = try arguments.resolvedRuntimeConfiguration(
                forceLogitsHead: false, imagePrompt: false)
        }
    }

    /// `--prefill off` with an image is silently coerced back to chunked deep in
    /// the runtime, while the docs say off disables that path. The run is
    /// correct; saying nothing about it is what made the flag look honoured.
    @Test func anImagePromptSaysWhenItOverridesTheRequestedPrefill() {
        #expect(prefillCoercionNotice(hasImages: true, config: .off)
                == "[prefill coerced to chunked: image prompts require it]")
        // Nothing to say when the request already matches what will run.
        #expect(prefillCoercionNotice(
            hasImages: true, config: .production(chunkTokens: 128)) == nil)
        // And nothing at all for a text prompt, whatever the prefill mode.
        #expect(prefillCoercionNotice(hasImages: false, config: .off) == nil)
    }

    @Test func theCLIDefaultsToTheSharedResidencyPolicy() throws {
        let args = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(args.visionResidency == VisionResidencyPolicy.defaultPolicy)
        #expect(args.visionResidency == .onDemand)
        let explicit = try Args.parse(
            ["--model", "m.gturbo", "--prompt", "hi", "--vision-residency", "keep-ready"])
        #expect(explicit.visionResidency == .keepReady,
                "the opt-in is still reachable")
    }

    @Test(.enabled(
        if: MTLCreateSystemDefaultDevice()?.supportsFamily(.apple8) == false,
        "requires an Apple7 or older Metal device"))
    func apple7ImageRunRejectsHardwareBeforeReadingTheModel() async throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-cli-model-\(UUID().uuidString).gturbo")
        let arguments = try Args.parse([
            "--model", missing.path,
            "--chat-prompt", "describe",
            "--image", missing.appendingPathComponent("missing.png").path,
        ])
        let errors = Pipe()

        let result = await run(args: arguments, stderr: errors.fileHandleForWriting)
        errors.fileHandleForWriting.closeFile()
        let message = String(
            decoding: errors.fileHandleForReading.readDataToEndOfFile(),
            as: UTF8.self)

        #expect(result.exitCode == 1)
        #expect(message.contains("M2 or newer"),
                "model IO masked the unsupported-hardware error: \(message)")
    }
}
