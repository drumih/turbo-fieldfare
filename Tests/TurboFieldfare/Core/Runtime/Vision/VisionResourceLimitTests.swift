import CoreGraphics
import UniformTypeIdentifiers
import TurboFieldfareFormat
import Foundation
import ImageIO
import Metal
import Testing
@testable import TurboFieldfare

/// Resource-limit and buffer-validity guards. Each test fails against the
/// behaviour each comment describes.
@Suite struct VisionResourceLimitTests {
    /// The per-side, pixel and decoded-bytes guards all threw the pixel-limit
    /// error, so two of the three sent the user to resize against a constraint
    /// their image already met.
    @Test func eachRejectionGuardNamesItsOwnLimit() throws {
        let manager = FileManager.default
        let directory = manager.temporaryDirectory
            .appendingPathComponent("limit-errors-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: directory) }
        func read(width: Int, height: Int, limits: VisionImageLimits) throws {
            let url = directory.appendingPathComponent(
                "img-\(width)x\(height)-\(UUID().uuidString).png")
            try Self.writeSolidPNG(width: width, height: height, to: url)
            let opened = try VisionImageSource(fileURL: url)
                .open(maximumEncodedBytes: limits.maximumEncodedBytes)
            _ = try ImageMetadataReader(limits: limits).read(opened: opened)
        }

        // 300x50 is far under the default pixel cap; only a side exceeds.
        #expect {
            try read(width: 300, height: 50,
                     limits: VisionImageLimits(maximumSourceDimension: 100))
        } throws: { error in
            guard case VisionImageError.sideTooLarge(
                let width, let height, let limit) = error else { return false }
            return width == 300 && height == 50 && limit == 100
        }

        // Both sides fit; only the pixel product exceeds.
        #expect {
            try read(width: 300, height: 50,
                     limits: VisionImageLimits(maximumSourcePixels: 10_000))
        } throws: { error in
            guard case VisionImageError.dimensionsTooLarge(
                let width, let height, let limit) = error else { return false }
            return width == 300 && height == 50 && limit == 10_000
        }

        // Sides and pixels fit; only the decoded-surface ceiling exceeds.
        let decodedBytes = 320 * 320 * 4
        #expect {
            try read(width: 320, height: 320,
                     limits: VisionImageLimits(maximumDecodedBytes: decodedBytes / 2))
        } throws: { error in
            guard case VisionImageError.decodedBytesTooLarge(
                let width, let height, let limit) = error else { return false }
            return width == 320 && height == 320 && limit == decodedBytes / 2
        }
    }

    /// The coordinate validator dereferences `contents()`, which is invalid for
    /// private storage — the natural choice for a caller building buffers only
    /// GPU kernels consume. The validation added to prevent GPU faults must
    /// throw, not fault first.
    @Test(.enabled(if: Self.visionModelURL != nil && Self.supportsVisionRuntime,
                   "requires an installed vision model on M2 or newer"))
    func privateStoragePositionsAreRefusedRatherThanDereferenced() throws {
        let modelURL = try #require(Self.visionModelURL)
        let context = try MetalContext()
        let config = VisionConfig()
        let runtime = try VisionRuntime.open(
            textModelURL: modelURL, context: context, environment: [:])

        let rows = config.poolingKernel * config.poolingKernel
        let patches = try #require(context.device.makeBuffer(
            length: rows * config.patchDimension * 2, options: .storageModeShared))
        let positions = try #require(context.device.makeBuffer(
            length: rows * 2 * MemoryLayout<Int32>.stride,
            options: .storageModePrivate))
        #expect {
            _ = try runtime.encodePatches(
                patchesBF16: patches,
                positionsInt32x2: positions,
                patchGridWidth: config.poolingKernel,
                patchGridHeight: config.poolingKernel)
        } throws: { error in
            guard case VisionRuntimeError.invalidInput(let detail) = error else {
                return false
            }
            return detail.contains("CPU-visible")
        }
    }

    /// The use lease opens its lock with `O_CREAT`, so pointing the runtime at
    /// a directory that merely exists — a `.partial` download, a mistyped path —
    /// left a zero-byte `.<name>.use.lock` there that nothing ever removes. The
    /// existence check in front of it had already been added for exactly this
    /// reason; the validity check still ran after the lease.
    @Test(.enabled(if: Self.visionModelURL != nil && Self.supportsVisionRuntime,
                   "requires an installed vision model on M2 or newer"))
    func aDirectoryThatIsNotAPackLeavesNoLockFileBehind() throws {
        let modelURL = try #require(Self.visionModelURL)
        let manager = FileManager.default
        let parent = manager.temporaryDirectory
            .appendingPathComponent("not-a-pack-\(UUID().uuidString)", isDirectory: true)
        let candidate = parent.appendingPathComponent(
            "model.vision.gturbo", isDirectory: true)
        try manager.createDirectory(at: candidate, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: parent) }
        // Pack-shaped enough to reach the open, empty enough to be refused.
        try Data("not a manifest".utf8).write(
            to: candidate.appendingPathComponent("vision_weights.bin"))

        let context = try MetalContext()
        #expect(throws: (any Error).self) {
            _ = try VisionRuntime.open(
                textModelURL: modelURL, context: context,
                visionPackURL: candidate, environment: [:])
        }

        let lock = parent.appendingPathComponent(".model.vision.gturbo.use.lock")
        #expect(!manager.fileExists(atPath: lock.path),
                "a refused open left a lock file with no owner at \(lock.path)")
        #expect(try manager.contentsOfDirectory(atPath: parent.path).sorted()
                == ["model.vision.gturbo"])
    }

    /// One `VisionRuntime` init used to compile the identical tensorops source
    /// three times. Libraries are immutable, so the same module for the same
    /// device must come back as the same object.
    @Test func privateLibraryIsCompiledOncePerDeviceAndModule() throws {
        let context = try MetalContext()
        let first = try MetalContext.privateLibrary(
            device: context.device, module: "tensorops")
        let second = try MetalContext.privateLibrary(
            device: context.device, module: "tensorops")
        #expect(first === second, "the same module was compiled twice for one device")
    }

    @Test func imageTowerFailsClosedBeforeApple8() throws {
        #expect {
            try VisionRuntime.requireSupportedDevice(supportsApple8: false)
        } throws: { error in
            guard case VisionRuntimeError.unsupportedKernel(let detail) = error else {
                return false
            }
            return detail.contains("M2 or newer")
        }
        try VisionRuntime.requireSupportedDevice(supportsApple8: true)
    }

    @Test(.enabled(if: !Self.supportsVisionRuntime,
                   "requires an Apple7 or older Metal device"))
    func unsupportedDeviceIsRejectedBeforeModelIO() throws {
        let context = try MetalContext()
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-model-\(UUID().uuidString)")
        #expect {
            _ = try VisionRuntime.open(textModelURL: missing, context: context)
        } throws: { error in
            guard case VisionRuntimeError.unsupportedKernel(let detail) = error else {
                return false
            }
            return detail.contains("M2 or newer")
        }
    }

    @Test func defaultKernelSelectionKeepsPromotedAndBaselinePathsDistinct() {
        let promoted = VisionRuntime.resolvedKernelEnvironment([:])
        #expect(promoted["TURBO_FIELDFARE_VISION_ATTENTION_MPP"] == "1")
        #expect(promoted["TURBO_FIELDFARE_VISION_REGISTER_GEMM"] == "1")

        let baseline = VisionRuntime.resolvedKernelEnvironment(
            ["TURBO_FIELDFARE_VISION_BASELINE_KERNELS": "1"])
        #expect(baseline["TURBO_FIELDFARE_VISION_ATTENTION_MPP"] == nil)
        #expect(baseline["TURBO_FIELDFARE_VISION_REGISTER_GEMM"] == nil)

        let explicit = VisionRuntime.resolvedKernelEnvironment([
            "TURBO_FIELDFARE_VISION_ATTENTION_Q8": "1",
            "TURBO_FIELDFARE_VISION_REGISTER_ATTENTION": "1",
        ])
        #expect(explicit["TURBO_FIELDFARE_VISION_ATTENTION_MPP"] == nil)
        #expect(explicit["TURBO_FIELDFARE_VISION_REGISTER_GEMM"] == nil)
    }

    private static func writeSolidPNG(width: Int, height: Int, to url: URL) throws {
        let colorSpace = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let bitmap = try #require(CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        bitmap.setFillColor(CGColor(gray: 0.5, alpha: 1))
        bitmap.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = try #require(bitmap.makeImage())
        let destination = try #require(CGImageDestinationCreateWithURL(
            url as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
    }

    /// An installed text model with its companion pack beside it, or nil when
    /// this checkout has none — the model-dependent cases then skip.
    ///
    /// Searched by walking up from this source file rather than from the
    /// working directory. The working directory is wherever the suite happens
    /// to be run from, so a checkout nested inside another one silently skipped
    /// every case here while a model sat one level up.
    static func visionModelURL(from file: StaticString = #filePath) -> URL? {
        let manager = FileManager.default
        var directory = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
        while directory.path != "/" {
            let candidate = directory
                .appendingPathComponent("scratch")
                .appendingPathComponent("gemma4.gturbo")
            if let companion = try? VisionPackLocation.companionURL(
                forTextModel: candidate),
               manager.fileExists(atPath: candidate.path),
               manager.fileExists(atPath: companion.path) {
                return candidate
            }
            directory = directory.deletingLastPathComponent()
        }
        return nil
    }

    static var visionModelURL: URL? { visionModelURL() }

    static var supportsVisionRuntime: Bool {
        MTLCreateSystemDefaultDevice()?.supportsFamily(.apple8) == true
    }

    @Test func osSidecarsDoNotInvalidateAPack() {
        for name in [".DS_Store", "._manifest.json", "._vision_weights.bin",
                     ".Spotlight-V100", ".fseventsd", "manifest.json.icloud"] {
            #expect(GTurboVisionFormatV1.isSidecarEntry(name),
                    "\(name) would still be treated as part of the pack")
        }
        // The other direction is the one that matters more: `VisionWeightStore`
        // and the pack verifier both drop every entry this matches before
        // comparing the directory against the pack, so a rule broad enough to
        // swallow a required file reports the same "image support is
        // unavailable" as the sidecar it was widened for — for a pack that is
        // complete on disk. The names are spelled out as well as read off the
        // constants, so renaming a constant cannot quietly empty this check.
        let packEntries = ["manifest.json", "vision_weights.bin",
                           "processor_config.json", "verified-install.json"]
        #expect(packEntries == [GTurboVisionFormatV1.manifestFile,
                                GTurboVisionFormatV1.weightsFile,
                                GTurboVisionFormatV1.processorFile,
                                GTurboVisionFormatV1.receiptFile],
                "the pack's entry names changed; this list has to change with them")
        // `_manifest.json` is the near miss for the AppleDouble prefix: a real
        // file, one character from the rule that ignores one.
        for name in packEntries + ["vision_weights.bin.tmp", "_manifest.json"] {
            #expect(!GTurboVisionFormatV1.isSidecarEntry(name),
                    "\(name) would be ignored, which is how a real file goes missing")
        }
    }

    /// A deleted pack file used to be reported as the surviving files being
    /// "unexpected", so the message pointed at three files that belong there
    /// and never named the one that had gone. Both the runtime store and the
    /// installer's verifier phrase it through this helper, so both are covered.
    @Test func aMissingPackFileIsReportedAsMissingRatherThanAsSurplus() {
        let expected: Set<String> = [
            GTurboVisionFormatV1.manifestFile,
            GTurboVisionFormatV1.weightsFile,
            GTurboVisionFormatV1.processorFile,
            GTurboVisionFormatV1.receiptFile,
        ]

        let withoutProcessor = expected.subtracting([GTurboVisionFormatV1.processorFile])
        let missing = GTurboVisionFormatV1.entryDifference(
            expected: expected, actual: withoutProcessor)
        #expect(missing == "missing [\"processor_config.json\"]")
        #expect(!missing.contains("unexpected"),
                "a deleted file must not be described as surplus entries")

        let withStray = expected.union(["stray.bin"])
        #expect(GTurboVisionFormatV1.entryDifference(
            expected: expected, actual: withStray) == "unexpected [\"stray.bin\"]")

        // Both at once, which is what a half-written pack looks like.
        let both = withoutProcessor.union(["stray.bin"])
        #expect(GTurboVisionFormatV1.entryDifference(expected: expected, actual: both)
                == "missing [\"processor_config.json\"], unexpected [\"stray.bin\"]")
    }

    /// The decoded-byte ceiling is inclusive. This catches a `<=` quietly
    /// changing to `<`, which would reject an image exactly at the contract.
    @Test func theDecodedCeilingAdmitsExactlyItselfAndRefusesOneByteMore() throws {
        let side = 320
        let decodedBytes = side * side * 4
        let manager = FileManager.default
        let directory = manager.temporaryDirectory
            .appendingPathComponent("decode-boundary-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: directory) }
        let url = directory.appendingPathComponent("square.png")
        try Self.writeSolidPNG(width: side, height: side, to: url)

        func read(ceiling: Int) throws {
            let limits = VisionImageLimits(maximumDecodedBytes: ceiling)
            let opened = try VisionImageSource(fileURL: url)
                .open(maximumEncodedBytes: limits.maximumEncodedBytes)
            _ = try ImageMetadataReader(limits: limits).read(opened: opened)
        }

        // Exactly at the ceiling: admitted.
        try read(ceiling: decodedBytes)
        // One byte under it: refused, and by the decoded-bytes guard rather
        // than a dimension guard, which is what the user is told to act on.
        #expect {
            try read(ceiling: decodedBytes - 1)
        } throws: { error in
            guard case VisionImageError.decodedBytesTooLarge(_, _, let limit) = error else {
                return false
            }
            return limit == decodedBytes - 1
        }
    }

    @Test func oversizedSourcesAreRefusedWhateverTheFormat() throws {
        let side = 320
        let decodedBytes = side * side * 4
        // Injected rather than shipped so the light suite does not allocate a
        // production-size surface merely to prove the format-independent gate.
        let limits = VisionImageLimits(maximumDecodedBytes: decodedBytes / 2)

        let manager = FileManager.default
        let directory = manager.temporaryDirectory
            .appendingPathComponent("decode-cap-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: directory) }

        let reader = ImageMetadataReader(limits: limits)
        for (type, ext) in [(UTType.png, "png"), (UTType.jpeg, "jpg")] {
            let url = directory.appendingPathComponent("big.\(ext)")
            try Self.writeSolidImage(width: side, height: side, type: type, to: url)
            let opened = try VisionImageSource(fileURL: url)
                .open(maximumEncodedBytes: limits.maximumEncodedBytes)
            #expect(throws: VisionImageError.self) { _ = try reader.read(opened: opened) }
        }

        // The shipped ceiling must admit the largest supported 8-bit camera
        // input. This fails if the earlier 96 MiB low-end policy returns.
        let shipped = VisionImageLimits()
        let pixels = 8_000 * 6_000
        #expect(pixels <= shipped.maximumSourcePixels)
        #expect(8_000 <= shipped.maximumSourceDimension)
        #expect(pixels * 4 <= shipped.maximumDecodedBytes)
    }

    private static func writeSolidImage(
        width: Int, height: Int, type: UTType, to url: URL
    ) throws {
        let colorSpace = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let bitmap = try #require(CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        bitmap.setFillColor(CGColor(gray: 0.5, alpha: 1))
        bitmap.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = try #require(bitmap.makeImage())
        let destination = try #require(CGImageDestinationCreateWithURL(
            url as CFURL, type.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
    }

}
