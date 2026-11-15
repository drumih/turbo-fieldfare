//
//  Model+Loading.swift
//  TinyTitan
//
//  Opening a `.ssdai` install: the hashing/verification pipeline that turns a
//  directory into a typed `Model`, split out of `Model.swift` so the model's
//  shape and the code that loads it are separate reads.
//

import Darwin
import Foundation
import Metal
import TinyTitanFormat
import TinyTitanCore

extension Model {

    /// Open a `.ssdai/` directory and return a typed handle. Eagerly verifies
    /// SHA-256 of `model_weights.bin` and `packed_experts/layout.json`; layer
    /// files are verified lazily on first `routedExpert(...)` touch.
    /// lint:allow-long a sequential load pipeline -- open, hash, verify the
    /// receipt, decode the layout, map the resident buffer -- whose stages
    /// share a descriptor, sizes and timing stats. Extracting any of them
    /// needs six or seven parameters, trading one readable sequence for
    /// several functions with unwieldy signatures.
    public static func load(
        directoryURL: URL,
        device: MTLDevice,
        expecting: ArchConfig = .qwen36_35B_A3B,
        streamingMode: ExpertStreamingMode = .pread(slotCount: 32),
        expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
        expertIOTimeline: RuntimeExpertIOTimeline = .shared,
        integrityPolicy: ModelIntegrityPolicy? = nil,
        loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil
    ) throws -> Model {
        var stats = ModelLoadStats()
        defer {
            loadStats?.pointee = stats
        }
        let resolvedIntegrityPolicy = integrityPolicy ?? .fullSha256

        // -- create the directory handle and open manifest
        let modelDirectory = try SSDAIModelDirectory(rootURL: directoryURL)
        let manifestFD: Int32
        do {
            manifestFD = try modelDirectory.openFile("manifest.json")
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        defer { close(manifestFD) }

        // -- read manifest data and compute hash from the in-memory buffer
        let manifestData = try modelDirectory.readMetadata(
            fileDescriptor: manifestFD,
            relativePath: "manifest.json",
            maxBytes: ManifestReader.defaultMaxBytes)
        let manifestSize = UInt64(manifestData.count)
        let manifestShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let manifestSha = Sha256Verifier.hashData(manifestData)
        stats.manifestSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - manifestShaStart

        // -- optional trusted-receipt validation
        let receipt: VerifiedInstallReceipt?
        var trustedReceiptUsable = false
        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            do {
                let receiptFD = try modelDirectory.openFile(
                    VerifiedInstallReceiptReader.fileName)
                defer { close(receiptFD) }
                let receiptData = try modelDirectory.readMetadata(
                    fileDescriptor: receiptFD,
                    relativePath: VerifiedInstallReceiptReader.fileName,
                    maxBytes: VerifiedInstallReceiptReader.defaultMaxBytes)
                let loadedReceipt = try JSONDecoder().decode(
                    VerifiedInstallReceipt.self, from: receiptData)
                try VerifiedInstallReceiptReader.validateManifestBinding(
                    loadedReceipt,
                    directoryURL: directoryURL,
                    manifestSha256: manifestSha)
                stats.receiptValidationNanos &+=
                    clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
                receipt = loadedReceipt
                trustedReceiptUsable = true
            } catch {
                // The trusted-receipt policy is strict: a missing or invalid
                // receipt is a hard error, because silently falling back to a
                // full re-hash would mask tampering or a moved directory and
                // defeat the policy's purpose.
                if let receiptError = error as? ModelError,
                    case .trustedReceiptInvalid = receiptError
                {
                    throw receiptError
                }
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(VerifiedInstallReceiptReader.fileName): \(error)")
            }
        } else {
            receipt = nil
        }

        let manifest = try ManifestReader.decode(
            data: manifestData, expecting: expecting)
        if let receipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try VerifiedInstallReceiptReader.validate(
                receipt,
                directoryURL: directoryURL,
                manifest: manifest,
                manifestSha256: manifestSha,
                manifestSize: manifestSize)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        // -- verify the small, always-touched files before mapping model data
        let weightsURL = directoryURL.appendingPathComponent("model_weights.bin")
        guard let weightsEntry = manifest.files["model_weights.bin"] else {
            throw ModelError.missingFile(name: "model_weights.bin")
        }
        guard let layoutEntry = manifest.files["packed_experts/layout.json"] else {
            throw ModelError.missingFile(name: "packed_experts/layout.json")
        }

        let weightsFD = try modelDirectory.openFile("model_weights.bin")
        defer { close(weightsFD) }
        let layoutFD = try modelDirectory.openFile("packed_experts/layout.json")
        defer { close(layoutFD) }

        // Read layout.json and validate size via modelDirectory
        let layoutData = try modelDirectory.readMetadata(
            fileDescriptor: layoutFD,
            relativePath: "packed_experts/layout.json",
            maxBytes: PackedExpertsLayoutReader.defaultMaxBytes)
        guard UInt64(layoutData.count) == layoutEntry.size else {
            throw ModelError.tensorSizeMismatch(
                name: "packed_experts/layout.json",
                expected: layoutEntry.size,
                actual: UInt64(layoutData.count))
        }

        // Validate weights file size via modelDirectory
        let weightsSize = try modelDirectory.fileSize(
            fileDescriptor: weightsFD, relativePath: "model_weights.bin")
        guard weightsSize == weightsEntry.size else {
            throw ModelError.tensorSizeMismatch(
                name: "model_weights.bin",
                expected: weightsEntry.size,
                actual: weightsSize)
        }

        // SHA-256: weights via FD, layout via in-memory data. Under a usable
        // trusted-receipt policy the installer already pinned these hashes at
        // install time, so re-hashing the full weights file is skipped; the
        // payload is instead warmed with F_RDADVISE so GPU first-touch does
        // not fault on cold pages. A receipt that failed to validate falls
        // back to the full hash here.
        if resolvedIntegrityPolicy == .fullSha256 || !trustedReceiptUsable {
            let eagerShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try Sha256Verifier.verifyFile(
                fileDescriptor: weightsFD,
                named: "model_weights.bin",
                expectedHex: weightsEntry.sha256)
            guard
                Sha256Verifier.hashData(layoutData).lowercased()
                    == layoutEntry.sha256.lowercased()
            else {
                throw ModelError.checksumMismatch(file: "packed_experts/layout.json")
            }
            stats.eagerSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - eagerShaStart
        } else {
            _ = RDAdvice.call(fd: weightsFD, offset: 0, byteCount: weightsSize)
        }

        // -- decode layout from TinyTitanFormat wire codec
        let layout = try PackedExpertsLayoutReader.decode(
            data: layoutData,
            manifest: manifest)
        if trustedReceiptUsable {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try validateTrustedReceiptLayerLayout(
                modelDirectory: modelDirectory,
                manifest: manifest,
                layout: layout)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        // -- load resident index using the FD passed from openFile()
        let residentIndex = try ResidentIndexReader.load(
            fileDescriptor: weightsFD, displayPath: "model_weights.bin")
        try validateRuntimeSchema(
            residentIndex: residentIndex,
            layout: layout,
            manifest: manifest,
            config: expecting)

        // The resident index must account for the complete weights file.
        let fileSize = weightsSize
        let (expectedSize, overflow) = residentIndex.header.indexSize
            .addingReportingOverflow(residentIndex.header.residentSize)
        if overflow || fileSize != expectedSize {
            throw ModelError.indexCorrupt(
                detail: """
                    model_weights.bin size \(fileSize) != indexSize \
                    \(residentIndex.header.indexSize) + residentSize \
                    \(residentIndex.header.residentSize) = \(expectedSize)
                    """)
        }

        // -- create resident buffer, reusing the opened FD
        let residentBuffer = try ResidentBuffer(
            fileURL: weightsURL,
            fileOffset: residentIndex.header.indexSize,
            residentSize: residentIndex.header.residentSize,
            device: device,
            fileDescriptor: weightsFD)

        // A checkpoint may keep the small per-head tensors the GDN kernels read
        // as bf16 in fp32 instead (the dense Qwen 3.5 installs do, following
        // their source checkpoints); promote those once here.
        let promotedBF16 = try Model.buildBF16ReadableViews(
            device: device,
            schema: TensorSchema.schema(for: expecting.modelFamily),
            config: expecting,
            residentIndex: residentIndex,
            residentBuffer: residentBuffer.buffer)

        return Model(
            device: device,
            config: expecting,
            streamingMode: streamingMode,
            expertCachePolicy: expertCachePolicy,
            expertIOTimeline: expertIOTimeline,
            integrityPolicy: resolvedIntegrityPolicy,
            residentBuffer: residentBuffer,
            residentIndex: residentIndex,
            packedExpertsLayout: layout,
            manifest: manifest,
            directoryURL: directoryURL,
            modelDirectory: modelDirectory,
            promotedBF16: promotedBF16)
    }
}
