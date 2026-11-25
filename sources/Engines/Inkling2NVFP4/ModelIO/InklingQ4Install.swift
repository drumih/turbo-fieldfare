import Foundation
import TinyTitanCore
import TinyTitanFormat

/// A verified Inkling 2 Q4 `.ssdai` install (TT-121 phase 2): its manifest,
/// the resident index of `model_weights.bin` and the packed-expert layout,
/// opened only after the install receipt matches the manifest and the
/// directory it was issued for.
package struct InklingQ4Install {
    enum Error: Swift.Error, CustomStringConvertible, Equatable {
        case notInkling(String)
        case receipt(String)
        case layout(String)

        var description: String {
            switch self {
            case .notInkling(let detail): "not an Inkling 2 NVFP4 install: \(detail)"
            case .receipt(let detail): "install receipt: \(detail)"
            case .layout(let detail): "packed-expert layout: \(detail)"
            }
        }
    }

    /// The model identity the repacker writes for this install.
    static let modelIDPrefix = "inkling-2-mini-nvfp4"
    static let residentFile = "model_weights.bin"
    static let layoutFile = "packed_experts/layout.json"

    /// Whether `url` is a directory holding this variant's install (by its
    /// manifest identity; the full checks run when it is opened).
    package static func recognizes(_ url: URL) -> Bool {
        guard let manifest = try? SSDAIManifestReader.read(directoryURL: url) else { return false }
        return manifest.modelID.hasPrefix(modelIDPrefix)
    }

    let directory: URL
    let manifest: Manifest
    /// The checkpoint's resident index seen through the graph's tensor names
    /// (`InklingNVFP4TensorMap`); `conversions` are the roles built at load.
    let resident: ResidentIndex
    /// Raw checkpoint resident index, preserving `mtp.*` tensors (TT-129).
    let checkpointResident: ResidentIndex
    let conversions: [String: InklingNVFP4Conversion]
    /// The checkpoint's packed layout (where experts are read from) and how
    /// each layer's experts sit in a slot once repacked.
    let layout: SSDAIPackedExpertsLayoutV1
    let slotPlans: [InklingNVFP4SlotPlan]

    init(directory: URL) throws {
        self.directory = directory.standardizedFileURL
        manifest = try SSDAIManifestReader.read(directoryURL: self.directory)
        guard manifest.modelID.hasPrefix(Self.modelIDPrefix),
            manifest.arch.variantGeometry != nil
        else {
            throw Error.notInkling("model \(manifest.modelID)")
        }
        try Self.verifyReceipt(directory: self.directory, manifest: manifest)
        let residentURL = self.directory.appendingPathComponent(Self.residentFile)
        let checkpoint = try ResidentIndexReader.load(fileURL: residentURL)
        checkpointResident = checkpoint
        (resident, conversions) = try Self.graphView(
            checkpoint: checkpoint, layers: manifest.numLayers)
        let files = try SSDAIModelDirectory(rootURL: self.directory)
        layout = try SSDAIPackedExpertsLayoutCodec.decode(
            try files.readMetadata(
                Self.layoutFile, maxBytes: SSDAIFormatV1.packedExpertsLayoutMaxBytes))
        try SSDAIV1StructuralValidator.validate(layout)
        slotPlans = try Self.slotPlans(
            layout: layout, checkpoint: checkpoint, residentFile: residentURL)
        guard layout.numLayers == manifest.numLayers,
            layout.expertsPerLayer == manifest.expertsPerLayer
        else {
            throw Error.layout(
                "\(layout.numLayers) layers x \(layout.expertsPerLayer) experts against the "
                    + "manifest's \(manifest.numLayers) x \(manifest.expertsPerLayer)")
        }
    }

    /// The receipt the installer wrote must describe this manifest, its file
    /// set and sizes, and this directory (a moved install is re-attested with
    /// `TinyTitanRepack --verify-install`, never edited).
    static func verifyReceipt(directory: URL, manifest: Manifest) throws {
        struct Receipt: Decodable {
            struct File: Decodable {
                let size: UInt64
                let sha256: String
            }
            let schemaVersion: Int
            let manifestSha256: String
            let modelDirectoryPath: String
            let files: [String: File]
        }
        let files = try SSDAIModelDirectory(rootURL: directory)
        let receiptName = SSDAIInstallFileNames.verifiedInstallReceipt
        let receipt: Receipt
        do {
            receipt = try JSONDecoder().decode(
                Receipt.self,
                from: try files.readMetadata(receiptName, maxBytes: 64 << 20))
        } catch {
            throw Error.receipt("\(receiptName) is missing or unreadable: \(error)")
        }
        let manifestBytes = try files.readMetadata(
            SSDAIInstallFileNames.manifest, maxBytes: SSDAIManifestReader.defaultMaxBytes)
        let manifestSha = Sha256Verifier.hashData(manifestBytes)
        guard receipt.schemaVersion == 1,
            receipt.manifestSha256.lowercased() == manifestSha.lowercased()
        else {
            throw Error.receipt("it does not describe this manifest")
        }
        guard receipt.modelDirectoryPath == directory.standardizedFileURL.path else {
            throw Error.receipt(
                "issued for \(receipt.modelDirectoryPath), the model is at "
                    + "\(directory.path); re-issue it with TinyTitanRepack --verify-install "
                    + "--input-ssdai \(directory.path)")
        }
        var expected = Set(manifest.files.keys)
        expected.insert(SSDAIInstallFileNames.manifest)
        guard Set(receipt.files.keys) == expected else {
            throw Error.receipt("its file set differs from the manifest's")
        }
        for (name, entry) in manifest.files {
            guard let attested = receipt.files[name], attested.size == entry.size,
                attested.sha256.lowercased() == entry.sha256.lowercased(),
                try files.fileSize(name) == entry.size
            else {
                throw Error.receipt("\(name) does not match its attested size and digest")
            }
        }
    }
}
