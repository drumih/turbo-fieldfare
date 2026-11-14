import Darwin
import Foundation
import TinyTitanFormat
import TinyTitanCore

public struct VerifyInstallOptions: Sendable {
    public let inputSSDAI: String

    public init(inputSSDAI: String) {
        self.inputSSDAI = inputSSDAI
    }
}

public struct VerifyInstallResult: Sendable {
    public let receiptPath: String
    public let fileCount: Int
    public let bytesVerified: UInt64
    public let unexpectedEntries: [String]
}

public enum VerifiedInstallTool {
    // 64 MiB: sized for Qwen 3.6's ~22 MB layout.json (40 layers x 256 experts).
    public static let metadataMaxBytes: UInt64 = 64 * 1024 * 1024
    // Hard ceiling for any single payload file named by the manifest. The
    // manifest size remains the exact per-file contract (verified below); this
    // cap only rejects absurd corrupted manifests before hashing.
    //
    // 128 GiB, sized against the largest legitimate payload: Qwen3.8-Flash-Next's
    // n-gram embedding table is 102,400,491,520 bytes (95.4 GiB) of fp16 rows.
    // The previous 64 GiB predated any file that large and rejected a correct
    // install. Kept finite, and well under any plausible disk, so a manifest
    // claiming a terabyte is still refused before hashing begins.
    /// The per-file sanity bound on a manifest entry, taken from the repacker's
    /// own acceptance limit (`RepackPlanner.maximumPassthroughFileBytes`) rather
    /// than chosen separately: a cap below what the repacker will write makes an
    /// install the tool just produced impossible to verify.
    public static let payloadMaxBytes: UInt64 = RepackPlanner.maximumPassthroughFileBytes

    public static func run(options: VerifyInstallOptions) throws -> VerifyInstallResult {
        let access = try SSDAIDirectoryAccess(rootPath: options.inputSSDAI)
        let manifestPath = "manifest.json"
        try SSDAIPathValidator.validateRelativePath(
            manifestPath, field: "manifest.files.manifest.json")
        let manifestSize = try access.fileSize(manifestPath)
        let manifestSha = try access.hash(manifestPath, noCache: true)
        let manifest = try loadManifest(access: access)
        try validatePackedExpertLayout(access: access, manifest: manifest)
        try validateQuantAgainstResident(access: access, manifest: manifest)

        var files: [RepackAudit.OutputFile] = []
        files.reserveCapacity(manifest.files.count)
        var bytesVerified = manifestSize
        for relativePath in manifest.files.keys.sorted() {
            guard let entry = manifest.files[relativePath] else { continue }
            // Path validation before any filesystem operation: reject `..`,
            // absolute and non-normalized names, and duplicate filesystem keys.
            try SSDAIPathValidator.validateRelativePath(
                relativePath, field: "manifest.files.\(relativePath)")
            guard entry.size <= payloadMaxBytes else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) manifest size \(entry.size) exceeds "
                        + "the \(payloadMaxBytes)-byte per-file cap")
            }
            // All reads go through the root-anchored openat chain with
            // O_NOFOLLOW at every level (no symlink escapes) and the
            // descriptor's type is verified after open.
            let actualSize = try access.fileSize(relativePath)
            guard actualSize == entry.size else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != manifest \(entry.size)")
            }
            let actualSha = try access.hash(relativePath, noCache: true)
            guard actualSha.lowercased() == entry.sha256.lowercased() else {
                throw RepackError.configurationInvalid(detail: "\(relativePath) SHA mismatch")
            }
            let (verified, overflow) = bytesVerified.addingReportingOverflow(actualSize)
            guard !overflow else {
                throw RepackError.configurationInvalid(detail: "verified byte total overflows")
            }
            bytesVerified = verified
            files.append(
                RepackAudit.OutputFile(
                    relativePath: relativePath,
                    size: actualSize,
                    sha256: actualSha))
        }
        let unexpectedEntries = try findUnexpectedEntries(access: access, manifest: manifest)

        let receiptData = try VerifiedInstallReceiptWriter.encode(
            outputDir: access.rootPath,
            manifestSha256: manifestSha,
            manifestSize: manifestSize,
            sourceRepoID: nil,
            sourceRevision: manifest.sourceSnapshotHash,
            toolVersion: "TinyTitanRepack verify-install",
            files: files)
        let receiptPath =
            access.rootPath
            + "/" + VerifiedInstallReceiptWriter.fileName
        try receiptData.write(to: URL(fileURLWithPath: receiptPath), options: .atomic)
        return VerifyInstallResult(
            receiptPath: receiptPath,
            fileCount: files.count + 1,
            bytesVerified: bytesVerified,
            unexpectedEntries: unexpectedEntries)
    }

    static func validatePackedExpertLayout(inputSSDAI: String) throws {
        let access = try SSDAIDirectoryAccess(rootPath: inputSSDAI)
        let manifest = try loadManifest(access: access)
        try validatePackedExpertLayout(access: access, manifest: manifest)
    }

    /// Read the resident index and hand it to the width cross-check.
    ///
    /// The index is the head of `model_weights.bin`, and that file is the model
    /// itself, so this reads it in two bounded steps -- the 24-byte header, then
    /// exactly the index it names -- and never touches the payload.
    private static func validateQuantAgainstResident(
        access: SSDAIDirectoryAccess,
        manifest: Manifest
    ) throws {
        guard let quant = manifest.quant else {
            throw RepackError.configurationInvalid(detail: "manifest.json has no quant block")
        }
        let relativePath = "model_weights.bin"
        guard manifest.files[relativePath] != nil else {
            throw RepackError.configurationInvalid(detail: "manifest missing \(relativePath)")
        }
        let headerBytes = try access.readPrefix(
            relativePath, maxBytes: UInt64(SSDAIFormatV1.residentHeaderBytes))
        guard headerBytes.count == SSDAIFormatV1.residentHeaderBytes else {
            throw RepackError.configurationInvalid(
                detail: "\(relativePath) is shorter than the resident index header")
        }
        let header = try headerBytes.withUnsafeBytes {
            try SSDAIResidentIndexCodec.decodeHeader($0)
        }
        guard header.indexSize <= UInt64(SSDAIFormatV1.residentIndexMaxBytes) else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(relativePath) index \(header.indexSize) exceeds the "
                    + "\(SSDAIFormatV1.residentIndexMaxBytes)-byte v1 cap")
        }
        let indexBytes = try access.readPrefix(relativePath, maxBytes: header.indexSize)
        guard indexBytes.count == Int(header.indexSize) else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(relativePath) holds \(indexBytes.count) bytes but its header "
                    + "claims an index of \(header.indexSize)")
        }
        let entries = try indexBytes.withUnsafeBytes {
            try SSDAIResidentIndexCodec.decodeRegion($0, header: header)
        }
        try validateQuantAgainstResident(
            quant: quant,
            expertsPerLayer: manifest.expertsPerLayer,
            entries: entries)
    }

    /// Cross-check the manifest's declared widths against the resident bytes.
    ///
    /// Everything else in this file hashes well-formed files. This is the only
    /// check that asks whether the manifest's *description* of the payload is
    /// true, and it exists because two bugs shipped past all the others:
    ///
    ///   - the repacker wrote only the five width slots and dropped the source
    ///     checkpoint's per-tensor widths, so a 4-bit build's 8-bit attention
    ///     K/V were dequantized as 4-bit;
    ///   - `routedExpert` started at a literal 4, so every 8-bit dense install
    ///     advertised itself as 4-bit and the catalog skipped it as a duplicate
    ///     of the real 4-bit one. It installed, verified, loaded, and could not
    ///     be selected.
    ///
    /// Both passed every existing check, including this tool's, because a
    /// manifest can be internally consistent and still lie about the bytes beside
    /// it. The width is not something the manifest has to be trusted for: a u32
    /// packed weight's byte extent determines it exactly --
    /// `sizeBytes = rows * columns * bits / 8` -- so the payload is the
    /// authority and the manifest is what gets checked.
    ///
    /// The question asked is "would this install *read* correctly", not "was it
    /// written by the current writer". For each packed tensor the check resolves
    /// the width exactly as the CPU reader does -- an explicit per-tensor entry,
    /// else the slot the reader falls back to -- and requires it to equal what
    /// the bytes say. That is why the duplication of the reader's fallback rule
    /// below is deliberate: it is the contract being verified, and it is what
    /// makes the failure name the tensors that would be dequantized wrongly
    /// rather than every tensor the writer happened not to annotate.
    static func validateQuantAgainstResident(
        quant: SSDAIManifestQuantV1,
        expertsPerLayer: Int,
        entries: [SSDAIResidentIndexEntryV1]
    ) throws {
        // Every dtype-0 entry is a packed u32 `.weight`: the planner only marks a
        // tensor quantized when its source dtype is u32 and its name ends in
        // `.weight`, and everything else is stored bf16.
        let attentionSlot = quant.attention.weightBits
        let embeddingSlot = quant.embedding.weightBits
        var impliedCounts: [Int: Int] = [:]
        var unreadable: [String] = []

        // A bf16 tensor inside a quantized slot must be named in `promoted`.
        // Without that the manifest says the slot's width applies to it, and a
        // reader that trusts the slot would dequantize bf16 bytes as integers;
        // this is the check that makes the claim true rather than assumed.
        let promotedStems = Set(quant.promoted ?? [])
        let slotRoleSuffixes = [
            ".embed_tokens", ".lm_head",
            ".self_attn.q_proj", ".self_attn.k_proj",
            ".self_attn.v_proj", ".self_attn.o_proj",
            ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z",
            ".linear_attn.in_proj_a", ".linear_attn.in_proj_b",
            ".linear_attn.out_proj", ".mlp.gate",
            ".mlp.shared_expert.gate_proj",
            ".mlp.shared_expert.up_proj",
            ".mlp.shared_expert.down_proj",
        ]
        for entry in entries where entry.dtype != SSDAIFormatV1.DType.u32.rawValue {
            guard entry.name.hasSuffix(".weight") else { continue }
            let stem = String(entry.name.dropLast(".weight".count))
            let isSlotRole = slotRoleSuffixes.contains { stem.hasSuffix($0) }
            if isSlotRole, !promotedStems.contains(stem) {
                throw RepackError.configurationInvalid(
                    detail: "\(entry.name) is stored at the checkpoint's bf16 but the "
                        + "manifest does not list it in `promoted`, so its slot's width "
                        + "would be read as applying to it")
            }
        }
        for entry in entries where entry.dtype == SSDAIFormatV1.DType.u32.rawValue {
            let implied = try impliedWidth(of: entry)
            impliedCounts[implied, default: 0] += 1
            // Overrides are keyed by stem, without the `.weight` suffix.
            let stem =
                entry.name.hasSuffix(".weight")
                ? String(entry.name.dropLast(".weight".count)) : entry.name

            let resolved: Int
            if let declared = quant.overrides?[stem]?.weightBits {
                resolved = declared
            } else if expertsPerLayer == 0 {
                // `AffineSnapshot.init(ssdai:)`: the embedding slot covers the
                // tied head as well, and the attention slot is the default.
                resolved =
                    stem.hasSuffix("embed_tokens") || stem.hasSuffix("lm_head")
                    ? embeddingSlot : attentionSlot
            } else {
                // A packed-expert install keeps its routed-expert widths in
                // `packed_experts/layout.json`, and its resident tensors are the
                // GPU path's business, not the CPU reader's. Nothing to check.
                continue
            }

            if resolved != implied {
                unreadable.append(
                    "\(entry.name) would be read as \(resolved)-bit but "
                        + "is \(implied)-bit (\(entry.sizeBytes) bytes at "
                        + "\(entry.shape[0])x\(entry.shape[1]))")
            }
        }

        guard unreadable.isEmpty else {
            let shown = unreadable.prefix(3).joined(separator: "; ")
            let more = unreadable.count > 3 ? " and \(unreadable.count - 3) more" : ""
            throw RepackError.configurationInvalid(
                detail:
                    "the manifest's widths disagree with the resident payload, so this "
                    + "install would dequantize wrongly: \(shown)\(more). Repack it "
                    + "from its source snapshot; the bytes are fine, the description "
                    + "of them is not")
        }

        // With no routed experts the slot describes no tensor, so it is not
        // derived from anything and has to be checked on its own. It is what
        // `ManifestIdentity.weightBits` reads and what becomes the `_<bits>-Bit`
        // suffix in `/v1/models`, so a wrong value here is not cosmetic.
        //
        // The requirement is that the declared width is one the payload actually
        // uses, preferring the dominant one. Written as a *set* rather than
        // "the maximum" on purpose: resolving a tie by picking one of the tied
        // widths would make the verdict depend on dictionary iteration order,
        // which is not stable between runs. On a genuine tie any tied width is
        // accepted; with a clear majority only that width is.
        guard expertsPerLayer == 0, let most = impliedCounts.values.max() else { return }
        let dominant = Set(impliedCounts.filter { $0.value == most }.keys)
        let declared = quant.routedExpert.weightBits
        guard dominant.contains(declared) else {
            let total = impliedCounts.values.reduce(0, +)
            let histogram =
                impliedCounts
                .sorted { $0.key < $1.key }
                .map { "\($0.key)-bit x\($0.value)" }
                .joined(separator: ", ")
            throw RepackError.configurationInvalid(
                detail:
                    "the install has no routed experts, but declares a routed-expert "
                    + "width of \(declared) while its resident payload is "
                    + "\(histogram) over \(total) quantized tensors. That value names "
                    + "the model in /v1/models and decides which build the catalog "
                    + "thinks this is")
        }
    }

    /// The width a packed u32 weight's own byte extent implies.
    ///
    /// A u32 word holds `32 / bits` values, so a `rows x columns` logical matrix
    /// occupies `rows * columns * bits / 8` bytes and nothing else. The shape is
    /// the *logical* width in the resident index, unlike a safetensors header,
    /// which is what makes this invertible.
    private static func impliedWidth(of entry: SSDAIResidentIndexEntryV1) throws -> Int {
        let rows = UInt64(entry.shape[0])
        let columns = UInt64(entry.shape[1])
        guard rows > 0, columns > 0 else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(entry.name): packed weight with a zero dimension "
                    + "\(entry.shape[0])x\(entry.shape[1])")
        }
        guard entry.sizeBytes % 4 == 0 else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(entry.name): \(entry.sizeBytes) bytes is not a whole number "
                    + "of u32 words")
        }
        let (values, valuesOverflow) = (entry.sizeBytes / 4).multipliedReportingOverflow(by: 32)
        let (cells, cellsOverflow) = rows.multipliedReportingOverflow(by: columns)
        guard !valuesOverflow, !cellsOverflow else {
            throw RepackError.configurationInvalid(
                detail: "\(entry.name): dimensions overflow")
        }
        guard values % cells == 0 else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(entry.name): \(entry.sizeBytes) bytes at \(entry.shape[0])x"
                    + "\(entry.shape[1]) does not divide into a whole number of "
                    + "values per element")
        }
        let bits = values / cells
        guard bits == 4 || bits == 8 else {
            throw RepackError.configurationInvalid(
                detail:
                    "\(entry.name): implied width \(bits) is not a supported "
                    + "4- or 8-bit packing")
        }
        return Int(bits)
    }

    /// Reads the manifest through **both** decoders.
    ///
    /// The local struct carries the fields this tool needs, but it performs none
    /// of the format layer's structural validation: filesystem-equivalent
    /// duplicate paths (case-insensitive or Unicode-normalized), reserved
    /// artifact names such as `verified-install.json`, and file-path-versus-
    /// directory-prefix collisions. The runtime runs those checks on every load,
    /// so without them here this tool could certify a directory the runtime then
    /// refuses — the opposite of what a verifier is for. It also means a manifest
    /// listing `verified-install.json` is rejected rather than hashed and then
    /// overwritten by the receipt this run writes.
    private static func loadManifest(access: SSDAIDirectoryAccess) throws -> Manifest {
        let data = try loadMetadataJSON(access: access, relativePath: "manifest.json")
        do {
            let wire = try SSDAIManifestCodec.decodeUnchecked(data)
            try SSDAIManifestCodec.validate(wire)
        } catch {
            throw RepackError.configurationInvalid(
                detail: "manifest.json rejected by the format validator: \(error)")
        }
        do {
            return try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw RepackError.configurationInvalid(detail: "manifest.json invalid: \(error)")
        }
    }

    static func loadMetadataJSON(
        access: SSDAIDirectoryAccess,
        relativePath: String
    ) throws -> Data {
        try SSDAIPathValidator.validateRelativePath(
            relativePath, field: "metadata.\(relativePath)")
        return try access.readMetadata(relativePath, maxBytes: metadataMaxBytes)
    }

    private static func findUnexpectedEntries(
        access: SSDAIDirectoryAccess,
        manifest: Manifest
    ) throws -> [String] {
        let declaredFiles = Set(manifest.files.keys)
            .union(["manifest.json", VerifiedInstallReceiptWriter.fileName])
        var allowed = declaredFiles
        for path in declaredFiles {
            var parts = path.split(separator: "/").map(String.init)
            while parts.count > 1 {
                _ = parts.removeLast()
                allowed.insert(parts.joined(separator: "/"))
            }
        }
        allowed.insert("tokenizer")

        let entries = try access.relativeEntries()
        var unexpected: [String] = []
        for rel in entries {
            // .DS_Store may appear at any depth (Finder writes it into
            // subdirectories too).
            if rel == ".DS_Store" || rel.hasSuffix("/.DS_Store") { continue }
            if rel == "tokenizer" || rel.hasPrefix("tokenizer/") { continue }
            if !allowed.contains(rel) {
                unexpected.append(rel)
            }
        }
        return unexpected.sorted()
    }
}
