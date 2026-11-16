import Foundation
import Testing
import TinyTitanFormat

@testable import TinyTitanRepackCore
@testable import TinyTitanCore

/// The manifest must describe the payload truthfully, and `--verify-install`
/// must refuse an install that does not.
///
/// Two bugs shipped past every other check because a manifest can be internally
/// consistent and still lie about the bytes beside it:
///
///   - the repacker wrote only the five width slots and dropped the source
///     checkpoint's per-tensor widths, so a 4-bit build's 8-bit attention K/V
///     were dequantized as 4-bit -- the model loaded, ran, and talked fluent
///     nonsense;
///   - `routedExpert` started at a literal 4, so every 8-bit dense install
///     advertised itself as 4-bit and the catalog skipped it as a duplicate of
///     the real 4-bit one: an install that installs, verifies, loads, and
///     cannot be selected.
///
/// The width is not something the manifest has to be trusted for. A u32 packed
/// weight's byte extent determines it exactly, so the payload is the authority.
/// These tests hold both halves of that: that the writer keeps describing the
/// payload (end to end, through a real repack), and that the validator rejects
/// a description that has drifted from it.
@Suite struct QuantManifestPayloadAgreementTests {

    // MARK: - The writer, end to end

    /// A repack of the synthetic Qwen snapshot must carry per-tensor widths.
    ///
    /// The synthetic config declares 8-bit overrides on the router and the
    /// shared-expert gate against a 4-bit base (`SyntheticSnapshot.buildQwen`),
    /// which is the same shape as the real checkpoints: a "4-bit model" whose
    /// resident tensors are not uniformly 4-bit. If the writer drops them again,
    /// this fails.
    @Test func aRepackRecordsTheSourcePerTensorWidths() async throws {
        let root = temporaryRoot("quant-agreement-writer")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.ssdai")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwen(at: snapshot, weightBits: 4)
        try writeTokenizerFiles(at: snapshot)

        _ = try await RemoteStreamingRepacker.runLocalSnapshot(
            options: LocalSnapshotRepackOptions(
                inputSnapshotDir: snapshot,
                outputDir: output,
                modelID: "synthetic-qwen-4bit",
                minFreeReserveBytes: 0))

        let manifest = try readManifestObject(at: output)
        let quant = try #require(manifest["quant"] as? [String: Any])
        let slots: Set<String> = [
            "embedding", "attention", "router",
            "sharedExpert", "routedExpert",
        ]
        let overrides = Set(quant.keys).subtracting(slots)
        #expect(
            !overrides.isEmpty,
            Comment(
                rawValue:
                    "the repack recorded no per-tensor widths; the slots alone cannot "
                    + "describe a build whose tensors are not all one width"))

        // The declared 8-bit tensors must be recorded as 8-bit, not left to a
        // 4-bit slot.
        let router = try #require(
            quant["language_model.model.layers.0.mlp.gate"]
                as? [String: Any])
        #expect(router["weightBits"] as? Int == 8)
        #expect(quant["router"] is [String: Any])
    }

    /// A freshly repacked synthetic install must pass `--verify-install`.
    ///
    /// This is the check the installer runs after every repack, so a writer that
    /// stops describing its payload fails the install rather than reaching a
    /// user.
    @Test func aFreshRepackVerifies() async throws {
        let root = temporaryRoot("quant-agreement-verify")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.ssdai")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwen(at: snapshot, weightBits: 4)
        try writeTokenizerFiles(at: snapshot)

        _ = try await RemoteStreamingRepacker.runLocalSnapshot(
            options: LocalSnapshotRepackOptions(
                inputSnapshotDir: snapshot,
                outputDir: output,
                modelID: "synthetic-qwen-4bit",
                minFreeReserveBytes: 0))

        let result = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputSSDAI: output))
        #expect(result.fileCount > 0)
    }

    /// Tampering with a width in the manifest must fail verification.
    ///
    /// The byte comparison is what makes the check independent of the writer:
    /// the manifest is edited, the payload is not.
    @Test func aLyingOverrideIsRefused() async throws {
        let root = temporaryRoot("quant-agreement-tamper")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.ssdai")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwen(at: snapshot, weightBits: 4)
        try writeTokenizerFiles(at: snapshot)

        _ = try await RemoteStreamingRepacker.runLocalSnapshot(
            options: LocalSnapshotRepackOptions(
                inputSnapshotDir: snapshot,
                outputDir: output,
                modelID: "synthetic-qwen-4bit",
                minFreeReserveBytes: 0))

        let manifestPath = (output as NSString).appendingPathComponent("manifest.json")
        var manifest = try readManifestObject(at: output)
        var quant = try #require(manifest["quant"] as? [String: Any])
        var router = try #require(
            quant["language_model.model.layers.0.mlp.gate"]
                as? [String: Any])
        router["weightBits"] = 4  // the payload is 8-bit
        quant["language_model.model.layers.0.mlp.gate"] = router
        manifest["quant"] = quant
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: manifestPath))

        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputSSDAI: output))
        }
    }

    // MARK: - The validator's own rules

    /// Neither reserved artifact may appear in the manifest's payload table.
    ///
    /// `verified-install.json` attests an install from outside it, so a `files`
    /// entry naming it would have the install attesting to its own receipt. The
    /// name is declared once, in `TinyTitanFormat`, for the writer, the reader
    /// and this validator, and the test reads it from there rather than
    /// hardcoding it: a rename that updates one side only fails here instead of
    /// shipping the split.
    @Test func aReservedArtifactNameIsRefusedAsPayload() async throws {
        let root = temporaryRoot("reserved-names")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.ssdai")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwen(at: snapshot, weightBits: 4)
        try writeTokenizerFiles(at: snapshot)
        _ = try await RemoteStreamingRepacker.runLocalSnapshot(
            options: LocalSnapshotRepackOptions(
                inputSnapshotDir: snapshot,
                outputDir: output,
                modelID: "synthetic-qwen-4bit",
                minFreeReserveBytes: 0))

        let clean = try readManifestObject(at: output)
        for name in SSDAIInstallFileNames.reserved.sorted() {
            var manifest = clean
            var files = try #require(manifest["files"] as? [String: Any])
            files[name] = ["size": 1, "sha256": String(repeating: "0", count: 64)]
            manifest["files"] = files
            #expect(
                throws: TinyTitanFormatError.invalid(
                    field: "manifest.files.\(name)",
                    reason: "reserved artifact filename")
            ) {
                _ = try SSDAIManifestCodec.decode(
                    try JSONSerialization.data(withJSONObject: manifest))
            }
        }
    }

    /// The dense shape: a slot fallback is what the CPU reader uses, so a slot
    /// that disagrees with the bytes is the original bug.
    @Test func aDenseSlotFallbackThatDisagreesWithThePayloadIsRefused() throws {
        // A 512x2048 tensor at 8 bits: 512 * 2048 / 4 = 262144 u32 words.
        let entry = packed(
            name: "language_model.model.layers.3.self_attn.k_proj.weight",
            rows: 512, columns: 2048, bits: 8)
        // The slots say 4-bit, and there are no per-tensor widths: this is the
        // pre-fix manifest exactly.
        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.validateQuantAgainstResident(
                quant: slots(embedding: 8, attention: 4, routedExpert: 4),
                expertsPerLayer: 0,
                entries: [entry])
        }
    }

    /// The same install is fine once the tensor carries its real width, which is
    /// what the writer now emits.
    @Test func anExplicitWidthMakesTheSameInstallReadable() throws {
        let entry = packed(
            name: "language_model.model.layers.3.self_attn.k_proj.weight",
            rows: 512, columns: 2048, bits: 8)
        // The body of the model is still 4-bit, so the payload's dominant width
        // is an unambiguous 4 and the model-width check is satisfied; only this
        // tensor disagrees with its slot.
        let body = [
            packed(
                name: "language_model.model.layers.0.mlp.down_proj.weight",
                rows: 512, columns: 2048, bits: 4),
            packed(
                name: "language_model.model.layers.0.mlp.up_proj.weight",
                rows: 512, columns: 2048, bits: 4),
        ]
        let overrides = [
            "language_model.model.layers.3.self_attn.k_proj":
                SSDAIManifestQuantSlotV1(
                    weightBits: 8, scheme: "affine",
                    scaleType: "BF16", biasType: "BF16",
                    groupSize: 64)
        ]
        try VerifiedInstallTool.validateQuantAgainstResident(
            quant: slots(embedding: 8, attention: 4, routedExpert: 4, overrides: overrides),
            expertsPerLayer: 0,
            entries: [entry] + body)
    }

    /// A payload with no dominant width must not make the verdict depend on
    /// dictionary iteration order: any width the payload actually uses is
    /// accepted, and nothing else is.
    @Test func aTiedPayloadAcceptsEitherWidthButNotAThird() throws {
        let four = packed(
            name: "language_model.model.layers.0.mlp.down_proj.weight",
            rows: 512, columns: 2048, bits: 4)
        let eight = packed(
            name: "language_model.model.layers.0.mlp.up_proj.weight",
            rows: 512, columns: 2048, bits: 8)
        // The 8-bit tensor carries its own width, so the *readability* check is
        // satisfied and only the model-width rule is under test here.
        let overrides = [
            "language_model.model.layers.0.mlp.up_proj":
                SSDAIManifestQuantSlotV1(
                    weightBits: 8, scheme: "affine",
                    scaleType: "BF16", biasType: "BF16",
                    groupSize: 64)
        ]
        // Both widths are present in the payload, so either declaration is
        // truthful.
        for declared in [4, 8] {
            try VerifiedInstallTool.validateQuantAgainstResident(
                quant: slots(
                    embedding: 8, attention: 4, routedExpert: declared,
                    overrides: overrides),
                expertsPerLayer: 0,
                entries: [four, eight])
        }
        // One that appears nowhere is not, however the tie resolves.
        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.validateQuantAgainstResident(
                quant: slots(
                    embedding: 8, attention: 4, routedExpert: 6,
                    overrides: overrides),
                expertsPerLayer: 0,
                entries: [four, eight])
        }
    }

    /// The embedding slot covers the tied head, so an embedding-slot tensor is
    /// judged against that slot and not the attention one.
    @Test func theEmbeddingSlotGovernsTheTiedHead() throws {
        let embed = packed(
            name: "language_model.model.embed_tokens.weight",
            rows: 256, columns: 2048, bits: 8)
        let head = packed(
            name: "language_model.lm_head.weight",
            rows: 256, columns: 2048, bits: 8)
        // attention is 4 but embedding is 8, and both entries are 8-bit.
        try VerifiedInstallTool.validateQuantAgainstResident(
            quant: slots(embedding: 8, attention: 4, routedExpert: 8),
            expertsPerLayer: 0,
            entries: [embed, head])
    }

    /// The 8-bit-install-claims-4-bit bug: with no routed experts the slot
    /// describes no tensor, so a wrong value is pure misinformation.
    @Test func aDenseInstallCannotMisdialTheModelWidth() throws {
        let embed = packed(
            name: "language_model.model.embed_tokens.weight",
            rows: 256, columns: 2048, bits: 8)
        let proj = packed(
            name: "language_model.model.layers.0.mlp.down_proj.weight",
            rows: 256, columns: 2048, bits: 8)
        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.validateQuantAgainstResident(
                quant: slots(embedding: 8, attention: 8, routedExpert: 4),
                expertsPerLayer: 0,
                entries: [embed, proj])
        }
        // The same payload with the width it actually uses is accepted.
        try VerifiedInstallTool.validateQuantAgainstResident(
            quant: slots(embedding: 8, attention: 8, routedExpert: 8),
            expertsPerLayer: 0,
            entries: [embed, proj])
    }

    /// A packed-expert install keeps its routed-expert widths in
    /// `packed_experts/layout.json`, and its resident tensors are the GPU path's
    /// business. The slot fallback must not be applied to it, or every existing
    /// MoE install would start failing verification.
    @Test func aPackedExpertInstallDoesNotUseTheSlotFallback() throws {
        // 4-bit bytes with a 2-bit-looking attention slot is not a real state;
        // what matters is that no slot fallback is consulted at all, so a
        // resident tensor with no override is left alone.
        let entry = packed(
            name: "language_model.model.layers.0.mlp.gate_proj.weight",
            rows: 512, columns: 2048, bits: 8)
        try VerifiedInstallTool.validateQuantAgainstResident(
            quant: slots(embedding: 4, attention: 4, routedExpert: 4),
            expertsPerLayer: 256,
            entries: [entry])
    }

    /// An explicit width is checked even on a packed-expert install, because a
    /// declared override is a claim about bytes wherever those bytes live.
    @Test func aLyingOverrideIsRefusedOnAPackedExpertInstallToo() throws {
        let entry = packed(
            name: "language_model.model.layers.0.self_attn.q_proj.weight",
            rows: 512, columns: 2048, bits: 8)
        let overrides = [
            "language_model.model.layers.0.self_attn.q_proj":
                SSDAIManifestQuantSlotV1(
                    weightBits: 4, scheme: "affine",
                    scaleType: "BF16", biasType: "BF16",
                    groupSize: 64)
        ]
        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.validateQuantAgainstResident(
                quant: slots(
                    embedding: 8, attention: 8, routedExpert: 4,
                    overrides: overrides),
                expertsPerLayer: 256,
                entries: [entry])
        }
    }

    /// A byte extent that cannot be a whole number of values per element is a
    /// broken index, not a width to guess at.
    @Test func aNonIntegralByteExtentIsRefused() throws {
        let entry = SSDAIResidentIndexEntryV1(
            name: "language_model.model.layers.0.mlp.down_proj.weight",
            dtype: SSDAIFormatV1.DType.u32.rawValue,
            fileOffset: 0, sizeBytes: 1000,
            shape: [512, 2048, 0, 0],
            scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0)
        #expect(throws: RepackError.self) {
            try VerifiedInstallTool.validateQuantAgainstResident(
                quant: slots(embedding: 4, attention: 4, routedExpert: 4),
                expertsPerLayer: 0,
                entries: [entry])
        }
    }

    /// An unquantized (bf16) entry is never dequantized, so it has no width to
    /// agree with and must not be judged.
    @Test func unquantizedEntriesAreNotJudged() throws {
        let norm = SSDAIResidentIndexEntryV1(
            name: "language_model.model.norm.weight",
            dtype: SSDAIFormatV1.DType.bf16.rawValue,
            fileOffset: 0, sizeBytes: 256,
            shape: [128, 0, 0, 0],
            scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0)
        try VerifiedInstallTool.validateQuantAgainstResident(
            quant: slots(embedding: 4, attention: 4, routedExpert: 4),
            expertsPerLayer: 0,
            entries: [norm])
    }

    // MARK: - Fixtures

    /// A packed u32 weight whose byte extent is exactly `rows * columns * bits / 8`.
    private func packed(
        name: String, rows: UInt32, columns: UInt32,
        bits: UInt64
    ) -> SSDAIResidentIndexEntryV1 {
        let bytes = UInt64(rows) * UInt64(columns) * bits / 8
        return SSDAIResidentIndexEntryV1(
            name: name,
            dtype: SSDAIFormatV1.DType.u32.rawValue,
            fileOffset: 0, sizeBytes: bytes,
            shape: [rows, columns, 0, 0],
            scaleOffset: 0, scaleSize: 0, biasOffset: 0, biasSize: 0)
    }

    private func slots(
        embedding: Int, attention: Int, routedExpert: Int,
        overrides: [String: SSDAIManifestQuantSlotV1]? = nil
    )
        -> SSDAIManifestQuantV1
    {
        func slot(_ bits: Int) -> SSDAIManifestQuantSlotV1 {
            SSDAIManifestQuantSlotV1(
                weightBits: bits, scheme: "affine",
                scaleType: "BF16", biasType: "BF16", groupSize: 64)
        }
        return SSDAIManifestQuantV1(
            embedding: slot(embedding),
            attention: slot(attention),
            router: slot(8),
            sharedExpert: slot(8),
            routedExpert: slot(routedExpert),
            overrides: overrides)
    }

    /// The three files `importLocalSnapshot` requires beside a snapshot.
    ///
    /// It copies them without parsing, so the contents only need to exist; the
    /// synthetic builder writes the model config and no tokenizer.
    private func writeTokenizerFiles(at directory: String) throws {
        let json = Data(#"{"version":"1.0"}"#.utf8)
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            try json.write(
                to: URL(fileURLWithPath: (directory as NSString).appendingPathComponent(name)))
        }
    }

    private func readManifestObject(at directory: String) throws -> [String: Any] {
        let data = try Data(
            contentsOf: URL(
                fileURLWithPath: (directory as NSString).appendingPathComponent("manifest.json")))
        return try #require(
            try JSONSerialization.jsonObject(with: data)
                as? [String: Any])
    }

    private func temporaryRoot(_ tag: String) -> String {
        let base = (FileManager.default.currentDirectoryPath as NSString)
            .appendingPathComponent(".build/test-artifacts")
        let path = (base as NSString).appendingPathComponent(tag)
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }
}
