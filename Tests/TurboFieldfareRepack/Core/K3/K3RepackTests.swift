import Foundation
import Synchronization
import Testing
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

/// Stage-D K3 repack profile tests, driven through `K3LocalSnapshotSource`
/// over a synthetic official-layout checkpoint on disk (no network).
@Suite(.serialized)
struct K3RepackTests {

    // MARK: - Helpers

    struct Built {
        let dir: String
        let snapshot: K3SyntheticCheckpoint.Snapshot
    }

    static func buildSnapshot(_ tag: String) throws -> Built {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-k3repack-\(tag)-\(UUID().uuidString)")
        let snapshot = try K3SyntheticCheckpoint.build(at: dir)
        return Built(dir: dir, snapshot: snapshot)
    }

    static func loadSnapshot(_ built: Built) async throws -> K3Snapshot {
        try await K3LocalSnapshotSource(directory: built.dir)
            .loadSnapshot(metadataDirectory: built.dir,
                          partialDirectory: built.dir,
                          audit: RepackAudit())
    }

    static func makePlan(_ snapshot: K3Snapshot, outputDir: String,
                         trunkQuant: K3TrunkQuant = .int4) throws -> K3RepackPlan {
        try K3RepackPlanner.plan(arch: snapshot.arch,
                                 shardHeaders: snapshot.shardHeaders,
                                 outputDir: outputDir,
                                 trunkQuant: trunkQuant)
    }

    static func localOptions(outputDir: String, resume: Bool = false,
                             rangeChunkBytes: Int = 65_536,
                             trunkQuant: K3TrunkQuant = .int4,
                             reuseExpertsFrom: String? = nil)
        -> K3RemoteStreamingRepackOptions {
        K3RemoteStreamingRepackOptions(
            repoID: "local/kimi-k3-synthetic",
            revision: "synthetic",
            outputDir: outputDir,
            requireKnownSource: false,
            trunkQuant: trunkQuant,
            rangeChunkBytes: rangeChunkBytes,
            minFreeReserveBytes: 0,
            overwrite: true,
            resume: resume,
            reuseExpertsFrom: reuseExpertsFrom,
            rangeRetryAttempts: 0,
            retryBaseDelayNs: 0)
    }

    static func cleanUp(_ paths: [String]) {
        for path in paths {
            try? FileManager.default.removeItem(atPath: path)
            try? FileManager.default.removeItem(atPath: path + ".partial")
            try? FileManager.default.removeItem(atPath: path + ".resume.json")
            try? FileManager.default.removeItem(atPath: path + ".install.lock")
        }
    }

    /// Parse a shard's safetensors header straight off disk.
    static func shardHeader(_ path: String) throws -> [String: K3SourceTensor] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        var headerSize: UInt64 = 0
        for i in 0..<8 { headerSize |= UInt64(data[i]) << UInt64(i * 8) }
        let header = try K3Safetensors.parseHeaderBytes(
            path: (path as NSString).lastPathComponent,
            fileSize: UInt64(data.count),
            headerBytes: data.subdata(in: 8..<(8 + Int(headerSize))))
        return Dictionary(header.tensors.map { ($0.name, $0) },
                          uniquingKeysWith: { _, _ in fatalError("duplicate tensor") })
    }

    static func shardBytes(_ path: String) throws -> Data {
        try Data(contentsOf: URL(fileURLWithPath: path))
    }

    static func sha256Hex(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }

    // MARK: - (a) plan classification is exact

    @Test func planClassificationIsExact() async throws {
        let built = try Self.buildSnapshot("classify")
        defer { Self.cleanUp([built.dir]) }
        let snapshot = try await Self.loadSnapshot(built)
        let outDir = built.dir + ".out"
        let plan = try Self.makePlan(snapshot, outputDir: outDir)

        // Exclusions: exactly the vision/projector tensors.
        #expect(plan.excludedTensorNames == [
            "mm_projector.weight",
            "vision_tower.encoder.layers.0.self_attn.q_proj.weight",
        ])

        // Every source tensor is accounted for: trunk (minus synthesized
        // conv1d biases) + experts + excluded == registry.
        var registryCount = 0
        for header in snapshot.shardHeaders { registryCount += header.tensors.count }
        let zerosCount = plan.tensors.filter { $0.transform == .synthesizeZeros }.count
        let expertCount = plan.layers.reduce(0) { $0 + $1.sources.count * $1.expertsPerLayer }
        #expect(plan.tensors.count + expertCount + plan.excludedTensorNames.count
                    == registryCount + zerosCount)
        // conv1d.bias synthesis: 3 per KDA layer x 4 KDA layers.
        #expect(zerosCount == 12)

        // MoE layer set (0-based): 1, 2, 3, 4; layer 0 is dense.
        #expect(plan.layers.map(\.layerIndex) == [1, 2, 3, 4])
        for layer in plan.layers {
            #expect(layer.expertsPerLayer == 16)
            #expect(layer.subtensors.map(\.name)
                        == ["w1_packed", "w1_scales", "w2_packed", "w2_scales",
                            "w3_packed", "w3_scales"])
            #expect(layer.sources.count == 6)
            #expect(layer.sources.allSatisfy { $0.count == 16 })
        }
        // Canonical tiny blob: 3 x (128*64/2 + 128*64/32) = 13,056 -> 16,384 stride.
        #expect(plan.layers.first?.expertStride == 16_384)
        let blobSize = plan.layers.first!.subtensors.reduce(UInt64(0)) { $0 + $1.sizeInBlob }
        #expect(blobSize == 13_056)

        // Manifest arch mirrors the tiny config (and the runtime tinyConfig).
        let arch = plan.manifestArch
        #expect(arch.numLayers == 5)
        #expect(arch.denseLayers == [1])
        #expect(arch.kdaLayers == [1, 2, 3, 5])
        #expect(arch.fullAttnLayers == [4])
        #expect(arch.kda.decayLowRankSize == 64)
        #expect(arch.kda.decayProjectionSize == 128)
        #expect(arch.mla.qLoraRank == 64)
        #expect(arch.moe.latentBottleneckSize == 128)
        #expect(arch.moe.expertIntermediateSize == 64)
        #expect(arch.moe.sharedExpertIntermediateSize == 64)
        #expect(arch.hiddenActivation == "situ_glu")
        #expect(arch.bosTokenID == 500 && arch.eosTokenID == 501)

        // Resident ordering: schema order, embed first, lm_head second.
        #expect(plan.resident.entries.first?.name
                    == "language_model.model.embed_tokens.weight")
        #expect(plan.resident.entries[1].name == "language_model.lm_head.weight")
        #expect(plan.resident.entries[2].name == "language_model.model.norm.weight")
    }

    @Test func planRejectsUnknownLanguageModelTensor() async throws {
        let built = try Self.buildSnapshot("unknown")
        defer { Self.cleanUp([built.dir]) }
        let snapshot = try await Self.loadSnapshot(built)
        var headers = snapshot.shardHeaders
        let bogus = K3SourceTensor(
            name: "language_model.model.surprise.weight",
            shardPath: "model-00001-of-00002.safetensors",
            dtype: .bf16, shape: [4], absoluteOffset: 0, sizeBytes: 8)
        headers[0] = K3Safetensors.Header(tensors: headers[0].tensors + [bogus])
        #expect(throws: RepackError.self) {
            try K3RepackPlanner.plan(arch: snapshot.arch, shardHeaders: headers,
                                     outputDir: built.dir + ".out", trunkQuant: .int4)
        }
    }

    @Test func planRejectsMissingTensor() async throws {
        let built = try Self.buildSnapshot("missing")
        defer { Self.cleanUp([built.dir]) }
        let snapshot = try await Self.loadSnapshot(built)
        var headers = snapshot.shardHeaders
        headers[0] = K3Safetensors.Header(
            tensors: headers[0].tensors.filter {
                $0.name != "language_model.model.norm.weight"
            })
        #expect(throws: RepackError.self) {
            try K3RepackPlanner.plan(arch: snapshot.arch, shardHeaders: headers,
                                     outputDir: built.dir + ".out", trunkQuant: .int4)
        }
    }

    @Test func planExcludesNextnTensors() async throws {
        let built = try Self.buildSnapshot("nextn")
        defer { Self.cleanUp([built.dir]) }
        let snapshot = try await Self.loadSnapshot(built)
        var headers = snapshot.shardHeaders
        let mtp = K3SourceTensor(
            name: "language_model.model.layers.5.nextn_embedding.weight",
            shardPath: "model-00001-of-00002.safetensors",
            dtype: .bf16, shape: [4], absoluteOffset: 0, sizeBytes: 8)
        headers[0] = K3Safetensors.Header(tensors: headers[0].tensors + [mtp])
        let plan = try K3RepackPlanner.plan(arch: snapshot.arch, shardHeaders: headers,
                                            outputDir: built.dir + ".out",
                                            trunkQuant: .int4)
        #expect(plan.excludedTensorNames.contains(
            "language_model.model.layers.5.nextn_embedding.weight"))
    }

    // MARK: - (b) full local repack -> structurally valid v2 bundle

    @Test func repackProducesStructurallyValidV2Bundle() async throws {
        let built = try Self.buildSnapshot("full")
        let output = built.dir + ".gturbo"
        defer { Self.cleanUp([built.dir, output]) }
        let snapshot = try await Self.loadSnapshot(built)
        let expectedIndexSha = snapshot.indexSha256Hex

        let result = try await K3RemoteStreamingRepacker(
            options: Self.localOptions(outputDir: output),
            source: K3LocalSnapshotSource(directory: built.dir)
        ).run()
        #expect(result.reusedBytes == 0)
        #expect(result.downloadedThisRunBytes == result.remoteBytesToDownload)
        #expect(!result.dryRun)

        // File set: payload + metadata + tokenizer sidecars; no internals.
        let entries = try FileManager.default.contentsOfDirectory(atPath: output)
        #expect(entries.contains("model_weights.bin"))
        #expect(entries.contains("packed_experts"))
        #expect(entries.contains("manifest.json"))
        #expect(entries.contains("verified-install.json"))
        #expect(entries.contains("tokenizer"))
        #expect(!entries.contains("trunk-staging.bin"))
        #expect(!entries.contains(".range.tmp"))
        #expect(!entries.contains(".remote-metadata"))
        let expertsDir = try FileManager.default.contentsOfDirectory(
            atPath: (output as NSString).appendingPathComponent("packed_experts"))
        #expect(expertsDir.sorted() == ["layer_01.bin", "layer_02.bin",
                                        "layer_03.bin", "layer_04.bin",
                                        "layout.json"])
        let tokenizerDir = try FileManager.default.contentsOfDirectory(
            atPath: (output as NSString).appendingPathComponent("tokenizer"))
        #expect(tokenizerDir.sorted() == ["config.json", "tiktoken.model",
                                          "tokenizer_config.json"])

        // Manifest + layout decode and cross-validate.
        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("manifest.json")))
        let manifest = try GTurboManifestCodecV2.decode(manifestData)
        let layoutData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("packed_experts/layout.json")))
        let layout = try GTurboPackedExpertsLayoutCodecV2.decode(layoutData)
        try GTurboV2StructuralValidator.crossValidate(manifest: manifest, layout: layout)

        #expect(manifest.numLayers == 5)
        #expect(manifest.expertsPerLayer == 16)
        #expect(manifest.expertStride == 16_384)
        #expect(manifest.modelID == "local/kimi-k3-synthetic")
        #expect(manifest.sourceSnapshotHash == "sha256:" + expectedIndexSha)
        #expect(manifest.flags == KimiK3FormatProfile.flags)
        #expect(manifest.quant == KimiK3FormatProfile.quant)
        #expect(manifest.arch.numLayers == 5)
        #expect(manifest.arch.kda.numHeads == 4)
        #expect(manifest.arch.mla.qkRopeHeadDim == 8)
        #expect(manifest.arch.attnRes.blockSize == 2)
        #expect(manifest.arch.moe.topKExperts == 4)

        // Every manifest file entry matches the bytes on disk.
        // 1 resident + 4 layer files + layout + 3 tokenizer sidecars.
        #expect(manifest.files.count == 9)
        for (rel, entry) in manifest.files {
            let path = (output as NSString).appendingPathComponent(rel)
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            #expect(UInt64(data.count) == entry.size, "\(rel) size drift")
            #expect(Self.sha256Hex(data) == entry.sha256, "\(rel) sha drift")
        }

        // Tokenizer sidecars are byte-identical to the synthetic source.
        for name in ["config.json", "tiktoken.model", "tokenizer_config.json"] {
            let bundled = try Data(contentsOf: URL(fileURLWithPath:
                (output as NSString).appendingPathComponent("tokenizer/\(name)")))
            let source = try Data(contentsOf: URL(fileURLWithPath:
                (built.dir as NSString).appendingPathComponent(name)))
            #expect(bundled == source, "tokenizer/\(name) mismatch")
        }

        // The receipt covers manifest + payload files.
        let receiptData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("verified-install.json")))
        let receipt = try JSONSerialization.jsonObject(with: receiptData) as! [String: Any]
        let receiptFiles = receipt["files"] as! [String: Any]
        #expect(receiptFiles["manifest.json"] != nil)
        #expect(receiptFiles["model_weights.bin"] != nil)
        #expect(receiptFiles["packed_experts/layer_01.bin"] != nil)
    }

    @Test func int8UpgradeClonesVerifiedExpertsAndDownloadsOnlyTrunk() async throws {
        let built = try Self.buildSnapshot("clone-upgrade")
        let output = built.dir + ".gturbo"
        defer { Self.cleanUp([built.dir, output]) }

        let initial = try await K3RemoteStreamingRepacker(
            options: Self.localOptions(outputDir: output),
            source: K3LocalSnapshotSource(directory: built.dir)
        ).run()
        let oldManifestData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("manifest.json")))
        let oldManifest = try GTurboManifestCodecV2.decode(oldManifestData)
        let oldExpertEntries = oldManifest.files.filter {
            $0.key.hasPrefix("packed_experts/layer_")
        }
        let oldExpertBytes = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent(
                "packed_experts/layer_01.bin")))

        let upgraded = try await K3RemoteStreamingRepacker(
            options: Self.localOptions(
                outputDir: output,
                trunkQuant: .int8,
                reuseExpertsFrom: output),
            source: K3LocalSnapshotSource(directory: built.dir)
        ).run()

        #expect(upgraded.remoteBytesToDownload < initial.remoteBytesToDownload)
        #expect(upgraded.reusedBytes
            == oldExpertEntries.values.reduce(UInt64(0)) { $0 + $1.size })
        #expect(upgraded.downloadedThisRunBytes == upgraded.remoteBytesToDownload)
        #expect(!FileManager.default.fileExists(atPath: output + ".partial"))
        #expect(!FileManager.default.fileExists(atPath: output + ".resume.json"))

        let newManifestData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("manifest.json")))
        let newManifest = try GTurboManifestCodecV2.decode(newManifestData)
        #expect(newManifest.quant == KimiK3FormatProfile.quantInt8)
        for (relativePath, oldEntry) in oldExpertEntries {
            #expect(newManifest.files[relativePath] == oldEntry)
        }
        let newExpertBytes = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent(
                "packed_experts/layer_01.bin")))
        #expect(newExpertBytes == oldExpertBytes)

        let receiptData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("verified-install.json")))
        let receipt = try JSONSerialization.jsonObject(with: receiptData) as! [String: Any]
        #expect(receipt["modelDirectoryPath"] as? String
            == URL(fileURLWithPath: output).standardizedFileURL.path)
    }

    @Test func expertReuseRefusesUnverifiedSourceWithoutReplacingIt() async throws {
        let built = try Self.buildSnapshot("clone-unverified")
        let output = built.dir + ".gturbo"
        defer { Self.cleanUp([built.dir, output]) }
        _ = try await K3RemoteStreamingRepacker(
            options: Self.localOptions(outputDir: output),
            source: K3LocalSnapshotSource(directory: built.dir)
        ).run()
        let manifestPath = (output as NSString).appendingPathComponent("manifest.json")
        let originalManifest = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
        try FileManager.default.removeItem(atPath:
            (output as NSString).appendingPathComponent("verified-install.json"))

        await #expect(throws: RepackError.self) {
            _ = try await K3RemoteStreamingRepacker(
                options: Self.localOptions(
                    outputDir: output,
                    trunkQuant: .int8,
                    reuseExpertsFrom: output),
                source: K3LocalSnapshotSource(directory: built.dir)
            ).run()
        }
        #expect(try Data(contentsOf: URL(fileURLWithPath: manifestPath))
            == originalManifest)
        #expect(!FileManager.default.fileExists(atPath: output + ".partial"))
    }

    // MARK: - (c) expert blob bytes are bit-identical to the source

    @Test func expertBlobBytesAreBitIdentical() async throws {
        let built = try Self.buildSnapshot("experts")
        let output = built.dir + ".gturbo"
        defer { Self.cleanUp([built.dir, output]) }
        _ = try await K3RemoteStreamingRepacker(
            options: Self.localOptions(outputDir: output),
            source: K3LocalSnapshotSource(directory: built.dir)
        ).run()

        let expertShard = built.snapshot.shardPaths[1]
        let header = try Self.shardHeader(expertShard)
        let shardData = try Self.shardBytes(expertShard)

        for layer in [1, 2, 3, 4] {
            let path = (output as NSString)
                .appendingPathComponent("packed_experts/layer_\(String(format: "%02d", layer)).bin")
            let layerData = try Data(contentsOf: URL(fileURLWithPath: path))
            #expect(layerData.count == 16 * 16_384)
            for expert in 0..<16 {
                var expected: [UInt8] = []
                for stem in ["w1", "w2", "w3"] {
                    for kind in ["weight_packed", "weight_scale"] {
                        let name = "language_model.model.layers.\(layer)"
                            + ".block_sparse_moe.experts.\(expert).\(stem).\(kind)"
                        let tensor = header[name]!
                        expected.append(contentsOf: shardData[
                            Int(tensor.absoluteOffset)..<Int(tensor.absoluteOffset + tensor.sizeBytes)])
                    }
                }
                #expect(expected.count == 13_056)
                let blobStart = expert * 16_384
                #expect(Array(layerData[blobStart..<(blobStart + 13_056)]) == expected,
                        "layer \(layer) expert \(expert) blob drift")
                #expect(layerData[(blobStart + 13_056)..<(blobStart + 16_384)].allSatisfy { $0 == 0 },
                        "layer \(layer) expert \(expert) stride padding not zero")
            }
        }
    }

    // MARK: - (d) quantized trunk dequant-matches the BF16 source

    @Test func trunkQuantizationMatchesSourceWithinGroupScale() async throws {
        let built = try Self.buildSnapshot("quant")
        let output = built.dir + ".gturbo"
        defer { Self.cleanUp([built.dir, output]) }
        let result = try await K3RemoteStreamingRepacker(
            options: Self.localOptions(outputDir: output),
            source: K3LocalSnapshotSource(directory: built.dir)
        ).run()
        let plan = result.plan

        let trunkShard = built.snapshot.shardPaths[0]
        let shardData = try Self.shardBytes(trunkShard)
        let residentData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("model_weights.bin")))

        func sourceFloats(_ tensor: K3SourceTensor) -> [Float] {
            let start = Int(tensor.absoluteOffset)
            switch tensor.dtype {
            case .bf16:
                return (0..<Int(tensor.elementCount)).map { i in
                    let lo = UInt16(shardData[start + 2 * i])
                    let hi = UInt16(shardData[start + 2 * i + 1])
                    return K3TrunkQuantizer.bf16ToFloat(lo | (hi << 8))
                }
            case .fp32:
                return (0..<Int(tensor.elementCount)).map { i in
                    var bits: UInt32 = 0
                    for j in 0..<4 { bits |= UInt32(shardData[start + 4 * i + j]) << UInt32(8 * j) }
                    return Float(bitPattern: bits)
                }
            case .u8:
                fatalError("unexpected U8 trunk tensor")
            }
        }
        func residentBytes(_ offset: UInt64, _ size: UInt64) -> [UInt8] {
            Array(residentData[Int(offset)..<Int(offset + size)])
        }
        func residentU16(_ offset: UInt64, _ count: Int) -> [UInt16] {
            (0..<count).map { i in
                UInt16(residentData[Int(offset) + 2 * i])
                    | (UInt16(residentData[Int(offset) + 2 * i + 1]) << 8)
            }
        }

        var affineChecked = 0
        var verbatimChecked = 0
        for tensor in plan.tensors {
            let entry = tensor.entry
            switch tensor.transform {
            case .affine(let bits, _):
                let source = tensor.source!
                let sourceValues = sourceFloats(source)
                let packedAll = residentBytes(entry.fileOffset, entry.sizeBytes)
                let scalesAll = residentU16(entry.scaleOffset,
                                            Int(entry.scaleSize) / 2)
                let biasesAll = residentU16(entry.biasOffset,
                                            Int(entry.biasSize) / 2)
                let columns = tensor.columns
                let rowPacked = columns * bits / 8
                let rowGroups = columns / 64
                var worstRatio: Float = 0
                for row in 0..<tensor.rows {
                    let packed = Array(packedAll[(row * rowPacked)..<((row + 1) * rowPacked)])
                    let scales = Array(scalesAll[(row * rowGroups)..<((row + 1) * rowGroups)])
                    let biases = Array(biasesAll[(row * rowGroups)..<((row + 1) * rowGroups)])
                    let dequant = K3TrunkQuantizer.dequantizeRowAffine(
                        packed: packed, scales: scales, biases: biases,
                        count: columns, bits: bits)
                    for k in 0..<columns {
                        let scale = K3TrunkQuantizer.bf16ToFloat(scales[k / 64])
                        let sourceValue = sourceValues[row * columns + k]
                        let err = abs(dequant[k] - sourceValue)
                        // Round-to-nearest int4/int8: error is at most half a
                        // quantum; clamping at the group extremes and the
                        // BF16-rounded companions keep it under one quantum.
                        #expect(err <= scale + 1e-6,
                                "\(entry.name) row \(row) col \(k): err \(err) > scale \(scale)")
                        if scale > 0 { worstRatio = max(worstRatio, err / scale) }
                    }
                }
                #expect(worstRatio <= 1.0)
                affineChecked += 1
            case .verbatim:
                let source = tensor.source!
                let expected = Array(shardData[Int(source.absoluteOffset)..<Int(
                    source.absoluteOffset + source.sizeBytes)])
                #expect(residentBytes(entry.fileOffset, entry.sizeBytes) == expected,
                        "\(entry.name) verbatim drift")
                verbatimChecked += 1
            case .widenFP32:
                let source = tensor.source!
                let expected = sourceFloats(source)
                let actual = residentBytes(entry.fileOffset, entry.sizeBytes)
                #expect(actual.count == expected.count * 4)
                for (i, value) in expected.enumerated() {
                    var bits: UInt32 = 0
                    for j in 0..<4 { bits |= UInt32(actual[4 * i + j]) << UInt32(8 * j) }
                    #expect(Float(bitPattern: bits) == value,
                            "\(entry.name) widen drift at \(i)")
                }
                verbatimChecked += 1
            case .truncateFP32(_, _, let keepCount):
                let sourceValues = sourceFloats(tensor.source!)
                let actual = residentBytes(entry.fileOffset, entry.sizeBytes)
                #expect(actual.count == keepCount * 4)
                for i in 0..<keepCount {
                    var bits: UInt32 = 0
                    for j in 0..<4 { bits |= UInt32(actual[4 * i + j]) << UInt32(8 * j) }
                    #expect(Float(bitPattern: bits) == sourceValues[i],
                            "\(entry.name) truncate drift at \(i)")
                }
                verbatimChecked += 1
            case .synthesizeZeros:
                #expect(residentBytes(entry.fileOffset, entry.sizeBytes)
                            .allSatisfy { $0 == 0 })
                verbatimChecked += 1
            }
        }
        #expect(affineChecked > 0)
        #expect(verbatimChecked > 0)
    }

    // MARK: - (7) resumable range plan, Gemma-shaped

    @Test func rangePlanIsDeterministicAndBounded() async throws {
        let built = try Self.buildSnapshot("ranges")
        defer { Self.cleanUp([built.dir]) }
        let snapshot = try await Self.loadSnapshot(built)
        let outDir = built.dir + ".out"
        let plan = try Self.makePlan(snapshot, outputDir: outDir)

        let first = try K3RangeCopyPlanner.plan(repackPlan: plan, rangeChunkBytes: 65_536)
        let second = try K3RangeCopyPlanner.plan(repackPlan: plan, rangeChunkBytes: 65_536)
        #expect(first.canonicalFingerprint == second.canonicalFingerprint)
        #expect(first.residentIndexSha256 == second.residentIndexSha256)

        // Gemma-shaped ranges: sequential ids, chunk-bounded, sorted
        // non-overlapping destinations (the shared validator already ran).
        for (index, copy) in first.coalescedCopies.enumerated() {
            #expect(copy.id == String(format: "range-%08d", index))
            #expect(copy.size <= 65_536)
            #expect(copy.size > 0)
            let sorted = copy.destinations.sorted {
                if $0.destinationPath != $1.destinationPath {
                    return $0.destinationPath < $1.destinationPath
                }
                if $0.destinationOffset != $1.destinationOffset {
                    return $0.destinationOffset < $1.destinationOffset
                }
                return $0.sourceOffset < $1.sourceOffset
            }
            #expect(copy.destinations == sorted)
        }
        #expect(first.remoteGapBytesDownloaded <= first.remoteBytesToDownload)
        #expect(first.remoteBytesToDownload > 0)

        // Expected outputs: resident + staging + one file per MoE layer.
        let rels = first.expectedOutputs.map(\.relativePath)
        #expect(rels.contains("model_weights.bin"))
        #expect(rels.contains("trunk-staging.bin"))
        #expect(rels.contains("packed_experts/layer_01.bin"))
        #expect(rels.contains("packed_experts/layer_04.bin"))
        #expect(!rels.contains("packed_experts/layer_00.bin"))  // dense layer
        for output in first.expectedOutputs where output.relativePath.hasPrefix("packed_experts/") {
            #expect(output.size == 16 * 16_384)
        }

        let trunkOnly = try K3RangeCopyPlanner.plan(
            repackPlan: plan, rangeChunkBytes: 65_536,
            includeExpertPayloads: false)
        #expect(trunkOnly.remoteBytesToDownload < first.remoteBytesToDownload)
        #expect(trunkOnly.scalarCopies.allSatisfy {
            !$0.destinationPath.contains("packed_experts/layer_")
        })
        #expect(trunkOnly.expectedOutputs.contains {
            $0.relativePath == "packed_experts/layer_01.bin"
        })
        #expect(trunkOnly.canonicalFingerprint != first.canonicalFingerprint)
    }

    // MARK: - (7) resume after cancellation

    @Test func cancellationPreservesCommittedRangesForResume() async throws {
        let built = try Self.buildSnapshot("resume")
        let output = built.dir + ".gturbo"
        defer { Self.cleanUp([built.dir, output]) }

        let seen = Mutex<Int>(0)
        let task = Task {
            try await K3RemoteStreamingRepacker(
                options: Self.localOptions(outputDir: output),
                source: K3LocalSnapshotSource(directory: built.dir)
            ).run { progress in
                guard case .copyingPayload(_, let downloaded, _) = progress,
                      downloaded > 0 else { return }
                if seen.withLock({ value -> Int in value += 1; return value }) == 3 {
                    withUnsafeCurrentTask { $0?.cancel() }
                }
            }
        }
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }

        let checkpoint = try RemoteInstallCheckpoint.load(from: output + ".resume.json")
        #expect(!checkpoint.completedRanges.isEmpty)

        let result = try await K3RemoteStreamingRepacker(
            options: Self.localOptions(outputDir: output, resume: true),
            source: K3LocalSnapshotSource(directory: built.dir)
        ).run()
        #expect(result.reusedBytes > 0)
        #expect(result.downloadedThisRunBytes < result.remoteBytesToDownload)
        #expect(FileManager.default.fileExists(
            atPath: (output as NSString).appendingPathComponent("manifest.json")))
        // The resumed bundle still decodes + cross-validates.
        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("manifest.json")))
        let manifest = try GTurboManifestCodecV2.decode(manifestData)
        let layoutData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("packed_experts/layout.json")))
        let layout = try GTurboPackedExpertsLayoutCodecV2.decode(layoutData)
        try GTurboV2StructuralValidator.crossValidate(manifest: manifest, layout: layout)
    }

    /// Resume state 2: the copy phase completed but the run died during
    /// quantization (staging still present); the resume must skip the copy,
    /// re-run phase 2 idempotently, and promote a valid bundle.
    @Test func resumeAfterCopyCompletesRerunsQuantization() async throws {
        let built = try Self.buildSnapshot("resume-quant")
        let output = built.dir + ".gturbo"
        defer { Self.cleanUp([built.dir, output]) }

        let task = Task {
            try await K3RemoteStreamingRepacker(
                options: Self.localOptions(outputDir: output),
                source: K3LocalSnapshotSource(directory: built.dir)
            ).run { progress in
                guard case .copyingPayload(_, let downloaded, let total) = progress,
                      downloaded == total else { return }
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }
        // All ranges committed; staging file still present.
        let checkpoint = try RemoteInstallCheckpoint.load(from: output + ".resume.json")
        #expect(!checkpoint.completedRanges.isEmpty)
        #expect(FileManager.default.fileExists(
            atPath: (output + ".partial" as NSString)
                .appendingPathComponent("trunk-staging.bin")))

        let result = try await K3RemoteStreamingRepacker(
            options: Self.localOptions(outputDir: output, resume: true),
            source: K3LocalSnapshotSource(directory: built.dir)
        ).run()
        #expect(result.reusedBytes == result.remoteBytesToDownload)
        #expect(result.downloadedThisRunBytes == 0)
        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("manifest.json")))
        let manifest = try GTurboManifestCodecV2.decode(manifestData)
        #expect(manifest.numLayers == 5)
    }

    // MARK: - K3ArchInfo parsing

    @Test func archInfoParsesSyntheticConfig() async throws {
        let built = try Self.buildSnapshot("arch")
        defer { Self.cleanUp([built.dir]) }
        let arch = try K3ArchInfo.load(configPath: built.snapshot.configPath)
        #expect(arch.hiddenSize == 256)
        #expect(arch.vocabSize == 512)
        #expect(arch.numLayers == 5)
        #expect(arch.denseLayers == [1])
        #expect(arch.kdaLayers == [1, 2, 3, 5])
        #expect(arch.fullAttnLayers == [4])
        #expect(arch.kdaNumHeads == 4)
        #expect(arch.kdaHeadDim == 32)
        #expect(arch.kdaConvWidth == 4)
        #expect(arch.kdaGateLowerBound == -5.0)
        #expect(arch.kdaFullRankOutputGate)
        #expect(arch.mlaNumHeads == 4)
        #expect(arch.mlaQLoraRank == 64)
        #expect(arch.mlaKVLoraRank == 64)
        #expect(arch.mlaQKNopeHeadDim == 16)
        #expect(arch.mlaQKRopeHeadDim == 8)
        #expect(arch.mlaVHeadDim == 16)
        #expect(arch.mlaOutputGate)
        #expect(arch.attnResBlockSize == 2)
        #expect(arch.moeNumExperts == 16)
        #expect(arch.moeTopKExperts == 4)
        #expect(arch.moeLatentBottleneckSize == 128)
        #expect(arch.moeExpertIntermediateSize == 64)
        #expect(arch.moeSharedExpertIntermediateSize == 64)
        #expect(arch.situGLUGateBeta == 4.0)
        #expect(arch.situGLUUpBeta == 25.0)
        #expect(arch.routerRenormalize)
        #expect(arch.routerCorrectionBias)
        #expect(arch.hiddenActivation == "situ_glu")
        #expect(arch.rmsNormEpsilon == 1e-5)
        #expect(!arch.tieWordEmbeddings)
        #expect(arch.denseMLPIntermediateSize == 256)
    }

    @Test func archInfoRefusesDefaults() throws {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-k3arch-\(UUID().uuidString)")
        defer { Self.cleanUp([dir]) }
        let snapshot = try K3SyntheticCheckpoint.build(at:
            (dir as NSString).appendingPathComponent("snapshot"))
        let base = try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: snapshot.configPath)))
            as! [String: Any]

        func write(_ root: [String: Any], _ tag: String) throws -> String {
            let path = (dir as NSString).appendingPathComponent("\(tag).json")
            try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
                .write(to: URL(fileURLWithPath: path))
            return path
        }

        // Missing text_config key -> throw.
        var missing = base
        var tc = missing["text_config"] as! [String: Any]
        tc.removeValue(forKey: "num_hidden_layers")
        missing["text_config"] = tc
        #expect(throws: RepackError.self) {
            try K3ArchInfo.load(configPath: write(missing, "missing"))
        }

        // Missing linear_attn_config -> throw.
        var noLinear = base
        var tc2 = noLinear["text_config"] as! [String: Any]
        tc2.removeValue(forKey: "linear_attn_config")
        noLinear["text_config"] = tc2
        #expect(throws: RepackError.self) {
            try K3ArchInfo.load(configPath: write(noLinear, "nolinear"))
        }

        // Unsorted layer list -> throw.
        var unsorted = base
        var tc3 = unsorted["text_config"] as! [String: Any]
        var la3 = tc3["linear_attn_config"] as! [String: Any]
        la3["kda_layers"] = [2, 1, 3, 5]
        tc3["linear_attn_config"] = la3
        unsorted["text_config"] = tc3
        #expect(throws: RepackError.self) {
            try K3ArchInfo.load(configPath: write(unsorted, "unsorted"))
        }

        // Non-SiTU activation -> throw.
        var act = base
        var tc4 = act["text_config"] as! [String: Any]
        tc4["hidden_act"] = "gelu"
        act["text_config"] = tc4
        #expect(throws: RepackError.self) {
            try K3ArchInfo.load(configPath: write(act, "act"))
        }
    }

    // MARK: - Quantizer unit checks (bit-parity with the house reference is
    // asserted in TurboFieldfareTestsCore, which can see both modules)

    @Test func quantizerRoundTripAndConstantGroup() throws {
        var rng = K3SyntheticPRNG(seed: 0x51)
        let row = (0..<256).map { _ in rng.uniform(-0.06, 0.06) }
        for bits in [4, 8] {
            let q = K3TrunkQuantizer.quantizeRowAffine(row, bits: bits)
            let deq = K3TrunkQuantizer.dequantizeRowAffine(
                packed: q.packed, scales: q.scales, biases: q.biases,
                count: row.count, bits: bits)
            for g in 0..<row.count / 64 {
                let scale = K3TrunkQuantizer.bf16ToFloat(q.scales[g])
                for k in 0..<64 {
                    #expect(abs(deq[g * 64 + k] - row[g * 64 + k]) <= scale + 1e-6)
                }
            }
        }
        // Constant group: exact reconstruction through scale=1/bias=value.
        let flat = [Float](repeating: 3.25, count: 128)
        let q = K3TrunkQuantizer.quantizeRowAffine(flat, bits: 4)
        let deq = K3TrunkQuantizer.dequantizeRowAffine(
            packed: q.packed, scales: q.scales, biases: q.biases,
            count: flat.count, bits: 4)
        #expect(deq == flat)
        // nibble order: low nibble carries the even element. ramp has min at
        // index 0 (q=0) and max at index 1 (q=15 after clamping).
        var ramp = [Float](repeating: 0, count: 64)
        ramp[0] = -1
        ramp[1] = 1
        let q2 = K3TrunkQuantizer.quantizeRowAffine(ramp, bits: 4)
        #expect(q2.packed[0] & 0x0F == 0)
        #expect(q2.packed[0] >> 4 == 15)
        let scale = K3TrunkQuantizer.bf16ToFloat(q2.scales[0])
        let bias = K3TrunkQuantizer.bf16ToFloat(q2.biases[0])
        #expect(bias == -1)
    }
}
