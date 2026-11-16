import Foundation
import Testing

@testable import TinyTitanRepackCore
@testable import TinyTitanCore

@Suite struct LocalSnapshotRepackerTests {
    @Test func importsMTPWithExplicitModelIdentityAndReceipt() async throws {
        let root = Self.temporaryRoot("local-mtp-import")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.ssdai")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwenMTP(at: snapshot)

        let result = try await RemoteStreamingRepacker.runLocalSnapshot(
            options: LocalSnapshotRepackOptions(
                inputSnapshotDir: snapshot,
                outputDir: output,
                modelID: "ornith-1.5-35b-a3b-mtp-4bit",
                minFreeReserveBytes: 0))

        #expect(result.outputDir == output)
        #expect(result.rangeRequestCount == 0)
        #expect(result.downloadedThisRunBytes == result.remoteBytesToDownload)
        let manifestData = try Data(
            contentsOf: URL(
                fileURLWithPath: (output as NSString).appendingPathComponent("manifest.json")))
        let manifestObject = try JSONSerialization.jsonObject(with: manifestData)
        let manifest = try #require(manifestObject as? [String: Any])
        #expect(
            manifest["modelID"] as? String
                == "ornith-1.5-35b-a3b-mtp-4bit")
        #expect(
            (manifest["arch"] as? [String: Any])?["family"] as? String
                == "qwen36_mtp")
        #expect(
            try Posix.entryKind(
                (output as NSString)
                    .appendingPathComponent("verified-install.json")) == .regular)
        #expect(
            try Posix.entryKind(
                (output as NSString)
                    .appendingPathComponent("packed_experts/layer_00.bin")) == .regular)
    }

    @Test func rejectsUnsafeShardPathBeforeCopying() throws {
        let root = Self.temporaryRoot("local-mtp-unsafe")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwenMTP(at: snapshot)
        let indexPath = (snapshot as NSString)
            .appendingPathComponent("model.safetensors.index.json")
        let data = try Data(contentsOf: URL(fileURLWithPath: indexPath))
        let indexObject = try JSONSerialization.jsonObject(with: data)
        var index = try #require(indexObject as? [String: Any])
        var weightMap = try #require(index["weight_map"] as? [String: String])
        for name in weightMap.keys { weightMap[name] = "../outside.safetensors" }
        index["weight_map"] = weightMap
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: indexPath))

        #expect(throws: RepackError.self) {
            _ = try LocalSnapshotLoader.load(directory: snapshot)
        }
    }

    /// Shared with `IndexLoaderSizeTests` below, which needs the same scratch
    /// directory convention.
    static func temporaryRoot(_ tag: String) -> String {
        let base = (FileManager.default.currentDirectoryPath as NSString)
            .appendingPathComponent(".build/test-artifacts")
        try? FileManager.default.createDirectory(
            atPath: base, withIntermediateDirectories: true)
        let path = (base as NSString)
            .appendingPathComponent("\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }
}

/// The index bound is not a style preference: a real checkpoint sits close to
/// it. These pin both halves of the contract so the limit can never be quietly
/// lowered back under a shipped model.
@Suite struct IndexLoaderSizeTests {
    private static let indexName = "model.safetensors.index.json"

    @Test func loadsAnIndexLargerThanTheBoundThatRefusedKATCoder() throws {
        let root = LocalSnapshotRepackerTests.temporaryRoot("oversized-index")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        _ = try SyntheticSnapshot.buildQwen(at: snapshot)

        let indexPath = (snapshot as NSString).appendingPathComponent(Self.indexName)
        let baseline = try Self.entryCount(at: indexPath)
        // A synthetic entry costs about 39 bytes; 6 MiB clears the old 4 MiB
        // bound with room to spare.
        let target = 6 * 1024 * 1024
        let injected = Int(UInt64(target) / 24)
        let size = try Self.inflateIndex(at: indexPath, extraEntries: injected)
        // The bound this file used to carry, and the reason the test exists:
        // KAT-Coder-V2.5-Dev's index is 9.7 MiB and was refused by it.
        #expect(
            size > 4 * 1024 * 1024,
            "the fixture must exceed the old 4 MiB bound, was \(size) bytes")

        let loaded = try LocalSnapshotLoader.load(directory: snapshot)
        #expect(
            loaded.metadata.weightMap.count == baseline + injected,
            "every injected entry survives the read")
        #expect(loaded.metadata.shardFilenames.count == 1)
    }

    @Test func refusesAnIndexLargerThanItsOwnBound() throws {
        let root = LocalSnapshotRepackerTests.temporaryRoot("huge-index")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        _ = try SyntheticSnapshot.buildQwen(at: snapshot)

        // A test-sized stand-in for the bound. The contract is proportional --
        // a file past the cap is refused -- and proving it against all 64 MiB
        // would write a 64 MiB fixture for no extra assurance. The declared
        // bound's own adequacy is pinned by `boundLeavesRoom...` below.
        let smallBound = IndexLoader.maximumIndexBytes / 64
        let indexPath = (snapshot as NSString).appendingPathComponent(Self.indexName)
        // Between the two bounds: past the test-sized one, still under the real
        // one, so the same file proves both directions.
        let extra = Int(smallBound) * 12 / 39
        let size = try Self.inflateIndex(at: indexPath, extraEntries: extra)
        #expect(size > smallBound, "the fixture must exceed the test bound")
        #expect(
            size < IndexLoader.maximumIndexBytes,
            "the fixture must stay under the real bound")

        #expect(throws: RepackError.self) {
            _ = try Posix.readBoundedData(indexPath, maximumBytes: smallBound)
        }
        // The real bound accepts what the old one refused.
        #expect(
            try Posix.readBoundedData(
                indexPath,
                maximumBytes: IndexLoader.maximumIndexBytes
            ).count
                == Int(size))
    }

    /// Keeps the declared bound above the largest index a shipped checkpoint
    /// legitimately produces. This is the half that would have caught the
    /// original bug before a 69 GB conversion ran into it.
    @Test func boundLeavesRoomForAShippedCheckpointsIndex() {
        let katCoderObserved: UInt64 = 9_665_024  // KAT-Coder-V2.5-Dev 4-bit
        #expect(
            IndexLoader.maximumIndexBytes >= 4 * katCoderObserved,
            "bound \(IndexLoader.maximumIndexBytes) is too close to a real index")
    }

    /// Rewrite the index with `extraEntries` synthetic `weight_map` entries,
    /// all pointing at the snapshot's real shard so the shard-path guard still
    /// passes. Returns the resulting file size in bytes.
    ///
    /// Written line by line on purpose: the in-memory variant (`JSONSerialization`
    /// over the inflated dictionary) allocates several times the file size, and
    /// the fixtures here only need to be valid JSON with a large `weight_map`.
    @discardableResult
    private static func inflateIndex(
        at path: String,
        extraEntries: Int
    ) throws -> UInt64 {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let root = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let map = try #require(root["weight_map"] as? [String: String])
        let shard = try #require(map.values.first)

        FileManager.default.createFile(atPath: path, contents: nil)
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
        defer { try? handle.close() }
        func write(_ text: String) throws {
            try handle.write(contentsOf: Data(text.utf8))
        }
        try write("{\"metadata\":{\"format\":\"mlx\"},\"weight_map\":{")
        var first = true
        for (key, value) in map {
            try write("\(first ? "" : ",")\"\(key)\":\"\(value)\"")
            first = false
        }
        for i in 0..<extraEntries {
            try write(",\"model.language_model.layers.0.injected\(i).weight\":\"\(shard)\"")
        }
        try write("}}")
        try handle.close()
        let attributes = try FileManager.default
            .attributesOfItem(atPath: path)
        return try #require(attributes[.size] as? UInt64)
    }

    /// How many `weight_map` entries the index already holds.
    private static func entryCount(at path: String) throws -> Int {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let root = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try #require(root["weight_map"] as? [String: String]).count
    }
}
