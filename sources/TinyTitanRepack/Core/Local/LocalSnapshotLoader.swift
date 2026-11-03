import Foundation

public struct LocalSnapshotRepackOptions: Sendable {
    public let inputSnapshotDir: String
    public let outputDir: String
    public let modelID: String
    /// Import only the `mtp.*` namespace as a one-layer draft head, the way
    /// `--model <name>-mtp` does for a pinned remote source. The planner
    /// already knows how to split the two namespaces; without this the local
    /// path can only ever produce the target.
    public let draftHead: Bool
    /// Hardlink `ngram_table.bin` from the snapshot instead of copying it.
    /// The table is 95 GiB and identical in every quantization of a model, so
    /// a copy is 95 GiB spent to hold the same bytes twice. Linking is safe
    /// here in a way it would not be across arbitrary installs: the file being
    /// linked *is* the one that would have been copied.
    public let shareNgramTable: Bool
    /// Delete each source shard as the last copy that reads it commits.
    ///
    /// The snapshot of a 118B model is ~125 GB and its install is another
    /// ~125 GB; on a host that cannot hold both, this lets the install grow
    /// into the space the source is releasing. It is opt-in and one-way: an
    /// interrupted run leaves a partly consumed snapshot that cannot be
    /// repacked again, which is why the default is off.
    public let consumeSnapshot: Bool
    public let overwrite: Bool
    public let minFreeReserveBytes: UInt64
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int

    public init(
        inputSnapshotDir: String,
        outputDir: String,
        modelID: String,
        draftHead: Bool = false,
        shareNgramTable: Bool = false,
        consumeSnapshot: Bool = false,
        overwrite: Bool = false,
        minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
        rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
        writeTileBytes: Int = WriterCore.tileBytes
    ) {
        self.inputSnapshotDir = inputSnapshotDir
        self.outputDir = outputDir
        self.modelID = modelID
        self.draftHead = draftHead
        self.shareNgramTable = shareNgramTable
        self.consumeSnapshot = consumeSnapshot
        self.overwrite = overwrite
        self.minFreeReserveBytes = minFreeReserveBytes
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
    }
}

struct LocalSnapshot {
    let metadata: IndexLoader.SourceMetadata
    let arch: ArchInfo
    let shardHeaders: [Safetensors.Header]
}

enum LocalSnapshotLoader {
    static func load(directory: String, draftHead: Bool = false) throws -> LocalSnapshot {
        let root = URL(fileURLWithPath: directory).standardizedFileURL.path
        guard try Posix.entryKind(root) == .directory else {
            throw RepackError.configurationInvalid(
                detail: "local snapshot is not a directory: \(root)")
        }
        let metadata = try IndexLoader.load(snapshotDir: root)
        var arch = try ArchInfo.load(configPath: metadata.configPath)
        if draftHead {
            arch = try ArchInfo.draftHead(
                from: arch,
                configPath: metadata.configPath)
        }
        let headers = try metadata.shardFilenames.map { shard in
            try loadHeader(shard: shard, directory: root)
        }
        return LocalSnapshot(
            metadata: metadata,
            arch: arch,
            shardHeaders: headers)
    }

    private static func loadHeader(
        shard: String,
        directory: String
    ) throws -> Safetensors.Header {
        guard !shard.isEmpty,
            !shard.hasPrefix("/"),
            !shard.contains(".."),
            !shard.contains("/"),
            !shard.contains("\\")
        else {
            throw RepackError.configurationInvalid(
                detail: "unsafe local snapshot shard path \(shard)")
        }
        let rootURL = URL(fileURLWithPath: directory, isDirectory: true)
        let path = URL(fileURLWithPath: shard, relativeTo: rootURL)
            .standardizedFileURL.path
        guard path.hasPrefix(directory + "/") else {
            throw RepackError.configurationInvalid(
                detail: "local snapshot shard escapes its directory: \(shard)")
        }
        let descriptor = try Posix.openReadNoFollow(path)
        defer { close(descriptor) }
        let fileSize = try Posix.fileSize(fd: descriptor, path: path)
        guard fileSize >= 8 else {
            throw RepackError.safetensorsHeaderInvalid(
                path: shard,
                detail: "short header prefix")
        }
        var prefix = [UInt8](repeating: 0, count: 8)
        try prefix.withUnsafeMutableBytes { bytes in
            guard let base = bytes.baseAddress else { return }
            try Posix.preadAll(
                fd: descriptor,
                path: path,
                buf: base,
                count: 8,
                offset: 0)
        }
        var headerSize: UInt64 = 0
        for index in 0..<8 {
            headerSize |= UInt64(prefix[index]) << UInt64(index * 8)
        }
        guard headerSize <= Safetensors.maxHeaderBytes,
            headerSize <= fileSize - 8,
            headerSize <= UInt64(Int.max)
        else {
            throw RepackError.safetensorsHeaderTooLarge(
                path: shard,
                size: headerSize)
        }
        var header = Data(count: Int(headerSize))
        try header.withUnsafeMutableBytes { bytes in
            guard let base = bytes.baseAddress else { return }
            try Posix.preadAll(
                fd: descriptor,
                path: path,
                buf: base,
                count: Int(headerSize),
                offset: 8)
        }
        return try Safetensors.parseHeaderBytes(
            path: shard,
            fileSize: fileSize,
            headerBytes: header)
    }
}
