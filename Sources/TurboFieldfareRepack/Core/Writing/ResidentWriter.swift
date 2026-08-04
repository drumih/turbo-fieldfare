import Foundation
import Darwin
import TurboFieldfareFormat

/// Writes the resident LM `.bin` file (`model_weights.bin`).
/// from a planned layout + a shard registry. Shards are mapped one at a time;
/// per-tensor writes go straight from mmap'd source memory through pwrite
/// tiles, with `madvise(MADV_DONTNEED)` after each tile.
enum ResidentWriter {

    static func write(plan: ResidentFilePlan,
                             shardsByPath: inout [String: MmapHandle],
                             audit: RepackAudit) throws -> RepackAudit.OutputFile {
        // 1. Create + size the output file.
        try Posix.mkdirP(((plan.path as NSString).deletingLastPathComponent))
        let fd = try Posix.openCreateRW(plan.path)
        defer { close(fd) }
        try Posix.ftruncate(fd, path: plan.path, size: plan.totalSize)

        // 2. Write the binary index page: header + entries + string table.
        try writeIndex(plan: plan, fd: fd, audit: audit)

        // 3. For each entry, copy the weight + scales + biases from the source
        // shards in tile-bounded pwrites.
        for e in plan.entries {
            try copyOne(srcTensor: e.sourceWeight, dstFd: fd, dstPath: plan.path,
                        dstOffset: e.fileOffset, sizeBytes: e.sizeBytes,
                        shardsByPath: &shardsByPath, audit: audit)
            if let scales = e.sourceScales {
                try copyOne(srcTensor: scales, dstFd: fd, dstPath: plan.path,
                            dstOffset: e.scaleOffset, sizeBytes: e.scaleSize,
                            shardsByPath: &shardsByPath, audit: audit)
            }
            if let biases = e.sourceBiases {
                try copyOne(srcTensor: biases, dstFd: fd, dstPath: plan.path,
                            dstOffset: e.biasOffset, sizeBytes: e.biasSize,
                            shardsByPath: &shardsByPath, audit: audit)
            }
        }

        try Posix.fsync(fd, path: plan.path)
        let size = try Posix.fileSize(fd: fd, path: plan.path)
        // 4. Hash the finished file by streaming.
        let sha = try WriterCore.hashEntireFile(path: plan.path, size: size, audit: audit)
        let rel = (plan.path as NSString).lastPathComponent
        let outFile = RepackAudit.OutputFile(relativePath: rel, size: size, sha256: sha)
        audit.outputFiles.append(outFile)
        return outFile
    }

    static func createAndWriteIndex(plan: ResidentFilePlan,
                                           audit: RepackAudit) throws -> Int32 {
        try Posix.mkdirP(((plan.path as NSString).deletingLastPathComponent))
        let fd = try Posix.openCreateRW(plan.path)
        do {
            try Posix.ftruncate(fd, path: plan.path, size: plan.totalSize)
            try writeIndex(plan: plan, fd: fd, audit: audit)
            return fd
        } catch {
            close(fd)
            throw error
        }
    }

    static func encodeIndex(plan: ResidentFilePlan) throws -> Data {
        guard plan.indexSize <= UInt64(Int.max),
              plan.indexSize <= GTurboFormatV1.residentIndexMaxBytes else {
            throw RepackError.configurationInvalid(
                detail: "resident index size \(plan.indexSize) exceeds v1 metadata cap")
        }
        let idxBytes = Int(plan.indexSize)
        guard idxBytes <= BoundedScratch.defaultLimitBytes else {
            throw RepackError.scratchExceeded(requested: idxBytes,
                                              limit: BoundedScratch.defaultLimitBytes)
        }
        guard plan.entries.count == plan.stringTableOffsets.count else {
            throw RepackError.configurationInvalid(
                detail: "resident index entry/string offset count mismatch")
        }
        let (entryTableBytes, tableOverflow) = plan.entries.count
            .multipliedReportingOverflow(by: GTurboBinary.indexEntryBytes)
        let (stringTableBase, baseOverflow) = GTurboBinary.indexHeaderBytes
            .addingReportingOverflow(entryTableBytes)
        guard !tableOverflow, !baseOverflow,
              stringTableBase <= idxBytes,
              plan.stringTable.count <= idxBytes - stringTableBase,
              stringTableBase <= Int(UInt32.max) else {
            throw RepackError.configurationInvalid(
                detail: "resident index table exceeds declared index region")
        }
        for (index, entry) in plan.entries.enumerated() {
            guard entry.name.utf8.count <= Int(UInt16.max),
                  entry.logicalShape4.count == 4,
                  GTurboFormatV1.DType(rawValue: entry.dtype) != nil else {
                throw RepackError.configurationInvalid(
                    detail: "resident index entry \(index) is not representable")
            }
            let absoluteNameOffset = UInt64(stringTableBase)
                + UInt64(plan.stringTableOffsets[index])
            guard absoluteNameOffset <= UInt64(UInt32.max),
                  absoluteNameOffset + UInt64(entry.name.utf8.count) <= UInt64(idxBytes) else {
                throw RepackError.configurationInvalid(
                    detail: "resident index entry \(index) name range is invalid")
            }
        }
        let idxBuf = UnsafeMutableRawBufferPointer.allocate(byteCount: idxBytes,
                                                            alignment: 16_384)
        defer { idxBuf.deallocate() }
        idxBuf.initializeMemory(as: UInt8.self, repeating: 0)
        GTurboBinary.writeIndexHeader(into: idxBuf.baseAddress!,
                                      indexSize: plan.indexSize,
                                      residentSize: plan.residentSize,
                                      entryCount: UInt64(plan.entries.count))
        let entriesBase = GTurboBinary.indexHeaderBytes
        for i in 0..<plan.entries.count {
            let dst = idxBuf.baseAddress!.advanced(by: entriesBase + i * GTurboBinary.indexEntryBytes)
            let nameOff = UInt32(stringTableBase) + plan.stringTableOffsets[i]
            GTurboBinary.writeIndexEntry(into: dst, entry: plan.entries[i], nameOffset: nameOff)
        }
        plan.stringTable.withUnsafeBufferPointer { src in
            let dst = idxBuf.baseAddress!.advanced(by: stringTableBase)
            memcpy(dst, src.baseAddress!, src.count)
        }
        let data = Data(bytes: idxBuf.baseAddress!, count: idxBytes)
        do {
            try data.withUnsafeBytes { raw in
                let header = try GTurboResidentIndexCodec.decodeHeader(raw)
                _ = try GTurboResidentIndexCodec.decodeRegion(raw, header: header)
            }
        } catch {
            throw RepackError.configurationInvalid(
                detail: "resident index encoding invalid: \(error)")
        }
        return data
    }

    private static func writeIndex(plan: ResidentFilePlan,
                                   fd: Int32,
                                   audit: RepackAudit) throws {
        let data = try encodeIndex(plan: plan)
        let idxBytes = data.count
        if idxBytes > audit.largestScratchBytes {
            audit.largestScratchBytes = idxBytes
        }
        try data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            try Posix.pwriteAll(fd: fd, path: plan.path,
                                buf: base, count: idxBytes, offset: 0)
        }
        audit.recordWrite(bytes: idxBytes)
    }

    private static func copyOne(srcTensor: SourceTensor,
                                dstFd: Int32, dstPath: String, dstOffset: UInt64,
                                sizeBytes: UInt64,
                                shardsByPath: inout [String: MmapHandle],
                                audit: RepackAudit) throws {
        let shard = try mappedShard(path: srcTensor.shardPath, shardsByPath: &shardsByPath)
        try WriterCore.pwriteTensorRegion(srcShard: shard,
                                          srcAbsoluteOffset: srcTensor.absoluteOffset,
                                          size: sizeBytes,
                                          dstFd: dstFd, dstPath: dstPath,
                                          dstOffset: dstOffset,
                                          audit: audit)
    }

    private static func mappedShard(path: String,
                                    shardsByPath: inout [String: MmapHandle]) throws -> MmapHandle {
        if let h = shardsByPath[path] { return h }
        let h = try MmapHandle(path: path)
        shardsByPath[path] = h
        return h
    }
}
