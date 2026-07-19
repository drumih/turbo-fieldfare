import Foundation
import Darwin

/// Creates the resident LM `.bin` file and writes its bounded binary index.
enum ResidentWriter {
    private static let maxIndexBytes = 1_048_575

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

    private static func writeIndex(plan: ResidentFilePlan,
                                   fd: Int32,
                                   audit: RepackAudit) throws {
        // The header region size is bounded by indexSize (a few KB); allocate
        // it once on the heap and pwrite once. Well under any reasonable budget.
        let idxBytes = Int(plan.indexSize)
        let idxBuf = UnsafeMutableRawBufferPointer.allocate(byteCount: idxBytes,
                                                            alignment: 16_384)
        defer { idxBuf.deallocate() }
        idxBuf.initializeMemory(as: UInt8.self, repeating: 0)
        if idxBytes > audit.largestScratchBytes {
            // The index page is bounded by the planner; fail instead of
            // allowing this one-shot allocation to reach 1 MiB.
            if idxBytes > maxIndexBytes {
                throw RepackError.scratchExceeded(requested: idxBytes,
                                                  limit: maxIndexBytes)
            }
            audit.largestScratchBytes = idxBytes
        }
        GTurboBinary.writeIndexHeader(into: idxBuf.baseAddress!,
                                      indexSize: plan.indexSize,
                                      residentSize: plan.residentSize,
                                      entryCount: UInt64(plan.entries.count))
        let entriesBase = GTurboBinary.indexHeaderBytes
        let stringTableBase = entriesBase + plan.entries.count * GTurboBinary.indexEntryBytes
        for i in 0..<plan.entries.count {
            let dst = idxBuf.baseAddress!.advanced(by: entriesBase + i * GTurboBinary.indexEntryBytes)
            let nameOff = UInt32(stringTableBase) + plan.stringTableOffsets[i]
            GTurboBinary.writeIndexEntry(into: dst, entry: plan.entries[i], nameOffset: nameOff)
        }
        plan.stringTable.withUnsafeBufferPointer { src in
            let dst = idxBuf.baseAddress!.advanced(by: stringTableBase)
            memcpy(dst, src.baseAddress!, src.count)
        }
        try Posix.pwriteAll(fd: fd, path: plan.path,
                            buf: idxBuf.baseAddress!, count: idxBytes, offset: 0)
        audit.recordWrite(bytes: idxBytes)
    }

}
