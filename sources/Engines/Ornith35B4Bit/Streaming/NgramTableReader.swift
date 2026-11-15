import Foundation

/// Row gather over Qwen3.8-Flash-Next's n-gram embedding table.
///
/// The table is ~320M rows of `embedDim / heads` fp16 values — 102 GB at the
/// production geometry, far past anything that could be resident. It is read
/// the way routed experts are: bypassing the page cache, so the declared
/// memory budget stays the real footprint.
///
/// The gather is much friendlier than an expert miss, though, and the
/// difference is worth stating because it shapes how this should be scheduled:
/// a row id depends only on token ids (see `PLEHash`), never on router output
/// or any activation. The rows for token N+1 are therefore knowable the moment
/// token N is sampled, so this read can be issued entirely off the decode
/// critical path — where an expert miss cannot be, because routing is not
/// known until the layer runs.
///
/// unchecked-invariant: immutable after `init`. The descriptor and geometry
/// are never mutated afterwards, and `pread` does not move a shared file
/// offset, so concurrent gathers only read.
public final class NgramTableReader: @unchecked Sendable {
    public enum Failure: Error, CustomStringConvertible {
        case openFailed(path: String, errno: Int32)
        case sizeMismatch(path: String, expected: UInt64, actual: UInt64)
        case invalidGeometry(rowDim: Int, rowCount: UInt64)
        case rowOutOfRange(row: UInt32, rowCount: UInt64)
        case readFailed(row: UInt32, errno: Int32)
        case shortRead(row: UInt32, expected: Int, got: Int)
        case badHeader(detail: String)

        public var description: String {
            switch self {
            case .openFailed(let p, let e):
                return "n-gram table open(\(p)) failed: \(String(cString: strerror(e)))"
            case .sizeMismatch(let p, let expected, let actual):
                return "n-gram table \(p) is \(actual) bytes; expected at least "
                    + "\(expected) and at most 1 MiB of alignment padding past "
                    + "it. The table does not match the manifest's geometry"
            case .invalidGeometry(let rowDim, let count):
                return "n-gram table geometry is not usable: rowDim \(rowDim), "
                    + "rowCount \(count). Both come from ple_constants.json, so "
                    + "the sidecar is corrupt or built for another model"
            case .rowOutOfRange(let row, let count):
                return "n-gram row \(row) is outside the table's \(count) rows"
            case .readFailed(let row, let e):
                return "n-gram row \(row) read failed: \(String(cString: strerror(e)))"
            case .shortRead(let row, let expected, let got):
                return "n-gram row \(row) short read: expected \(expected), got \(got)"
            case .badHeader(let detail):
                return "n-gram table header is not usable: \(detail)"
            }
        }
    }

    private let fd: Int32
    public let path: String
    /// Values per row: `embedDim / totalHeads` (160 in production).
    public let rowDim: Int
    /// Bytes one **decoded** row occupies in a gather destination: always
    /// `rowDim * 2`. Callers use this as the destination stride; the bytes a
    /// 4-bit row occupies on disk are smaller and stay private.
    public let rowBytes: Int
    /// Bytes one row occupies in the file.
    private let storedRowBytes: Int
    public let rowCount: UInt64
    /// How the rows are stored. fp16 is the shipped table; the 4-bit form is
    /// `tools/quantize_ngram_table.py`'s output (100 B a row at the production
    /// geometry instead of 320).
    public enum Format: Sendable, Equatable {
        case fp16
        case affine4Group32
    }
    public let format: Format
    /// Where row 0 starts: 0 for the fp16 table, the header size for the 4-bit.
    private let dataOffset: UInt64
    static let q4Magic: UInt32 = 0x3451_5454  // "TTQ4"
    static let q4Version: UInt32 = 1
    static let q4HeaderBytes: UInt64 = 40

    /// - Parameters:
    ///   - rowCount: expected rows, from the manifest's PLE geometry. The file
    ///     size is checked against it: a table that disagrees would otherwise
    ///     be read at plausible but wrong offsets, returning real embedding
    ///     data from the wrong rows.
    ///   - bypassCache: keep the gather out of the page cache. Default on,
    ///     because a bounded footprint is the point of streaming.
    public init(
        path: String, rowDim: Int, rowCount: UInt64,
        bypassCache: Bool = true,
        format: Format = .fp16
    ) throws {
        // Thrown, not a precondition: both values come from `ple_constants.json`,
        // which is a file on disk rather than a fact of the build. A corrupt or
        // foreign sidecar used to abort the process here instead of reporting
        // itself.
        guard rowDim > 0, rowCount > 0 else {
            throw Failure.invalidGeometry(rowDim: rowDim, rowCount: rowCount)
        }
        let opened = open(path, O_RDONLY)
        guard opened >= 0 else {
            throw Failure.openFailed(path: path, errno: errno)
        }
        if bypassCache { _ = fcntl(opened, F_NOCACHE, 1) }
        var st = stat()
        guard fstat(opened, &st) == 0 else {
            let code = errno
            close(opened)
            throw Failure.openFailed(path: path, errno: code)
        }
        // Checked, not wrapping. `rowCount &* bytes` could wrap to a small
        // number, which collapses `expected` and lets the size guard below
        // accept a table built for a completely different geometry.
        let bytes: Int
        switch format {
        case .fp16:
            let (value, overflow) = rowDim.multipliedReportingOverflow(
                by: MemoryLayout<Float16>.stride)
            guard !overflow, value > 0 else {
                close(opened)
                throw Failure.invalidGeometry(rowDim: rowDim, rowCount: rowCount)
            }
            bytes = value
        case .affine4Group32:
            guard rowDim > 0, rowDim % 32 == 0 else {
                close(opened)
                throw Failure.invalidGeometry(rowDim: rowDim, rowCount: rowCount)
            }
            let (value, overflow) = (rowDim / 2).addingReportingOverflow(
                (rowDim / 32) * 4)
            guard !overflow, value > 0 else {
                close(opened)
                throw Failure.invalidGeometry(rowDim: rowDim, rowCount: rowCount)
            }
            bytes = value
        }
        let storedBytes = bytes
        let headerBytes: UInt64 = format == .fp16 ? 0 : Self.q4HeaderBytes
        let (bodyBytes, sizeOverflow) = rowCount.multipliedReportingOverflow(
            by: UInt64(bytes))
        guard !sizeOverflow else {
            close(opened)
            throw Failure.invalidGeometry(rowDim: rowDim, rowCount: rowCount)
        }
        let (expected, headerOverflow) = headerBytes.addingReportingOverflow(bodyBytes)
        guard !headerOverflow else {
            close(opened)
            throw Failure.invalidGeometry(rowDim: rowDim, rowCount: rowCount)
        }
        // The checkpoint pads the table past its last addressable row (the
        // shipped one rounds up to a 512-row boundary), so an exact size is
        // the wrong check. What this must still catch is a table built for a
        // different geometry, and that differs by a whole head -- 20M rows,
        // 6.4 GB -- not by alignment. Allowing at most 1 MiB of surplus
        // separates the two by three orders of magnitude.
        let size = UInt64(st.st_size)
        let surplusLimit: UInt64 = 1 << 20
        guard size >= expected, size - expected <= surplusLimit else {
            close(opened)
            throw Failure.sizeMismatch(
                path: path, expected: expected,
                actual: size)
        }
        if format == .affine4Group32 {
            do {
                try Self.validateQ4Header(
                    fd: opened, rowDim: rowDim, rowCount: rowCount, rowBytes: storedBytes)
            } catch {
                close(opened)
                throw error
            }
        }
        self.fd = opened
        self.path = path
        self.rowDim = rowDim
        self.rowBytes = rowDim * MemoryLayout<Float16>.stride
        self.storedRowBytes = storedBytes
        self.rowCount = rowCount
        self.format = format
        self.dataOffset = headerBytes
    }

    deinit { close(fd) }

    private static func validateQ4Header(
        fd: Int32, rowDim: Int, rowCount: UInt64, rowBytes: Int
    ) throws {
        var header = [UInt8](repeating: 0, count: Int(q4HeaderBytes))
        let got = header.withUnsafeMutableBytes { raw -> Int in
            guard let base = raw.baseAddress else { return -1 }
            return pread(fd, base, Int(q4HeaderBytes), 0)
        }
        guard got == Int(q4HeaderBytes) else {
            throw Failure.badHeader(detail: "short read of the 40-byte header (\(got) bytes)")
        }
        let words = header.withUnsafeBytes { raw -> [UInt32] in
            var result: [UInt32] = []
            for offset in stride(from: 0, to: 36, by: 4) {
                result.append(raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
            }
            return result
        }
        let magic = words[0]
        let version = words[1]
        let storedRowDim = Int(words[2])
        let bits = words[3]
        let group = words[4]
        let storedRowBytes = Int(words[5])
        let storedRowCount = header.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 24, as: UInt64.self)
        }
        let storedHeader = words[8]
        guard magic == q4Magic, version == q4Version else {
            throw Failure.badHeader(detail: "magic/version do not match TTQ4 v1")
        }
        guard storedRowDim == rowDim, storedRowCount == rowCount,
            storedRowBytes == rowBytes, bits == 4, group == 32,
            storedHeader == UInt32(q4HeaderBytes)
        else {
            throw Failure.badHeader(
                detail: "geometry disagrees with the sidecar "
                    + "(rowDim \(storedRowDim)/\(rowDim), rows "
                    + "\(storedRowCount)/\(rowCount), rowBytes "
                    + "\(storedRowBytes)/\(rowBytes), bits \(bits), group \(group))")
        }
    }

    /// One 4-bit row: packed nibbles, then a bf16 scale and bias per group of
    /// 32 values. Little-endian, low nibble first, the converter's layout.
    private func decodeAffine4(
        _ source: UnsafeRawPointer, into destination: UnsafeMutableRawPointer
    ) {
        let groups = rowDim / 32
        let packed = source.bindMemory(to: UInt8.self, capacity: rowDim / 2)
        let scales = source.advanced(by: rowDim / 2)
            .bindMemory(to: UInt16.self, capacity: groups)
        let biases = source.advanced(by: rowDim / 2 + groups * 2)
            .bindMemory(to: UInt16.self, capacity: groups)
        let out = destination.bindMemory(to: UInt16.self, capacity: rowDim)
        for group in 0..<groups {
            let scale = Float(bitPattern: UInt32(scales[group]) << 16)
            let bias = Float(bitPattern: UInt32(biases[group]) << 16)
            for lane in 0..<32 {
                let index = group * 32 + lane
                let byte = packed[index >> 1]
                let value = (index & 1) == 0 ? byte & 0x0F : byte >> 4
                out[index] = Float16(Float(value) * scale + bias).bitPattern
            }
        }
    }

    /// Gathers `rows` into `destination`, which must hold
    /// `rows.count * rowDim` fp16 values. Rows land in the order given, which
    /// is head order — the caller concatenates them into the PLE input.
    public func gather(
        rows: [UInt32],
        into destination: UnsafeMutableRawPointer
    ) throws {
        if format == .affine4Group32 {
            let scratch = UnsafeMutableRawPointer.allocate(
                byteCount: rowBytes, alignment: 4)
            defer { scratch.deallocate() }
            for (i, row) in rows.enumerated() {
                guard UInt64(row) < rowCount else {
                    throw Failure.rowOutOfRange(row: row, rowCount: rowCount)
                }
                try readExact(
                    row: row,
                    offset: dataOffset &+ UInt64(row) &* UInt64(storedRowBytes),
                    bytes: storedRowBytes, into: scratch)
                decodeAffine4(
                    scratch,
                    into: destination.advanced(by: i * rowBytes))
            }
            return
        }
        for (i, row) in rows.enumerated() {
            guard UInt64(row) < rowCount else {
                throw Failure.rowOutOfRange(row: row, rowCount: rowCount)
            }
            try readExact(
                row: row,
                offset: dataOffset &+ UInt64(row) &* UInt64(storedRowBytes),
                bytes: storedRowBytes,
                into: destination.advanced(by: i * rowBytes))
        }
    }

    private func readExact(
        row: UInt32, offset: UInt64, bytes: Int, into destination: UnsafeMutableRawPointer
    ) throws {
        var moved = 0
        while moved < bytes {
            let n = pread(
                fd, destination.advanced(by: moved),
                bytes - moved, off_t(offset) + off_t(moved))
            if n < 0 {
                if errno == EINTR { continue }
                throw Failure.readFailed(row: row, errno: errno)
            }
            if n == 0 {
                throw Failure.shortRead(row: row, expected: bytes, got: moved)
            }
            moved += n
        }
    }

    /// Splits one position's rows across `threads` readers.
    ///
    /// One row is a random read, so a gather's wall time is round-trips rather
    /// than bytes; the 16 rows of one decode token ran serially and measured
    /// 14-15 ms/token on the 125B while the same disk was feeding the expert
    /// stream, where the prefill path's eight-reader gather takes a fraction of
    /// that. The workers write disjoint destination rows, so nothing is shared.
    public func gatherConcurrentlyRows(
        rows: [UInt32],
        into destination: UnsafeMutableRawPointer,
        threads: Int = 8
    ) throws {
        let usable = max(1, min(threads, rows.count))
        guard usable > 1 else {
            try gather(rows: rows, into: destination)
            return
        }
        nonisolated(unsafe) let destination = destination
        let failure = GatherFailure()
        DispatchQueue.concurrentPerform(iterations: usable) { worker in
            let lower = worker * rows.count / usable
            let upper = (worker + 1) * rows.count / usable
            guard lower < upper else { return }
            do {
                try gather(
                    rows: Array(rows[lower..<upper]),
                    into: destination.advanced(by: lower * rowBytes))
            } catch {
                failure.record(error)
            }
        }
        if let error = failure.error { throw error }
    }

    /// Gathers independent batches, fanning the reads across `threads` callers.
    ///
    /// Batch `i` lands at `destination + i * chunkStride`, which is how a caller
    /// laying one row block per sequence position expects it.
    ///
    /// One row is a 320 B random read, so a gather's cost is round-trips rather
    /// than bandwidth. On a 412-token prefill of the 125B install the reads
    /// took 660 ms serially, 184 ms on four readers, 109 ms on eight and 111 ms
    /// on sixteen -- so eight is the knee, and the default. A single-threaded
    /// prefill does those reads one after another before layer 0, so the latency
    /// is exposed rather than hidden behind GPU work.
    ///
    /// `pread` does not move the shared file offset, so concurrent readers need
    /// no lock. The workers write disjoint regions of `destination`, so no byte
    /// is shared. The first failure is recorded and rethrown after every worker
    /// has stopped; the remaining rows of a failed batch are abandoned, which is
    /// the same partial state a serial gather leaves.
    public func gatherConcurrently(
        batches: [[UInt32]],
        chunkStride: Int,
        into destination: UnsafeMutableRawPointer,
        threads: Int = 8
    ) throws {
        let usable = max(1, min(threads, batches.count))
        guard usable > 1 else {
            for (index, rows) in batches.enumerated() {
                try gather(
                    rows: rows,
                    into: destination.advanced(by: index * chunkStride))
            }
            return
        }
        let chunk = (batches.count + usable - 1) / usable
        // `concurrentPerform` returns only after every iteration has finished,
        // so the buffer outlives the closure; Swift cannot see that through the
        // call and unsafe pointers are not `Sendable`, which is what this
        // binding records.
        nonisolated(unsafe) let destination = destination
        let failure = GatherFailure()
        DispatchQueue.concurrentPerform(iterations: usable) { worker in
            let lower = worker * chunk
            let upper = min(lower + chunk, batches.count)
            guard lower < upper else { return }
            for index in lower..<upper where !failure.isSet {
                do {
                    try gather(
                        rows: batches[index],
                        into: destination.advanced(by: index * chunkStride))
                } catch {
                    failure.record(error)
                }
            }
        }
        if let error = failure.error { throw error }
    }

    /// Bytes one token's gather moves, for budgeting. At the production
    /// geometry this is 16 x 320 B = 5 KiB per token — three orders of
    /// magnitude under a routed-expert token's traffic.
    public func gatherBytes(headCount: Int) -> Int { headCount * rowBytes }

    /// First failure across the gather workers, if any.
    ///
    /// unchecked-invariant: `stored` is the only mutable state, and every read
    /// and write of it happens under `lock`. `lock` is a `let`, so no caller can
    /// swap it out from under a concurrent `record`.
    private final class GatherFailure: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: Error?

        var isSet: Bool {
            lock.lock()
            defer { lock.unlock() }
            return stored != nil
        }

        var error: Error? {
            lock.lock()
            defer { lock.unlock() }
            return stored
        }

        func record(_ error: Error) {
            lock.lock()
            if stored == nil { stored = error }
            lock.unlock()
        }
    }
}
