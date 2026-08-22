import Foundation

/// Parses bounded safetensors header bytes fetched by the remote installer.
/// Tensor payload coordinates remain absolute source-file offsets so later
/// range requests can copy only the required tiles.
enum Safetensors {
    static let maxHeaderBytes: UInt64 = 1 << 24  // 16 MB — generous; observed ~95 KB

    struct Header {
        let path: String
        /// Where tensor payloads begin, so a caller can bound a read against
        /// the header rather than trusting an offset the file supplied.
        let payloadBaseOffset: UInt64
        let tensors: [SourceTensor]
    }

    static func parseHeader(path: String) throws -> Header {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        var st = stat()
        if fstat(fd, &st) != 0 {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        let fileSize = UInt64(st.st_size)
        if fileSize < 8 {
            throw RepackError.safetensorsHeaderInvalid(path: path, detail: "file too short")
        }

        var headerSizeLE: UInt64 = 0
        try withUnsafeMutableBytes(of: &headerSizeLE) { raw in
            try Posix.preadAll(fd: fd, path: path, buf: raw.baseAddress!, count: 8, offset: 0)
        }
        let headerSize = UInt64(littleEndian: headerSizeLE)
        if headerSize > maxHeaderBytes || headerSize > fileSize - 8 {
            throw RepackError.safetensorsHeaderTooLarge(path: path, size: headerSize)
        }

        let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: Int(headerSize), alignment: 16)
        defer { buf.deallocate() }
        try Posix.preadAll(fd: fd, path: path, buf: buf.baseAddress!, count: Int(headerSize), offset: 8)
        let data = Data(bytesNoCopy: buf.baseAddress!, count: Int(headerSize), deallocator: .none)

        return try parseHeaderBytes(path: path,
                                    fileSize: fileSize,
                                    headerBytes: data)
    }

    static func parseHeaderBytes(path: String,
                                        fileSize: UInt64,
                                        headerBytes data: Data) throws -> Header {
        let headerSize = UInt64(data.count)
        // `fileSize - 8` underflows and traps on a file shorter than its own
        // length prefix, which is exactly what a hostile shard supplies.
        guard fileSize >= 8 else {
            throw RepackError.safetensorsHeaderInvalid(
                path: path, detail: "file too short")
        }
        if headerSize > maxHeaderBytes || headerSize > fileSize - 8 {
            throw RepackError.safetensorsHeaderTooLarge(path: path, size: headerSize)
        }
        let rawObj = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dict = rawObj as? [String: Any] else {
            throw RepackError.safetensorsHeaderInvalid(path: path, detail: "header is not a JSON object")
        }

        let payloadBase = UInt64(8) + headerSize
        var tensors: [SourceTensor] = []
        tensors.reserveCapacity(dict.count)
        for (name, value) in dict {
            if name == "__metadata__" { continue }
            guard let entry = value as? [String: Any] else {
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) is not a dict")
            }
            guard let dtypeStr = entry["dtype"] as? String else {
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) has no dtype")
            }
            let dtype: SourceTensor.Dtype
            switch dtypeStr {
            case "U32":  dtype = .u32
            case "BF16": dtype = .bf16
            case "F16":  dtype = .fp16
            case "F32":  dtype = .fp32
            default: throw RepackError.safetensorsUnknownDtype(path: path, dtype: dtypeStr)
            }
            guard let shape = entry["shape"] as? [Any] else {
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) has no shape")
            }
            let shapeU64: [UInt64] = try shape.map { element in
                guard let extent = Self.nonNegativeUInt64(element) else {
                    throw RepackError.safetensorsHeaderInvalid(
                        path: path,
                        detail: "entry for \(name) has non-integer shape entry")
                }
                return extent
            }
            guard let offs = entry["data_offsets"] as? [Any], offs.count == 2,
                  let begin = Self.nonNegativeUInt64(offs[0]),
                  let end = Self.nonNegativeUInt64(offs[1])
            else {
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) has bad data_offsets")
            }
            // Every term below comes from a JSON header that, for a remote
            // model, is not covered by any pinned digest - only the index file
            // is. Unchecked `+`/`-`/`*` on those values traps, so a hostile or
            // corrupt shard would abort the process instead of failing the
            // download.
            guard end >= begin else {
                throw RepackError.safetensorsHeaderInvalid(
                    path: path,
                    detail: "entry for \(name) has data_offsets end \(end) "
                        + "before begin \(begin)")
            }
            let size = end - begin
            let abs = try Self.checked(path: path, name: name, "absolute offset") {
                payloadBase.addingReportingOverflow(begin)
            }
            let endAbs = try Self.checked(path: path, name: name, "absolute end") {
                abs.addingReportingOverflow(size)
            }
            if endAbs > fileSize {
                throw RepackError.safetensorsTensorOutOfRange(path: path, name: name,
                                                              end: endAbs, fileSize: fileSize)
            }
            let elemBytes = UInt64(dtype.elementBytes)
            var elements = UInt64(1)
            for extent in shapeU64 {
                elements = try Self.checked(path: path, name: name, "shape product") {
                    elements.multipliedReportingOverflow(by: extent)
                }
            }
            let declaredBytes = try Self.checked(path: path, name: name, "element bytes") {
                elements.multipliedReportingOverflow(by: elemBytes)
            }
            if declaredBytes != size {
                throw RepackError.shapeMismatch(name: name,
                                                detail: "shape product \(elements)*\(elemBytes) != size \(size)")
            }
            tensors.append(SourceTensor(name: name, shardPath: path, dtype: dtype,
                                        shape: shapeU64,
                                        absoluteOffset: abs, sizeBytes: size))
        }
        return Header(path: path, payloadBaseOffset: payloadBase, tensors: tensors)
    }

    /// A whole number in `0...UInt64.max`, or nil.
    ///
    /// Sign alone is not enough. JSON has one number type, so `1.5` and `1e30`
    /// parse as `NSNumber` just as `1` does, and `uint64Value` answers both —
    /// by truncating the first and saturating the second. Either would let a
    /// header we did not write name a shape or an offset that is not the one it
    /// declares.
    private static func nonNegativeUInt64(_ value: Any) -> UInt64? {
        if let number = value as? Int { return number >= 0 ? UInt64(number) : nil }
        guard let number = value as? NSNumber else { return nil }
        guard number.compare(NSNumber(value: 0)) != .orderedAscending else { return nil }
        let double = number.doubleValue
        guard double == double.rounded(),
              double <= Double(UInt64.max),
              number.compare(NSNumber(value: UInt64.max)) != .orderedDescending else {
            return nil
        }
        return number.uint64Value
    }

    /// Turns an overflowing header computation into a diagnosable error rather
    /// than a trap.
    private static func checked(
        path: String,
        name: String,
        _ field: String,
        _ operation: () -> (partialValue: UInt64, overflow: Bool)
    ) throws -> UInt64 {
        let (value, overflow) = operation()
        guard !overflow else {
            throw RepackError.safetensorsHeaderInvalid(
                path: path, detail: "entry for \(name) overflows \(field)")
        }
        return value
    }
}
