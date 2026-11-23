import Foundation

/// Parses bounded safetensors header bytes fetched by the remote installer.
/// Tensor payload coordinates remain absolute source-file offsets so later
/// range requests can copy only the required tiles.
enum Safetensors {
    static let maxHeaderBytes: UInt64 = 1 << 24  // 16 MB — generous; observed ~95 KB

    struct Header {
        let tensors: [SourceTensor]
        /// The source dtype name of every tensor parsed as `.opaque`.
        var opaqueDTypeNames: [String: String] = [:]
    }

    /// `opaqueDTypes` names one-byte source dtypes (`U8`, `F8_E4M3`) the caller
    /// copies verbatim: they parse as `.opaque` instead of being refused, and
    /// `Header.opaqueDTypeNames` keeps what each one was.
    static func parseHeaderBytes(
        path: String,
        fileSize: UInt64,
        headerBytes data: Data,
        opaqueDTypes: Set<String> = []
    ) throws -> Header {
        let headerSize = UInt64(data.count)
        // The 8-byte size prefix must fit inside the file before we can
        // subtract it (underflows when fileSize < 8).
        if fileSize < 8 || headerSize > maxHeaderBytes || headerSize > fileSize - 8 {
            throw RepackError.safetensorsHeaderTooLarge(path: path, size: headerSize)
        }
        let rawObj = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dict = rawObj as? [String: Any] else {
            throw RepackError.safetensorsHeaderInvalid(
                path: path, detail: "header is not a JSON object")
        }

        let payloadBase = UInt64(8) + headerSize
        var tensors: [SourceTensor] = []
        var opaqueNames: [String: String] = [:]
        tensors.reserveCapacity(dict.count)
        for (name, value) in dict {
            if name == "__metadata__" { continue }
            guard let entry = value as? [String: Any] else {
                throw RepackError.safetensorsHeaderInvalid(
                    path: path,
                    detail: "entry for \(name) is not a dict")
            }
            guard let dtypeStr = entry["dtype"] as? String else {
                throw RepackError.safetensorsHeaderInvalid(
                    path: path,
                    detail: "entry for \(name) has no dtype")
            }
            let dtype: SourceTensor.Dtype
            switch dtypeStr {
            case "U32": dtype = .u32
            case "BF16": dtype = .bf16
            case "F16": dtype = .fp16
            case "F32": dtype = .fp32
            case _ where opaqueDTypes.contains(dtypeStr):
                dtype = .opaque
                opaqueNames[name] = dtypeStr
            default: throw RepackError.safetensorsUnknownDtype(path: path, dtype: dtypeStr)
            }
            guard let shape = entry["shape"] as? [Any] else {
                throw RepackError.safetensorsHeaderInvalid(
                    path: path,
                    detail: "entry for \(name) has no shape")
            }
            let shapeU64: [UInt64] = try shape.map { e in
                if let n = e as? NSNumber { return n.uint64Value }
                if let n = e as? Int {
                    guard n >= 0 else {
                        throw RepackError.safetensorsHeaderInvalid(
                            path: path,
                            detail: "entry for \(name) has a negative shape entry")
                    }
                    return UInt64(n)
                }
                throw RepackError.safetensorsHeaderInvalid(
                    path: path,
                    detail: "entry for \(name) has non-integer shape entry")
            }
            guard let offs = entry["data_offsets"] as? [Any], offs.count == 2,
                let begin = (offs[0] as? NSNumber)?.uint64Value
                    ?? (offs[0] as? Int).map({ UInt64($0) }),
                let end = (offs[1] as? NSNumber)?.uint64Value
                    ?? (offs[1] as? Int).map({ UInt64($0) })
            else {
                throw RepackError.safetensorsHeaderInvalid(
                    path: path,
                    detail: "entry for \(name) has bad data_offsets")
            }
            // Checked arithmetic: begin/end come from remote-controlled JSON,
            // so every combination is validated instead of relying on wrap
            // or trap semantics.
            guard begin <= end else {
                throw RepackError.safetensorsHeaderInvalid(
                    path: path, detail: "entry for \(name) has inverted data_offsets")
            }
            let (abs, baseOverflow) = payloadBase.addingReportingOverflow(begin)
            guard !baseOverflow, abs <= fileSize else {
                throw RepackError.safetensorsTensorOutOfRange(
                    path: path, name: name, end: abs, fileSize: fileSize)
            }
            let size = end - begin
            let (endAbs, rangeOverflow) = abs.addingReportingOverflow(size)
            if rangeOverflow || endAbs > fileSize {
                throw RepackError.safetensorsTensorOutOfRange(
                    path: path, name: name, end: endAbs, fileSize: fileSize)
            }
            let elemBytes = UInt64(dtype.elementBytes)
            let (elements, elementsOverflow) = shapeU64.reduce((UInt64(1), false)) {
                let (product, overflow) = $0.0.multipliedReportingOverflow(by: $1)
                return (product, overflow || $0.1)
            }
            let (bytes, bytesOverflow) = elements.multipliedReportingOverflow(by: elemBytes)
            if elementsOverflow || bytesOverflow || bytes != size {
                throw RepackError.shapeMismatch(
                    name: name,
                    detail: "shape product \(elements)*\(elemBytes) != size \(size)")
            }
            tensors.append(
                SourceTensor(
                    name: name, shardPath: path, dtype: dtype,
                    shape: shapeU64,
                    absoluteOffset: abs, sizeBytes: size))
        }
        return Header(tensors: tensors, opaqueDTypeNames: opaqueNames)
    }
}
