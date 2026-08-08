import Foundation

/// One physical tensor in a Kimi K3 safetensors shard. K3-own twin of the
/// Gemma `SourceTensor`: the K3 checkpoint stores routed-expert payloads as
/// U8 (two E2M1 nibbles per byte / one E8M0 scale byte per group of 32), a
/// dtype the Gemma-oriented `Safetensors` parser refuses, so the K3 pipeline
/// parses shard headers here. Coordinates are absolute source-file offsets.
struct K3SourceTensor: Sendable, Hashable {
    enum Dtype: UInt8, Sendable, Hashable {
        case u8   = 0
        case bf16 = 1
        case fp32 = 2

        var elementBytes: Int {
            switch self { case .u8: 1; case .bf16: 2; case .fp32: 4 }
        }
    }

    let name: String
    let shardPath: String
    let dtype: Dtype
    let shape: [UInt64]
    let absoluteOffset: UInt64
    let sizeBytes: UInt64

    var elementCount: UInt64 { shape.reduce(UInt64(1), *) }
}

/// Parses bounded K3 safetensors header bytes. Mirrors `Safetensors` with the
/// K3 dtype set (U8 / BF16 / F32); anything else throws.
enum K3Safetensors {
    static let maxHeaderBytes = Safetensors.maxHeaderBytes

    struct Header {
        let tensors: [K3SourceTensor]
    }

    static func parseHeaderBytes(path: String,
                                 fileSize: UInt64,
                                 headerBytes data: Data) throws -> Header {
        let headerSize = UInt64(data.count)
        if headerSize > maxHeaderBytes || headerSize > fileSize - 8 {
            throw RepackError.safetensorsHeaderTooLarge(path: path, size: headerSize)
        }
        let rawObj = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dict = rawObj as? [String: Any] else {
            throw RepackError.safetensorsHeaderInvalid(path: path, detail: "header is not a JSON object")
        }

        let payloadBase = UInt64(8) + headerSize
        var tensors: [K3SourceTensor] = []
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
            let dtype: K3SourceTensor.Dtype
            switch dtypeStr {
            case "U8":   dtype = .u8
            case "BF16": dtype = .bf16
            case "F32":  dtype = .fp32
            default: throw RepackError.safetensorsUnknownDtype(path: path, dtype: dtypeStr)
            }
            guard let shape = entry["shape"] as? [Any] else {
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) has no shape")
            }
            let shapeU64: [UInt64] = try shape.map { e in
                if let n = e as? NSNumber { return n.uint64Value }
                if let n = e as? Int { return UInt64(n) }
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) has non-integer shape entry")
            }
            guard let offs = entry["data_offsets"] as? [Any], offs.count == 2,
                  let begin = (offs[0] as? NSNumber)?.uint64Value ?? (offs[0] as? Int).map({ UInt64($0) }),
                  let end   = (offs[1] as? NSNumber)?.uint64Value ?? (offs[1] as? Int).map({ UInt64($0) })
            else {
                throw RepackError.safetensorsHeaderInvalid(path: path,
                                                           detail: "entry for \(name) has bad data_offsets")
            }
            let abs = payloadBase + begin
            let size = end - begin
            let endAbs = abs + size
            if endAbs > fileSize {
                throw RepackError.safetensorsTensorOutOfRange(path: path, name: name,
                                                              end: endAbs, fileSize: fileSize)
            }
            let elemBytes = UInt64(dtype.elementBytes)
            let elements = shapeU64.reduce(UInt64(1), *)
            if elements * elemBytes != size {
                throw RepackError.shapeMismatch(name: name,
                                                detail: "shape product \(elements)*\(elemBytes) != size \(size)")
            }
            tensors.append(K3SourceTensor(name: name, shardPath: path, dtype: dtype,
                                          shape: shapeU64,
                                          absoluteOffset: abs, sizeBytes: size))
        }
        return Header(tensors: tensors)
    }
}
