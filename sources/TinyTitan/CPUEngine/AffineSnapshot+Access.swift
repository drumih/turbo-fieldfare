import Foundation
import TinyTitanCore

// Reading tensors out of a snapshot: the shard lookup and the `floats` /
// `has` / `matrix` accessors.
//
// Split out of `AffineSnapshot.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion.
extension AffineSnapshot {

    private func shard(_ name: String) throws -> SafeTensorsFile {
        guard let file = placement[name], let shard = shards[file] else {
            throw SafeTensorsFile.Failure.missing(name)
        }
        return shard
    }

    public func floats(_ name: String) throws -> [Float] {
        if case .ssdai(let index, let weights) = storage {
            guard let entry = index.entries[name] else {
                throw SafeTensorsFile.Failure.missing(name)
            }
            guard let base = weights.base else {
                throw SafeTensorsFile.Failure.malformed("model_weights.bin could not be mapped")
            }
            let raw = UnsafeRawBufferPointer(
                start: base.advanced(by: Int(entry.fileOffset)),
                count: Int(entry.sizeBytes))
            // Dtype codes are the repacker's (`ietnyDtype`): 0 u32, 1 bf16,
            // 2 fp16, 3 fp32. The bf16 widening is the same bit trick the
            // safetensors reader uses -- the top sixteen bits of a float32 are
            // exactly a bfloat16.
            switch entry.dtype {
            case 3:
                return Array(raw.bindMemory(to: Float.self))
            case 1:
                return raw.bindMemory(to: UInt16.self).map {
                    Float(bitPattern: UInt32($0) << 16)
                }
            case 2:
                return raw.bindMemory(to: Float16.self).map(Float.init)
            default:
                throw SafeTensorsFile.Failure.unsupported(
                    dtype: "resident dtype \(entry.dtype)", name: name)
            }
        }
        return try shard(name).floats(name)
    }

    public func has(_ name: String) -> Bool {
        switch storage {
        case .safetensors: return placement[name] != nil
        case .ssdai(let index, _): return index.entries[name] != nil
        }
    }

    /// A quantized matrix by its `.weight` name.
    public func matrix(_ name: String) throws -> Matrix {
        if case .ssdai(let index, let weights) = storage {
            return try Self.matrix(
                name, index: index, weights: weights,
                groupSize: groupSize, bits: bits(forStem: stem(of: name)))
        }
        let stem =
            name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count)) : name
        let shard = try shard(name)
        let entry = try shard.entry(name)
        guard entry.shape.count == 2 else {
            throw SafeTensorsFile.Failure.malformed("\(name) is not a matrix")
        }
        let width = bits(forStem: stem)
        let lanes = 32 / width
        let rows = entry.shape[0]
        let columns = entry.shape[1] * lanes
        // Same division, same reason as the resident-index branch above.
        guard columns % groupSize == 0 else {
            throw SafeTensorsFile.Failure.malformed(
                "\(stem): width \(columns) at \(width) bits is not a whole number "
                    + "of \(groupSize)-element groups")
        }
        let scales = try shard.bytes(stem + ".scales")
        let biases = try shard.bytes(stem + ".biases")
        guard scales.count == rows * (columns / groupSize) * 2,
            biases.count == scales.count
        else {
            throw SafeTensorsFile.Failure.malformed(
                "\(stem): scales and biases do not match \(rows)x\(columns) "
                    + "at \(width) bits, group \(groupSize)")
        }
        guard let scalesBase = scales.baseAddress?.assumingMemoryBound(to: UInt16.self),
            let biasesBase = biases.baseAddress?.assumingMemoryBound(to: UInt16.self)
        else {
            throw SafeTensorsFile.Failure.malformed(
                "\(stem): scales or biases have no storage")
        }
        return Matrix(
            weights: try shard.bytes(name),
            scales: scalesBase,
            biases: biasesBase,
            rows: rows,
            columns: columns,
            bits: width,
            groupSize: groupSize)
    }
}
