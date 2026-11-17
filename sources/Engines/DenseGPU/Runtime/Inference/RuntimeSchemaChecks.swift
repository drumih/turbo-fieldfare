import Darwin
import Foundation
import Metal
import TinyTitanFormat
import TinyTitanCore

// The schema checks `validateRuntimeSchema` runs, bound to the index and quant
// slots they read.
//
// Split out of `Model.swift` (2026-09-28) under the 500-line-per-file rule
// (Task 8 of the cleanup runbook) as pure code motion.
/// The schema checks `validateRuntimeSchema` runs, bound to the index and
/// quant slots they read. Extracted from that function so the per-family and
/// per-layer rules below read as rules rather than as one 250-line body.
struct RuntimeSchemaChecks {
    let residentIndex: ResidentIndex
    let quant: ManifestQuant

    func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw ModelError.indexCorrupt(detail: "\(field) byte count overflows UInt64")
        }
        return value
    }

    func checkedIntMultiply(_ lhs: Int, _ rhs: Int, field: String) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw ModelError.indexCorrupt(detail: "\(field) dimension overflows Int")
        }
        return value
    }

    func entry(_ name: String) throws -> ResidentIndexEntry {
        guard let e = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        return e
    }

    /// Accepts a tensor the kernels read as bf16 whether the checkpoint kept it
    /// at bf16 or at fp32.
    ///
    /// The dense Qwen 3.5 installs keep `A_log` and the gated norm in fp32, as
    /// their source checkpoints do, while `gdn.metal` takes `device const
    /// bfloat*` for both; the runtime promotes them once at load (`Model`
    /// builds the bf16 view), so both widths are executable. Any other dtype is
    /// still refused -- the kernels would read it as bf16 regardless.
    func requireBF16OrFP32(_ name: String, count: Int) throws {
        let e = try entry(name)
        guard e.dtype == 1 || e.dtype == 3 else {
            throw ModelError.indexCorrupt(detail: "\(name) is neither BF16 nor FP32")
        }
        let dims = [e.shape.0, e.shape.1, e.shape.2, e.shape.3]
        let elements = dims.reduce(1) { $0 * ($1 == 0 ? 1 : Int($1)) }
        guard elements == count else {
            throw ModelError.tensorSizeMismatch(
                name: name, expected: UInt64(count), actual: UInt64(elements))
        }
    }

    func requireBF16(_ name: String, count: Int) throws {
        let e = try entry(name)
        guard e.dtype == 1 else {
            throw ModelError.indexCorrupt(detail: "\(name) is not BF16")
        }
        // Trailing zero dims encode a lower-rank tensor (e.g. a [2048]
        // vector is stored as shape (2048, 0, 0, 0)); treat them as 1.
        let dims = [e.shape.0, e.shape.1, e.shape.2, e.shape.3]
        let elements = dims.reduce(1) { $0 * ($1 == 0 ? 1 : Int($1)) }
        guard elements == count else {
            throw ModelError.tensorSizeMismatch(
                name: name, expected: UInt64(count), actual: UInt64(elements))
        }
    }

    /// Accepts a tensor that is either affine-quantized at the slot's width or
    /// kept at the checkpoint's own bf16.
    ///
    /// Promotion is per tensor, so a family can be unquantized inside a slot
    /// that is not. Checking only the slot would refuse a correct install; not
    /// checking at all would let a wrong one through, which for these tensors
    /// is silent -- the kernels pick their reading from the same dtype.
    func requireAffineOrBF16(
        _ name: String, rows: Int, columns: Int,
        slot: ManifestQuantSlot
    ) throws {
        let e = try entry(name)
        if e.dtype == 1 {
            try requireBF16(name, count: rows * columns)
            return
        }
        try requireAffine(name, rows: rows, columns: columns, slot: slot)
    }

    func requireAffine(
        _ name: String, rows: Int, columns: Int,
        slot: ManifestQuantSlot
    ) throws {
        let e = try entry(name)
        guard columns % slot.groupSize == 0 else {
            throw ModelError.indexCorrupt(
                detail: "\(name) columns \(columns) not divisible by group size \(slot.groupSize)")
        }
        // Bit-packed affine weights: rows*cols*bits must pack into whole
        // bytes (4-bit packs 2/byte, 6-bit packs across 32-bit words).
        let elementBits = try checkedMultiply(
            UInt64(rows) * UInt64(columns), UInt64(slot.weightBits),
            field: name)
        guard elementBits % 8 == 0 else {
            throw ModelError.indexCorrupt(
                detail: "\(name) \(slot.weightBits)-bit layout does not pack into whole bytes")
        }
        let weightBytes = elementBits / 8
        let auxBytes = try checkedMultiply(
            UInt64(rows) * UInt64(columns / slot.groupSize), 2, field: name)
        guard e.dtype == 0,  // U32-packed weights
            e.sizeBytes == weightBytes,
            e.scaleOffset > 0, e.scaleSize == auxBytes,
            e.biasOffset > 0, e.biasSize == auxBytes
        else {
            throw ModelError.tensorSizeMismatch(
                name: name, expected: weightBytes, actual: e.sizeBytes)
        }
    }

    func affineSizes(
        rows: Int, columns: Int, slot: ManifestQuantSlot,
        field: String
    ) throws -> (weight: UInt64, aux: UInt64, shape: (UInt32, UInt32)) {
        guard columns % slot.groupSize == 0 else {
            throw ModelError.indexCorrupt(
                detail:
                    "\(field) has an invalid quant layout (group \(slot.groupSize), \(slot.weightBits)-bit)"
            )
        }
        let elementBits = try checkedMultiply(
            UInt64(rows) * UInt64(columns), UInt64(slot.weightBits),
            field: field)
        guard elementBits % 8 == 0 else {
            throw ModelError.indexCorrupt(
                detail: "\(field) \(slot.weightBits)-bit layout does not pack into whole bytes")
        }
        let weightBytes = elementBits / 8
        let auxBytes = try checkedMultiply(
            UInt64(rows) * UInt64(columns / slot.groupSize), 2, field: field)
        return (weightBytes, auxBytes, (UInt32(rows), UInt32(columns)))
    }
}
