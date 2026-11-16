import Foundation

public enum Quantization {

    public static let groupSize: Int = 64

    // MARK: - BF16 helpers
    //
    // BF16 = top 16 bits of FP32 (8-bit exponent, 7-bit mantissa). Stored on
    // disk and on the GPU as the native `bfloat` type; in Swift we carry the
    // raw bits as `UInt16` and convert via this pair. Round-half-to-even on
    // encode matches the IEEE-754 default. No NaN/Inf special-case: fixtures
    // are bounded and finite, and the .ssdai importer copies BF16 bytes
    // through unchanged so we never call the encoder on weight data.

    @inline(always)
    public static func bf16Bits(_ x: Float) -> UInt16 {
        let bits = x.bitPattern
        let lsb = (bits >> 16) & 1
        let roundingBias: UInt32 = 0x7FFF &+ lsb
        let rounded = (bits &+ roundingBias) >> 16
        return UInt16(truncatingIfNeeded: rounded)
    }

    @inline(always)
    public static func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    // MARK: - INT4 affine

    /// MLX `affine` 4-bit row. Packed unsigned nibbles, BF16 scale + bias per
    /// group of 64. `scales` and `biases` carry BF16 bit patterns as `UInt16`
    /// so the same buffer can be uploaded to a Metal `device const bfloat*`.
    public struct Int4AffineRow {
        public let packed: [UInt8]  // N / 2 bytes; low nibble = even index, high = odd
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    /// Affine 4-bit quantize: `q ∈ [0..15]`, `w ≈ q * scale + bias`.
    /// Scale and bias are computed from per-group min/max, then rounded to BF16.
    /// Test-fixture only — the runtime importer never calls this.
    /// `group` defaults to the engine's 64. The PLE table's 160-wide rows need
    /// 32 (160 is not a multiple of 64), which is the checkpoint's own group
    /// for that axis.
    public static func quantizeInt4Affine(_ row: [Float]) -> Int4AffineRow {
        quantizeInt4Affine(row, group: groupSize)
    }

    /// `group` defaults to the engine's 64. The PLE table's 160-wide rows need
    /// 32 (160 is not a multiple of 64), which is the checkpoint's own group
    /// for that axis.
    public static func quantizeInt4Affine(_ row: [Float], group: Int) -> Int4AffineRow {
        precondition(
            group > 0 && group % 2 == 0 && row.count % group == 0,
            "row length \(row.count) is not a multiple of group \(group)")

        let nGroups = row.count / group
        var packed = [UInt8](repeating: 0, count: row.count / 2)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float = .infinity
            var wmax: Float = -.infinity
            for k in 0..<group {
                let w = row[g * group + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            // Constant group: scale=1, bias=value preserves exact reconstruction.
            let scaleF: Float
            let biasF: Float
            if wmax == wmin {
                scaleF = 1
                biasF = wmin
            } else {
                scaleF = (wmax - wmin) / 15.0
                biasF = wmin
            }
            // Round through BF16 first, then quantize against the rounded
            // values so the runtime decode (which reads BF16) reproduces the
            // same q the fixture stored.
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<group {
                let w = row[g * group + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(15, q))
                let nibble = UInt8(q) & 0x0F
                let byteIdx = g * (group / 2) + (k / 2)
                if (k & 1) == 0 {
                    packed[byteIdx] = (packed[byteIdx] & 0xF0) | nibble
                } else {
                    packed[byteIdx] = (packed[byteIdx] & 0x0F) | (nibble << 4)
                }
            }
        }
        return Int4AffineRow(packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeInt4Affine(_ r: Int4AffineRow, n: Int) -> [Float] {
        dequantizeInt4Affine(r, n: n, group: groupSize)
    }

    public static func dequantizeInt4Affine(
        _ r: Int4AffineRow, n: Int, group: Int
    ) -> [Float] {
        precondition(n == r.packed.count * 2)
        precondition(group > 0 && n % group == 0, "n \(n) is not a multiple of group \(group)")
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / group
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias = bf16ToFloat(r.biases[g])
            for k in 0..<group {
                let byteIdx = g * (group / 2) + (k / 2)
                let b = r.packed[byteIdx]
                let nibble: Int = (k & 1) == 0 ? Int(b & 0x0F) : Int(b >> 4)
                out[g * group + k] = Float(nibble) * scale + bias
            }
        }
        return out
    }

    // MARK: - INT8 affine

    public struct Int8AffineRow {
        public let packed: [UInt8]  // N unsigned bytes
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    public static func quantizeInt8Affine(_ row: [Float]) -> Int8AffineRow {
        quantizeInt8Affine(row, group: groupSize)
    }

    public static func quantizeInt8Affine(_ row: [Float], group: Int) -> Int8AffineRow {
        precondition(
            group > 0 && row.count % group == 0,
            "row length \(row.count) is not a multiple of group \(group)")

        let nGroups = row.count / group
        var packed = [UInt8](repeating: 0, count: row.count)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float = .infinity
            var wmax: Float = -.infinity
            for k in 0..<group {
                let w = row[g * group + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            let scaleF: Float
            let biasF: Float
            if wmax == wmin {
                scaleF = 1
                biasF = wmin
            } else {
                scaleF = (wmax - wmin) / 255.0
                biasF = wmin
            }
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<group {
                let w = row[g * group + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(255, q))
                packed[g * group + k] = UInt8(q)
            }
        }
        return Int8AffineRow(packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeInt8Affine(_ r: Int8AffineRow, n: Int) -> [Float] {
        dequantizeInt8Affine(r, n: n, group: groupSize)
    }

    public static func dequantizeInt8Affine(
        _ r: Int8AffineRow, n: Int, group: Int
    ) -> [Float] {
        precondition(n == r.packed.count)
        precondition(group > 0 && n % group == 0, "n \(n) is not a multiple of group \(group)")
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / group
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias = bf16ToFloat(r.biases[g])
            for k in 0..<group {
                out[g * group + k] = Float(r.packed[g * group + k]) * scale + bias
            }
        }
        return out
    }
}
