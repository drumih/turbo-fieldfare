import Foundation

/// Recursive Walsh-Hadamard transform (`/sqrt(2)` per stage). Implemented
/// divide-and-conquer style — splits the input into halves, recurses, then
/// combines. The kernel uses an iterative stride-doubling butterfly inside
/// SIMD threadgroups. Different control flow, different summation order.
public enum WhtRef {
    public static func apply(_ x: [Float]) -> [Float] {
        let d = x.count
        precondition(d > 0 && (d & (d - 1)) == 0, "D must be a power of two")
        if d == 1 { return x }
        return walsh(x)
    }

    private static func walsh(_ x: [Float]) -> [Float] {
        let n = x.count
        if n == 1 { return x }
        let half = n / 2
        // Split into low / high halves, recurse, then combine.
        let lo = walsh(Array(x[0..<half]))
        let hi = walsh(Array(x[half..<n]))
        let invSqrt2: Float = 1.0 / Float(2.0).squareRoot()
        var y = [Float](repeating: 0, count: n)
        for i in 0..<half {
            y[i] = (lo[i] + hi[i]) * invSqrt2
            y[half + i] = (lo[i] - hi[i]) * invSqrt2
        }
        return y
    }
}

public enum TurboQuantRef {
    private static let codebook: [Float] = [
        -2.7326, -2.0690, -1.6181, -1.2562,
        -0.9424, -0.6568, -0.3881, -0.1284,
         0.1284,  0.3881,  0.6568,  0.9424,
         1.2562,  1.6181,  2.0690,  2.7326,
    ]

    public static func materializePackedCache(
        _ cache: [UInt8],
        tokenCount: Int,
        headDim: Int,
        numHeads: Int,
        bytesPerHead: Int,
        bytesPerToken: Int,
        packedOffset: Int,
        scaleOffset: Int,
        layer: UInt32 = 0,
        rotationSeed: UInt32 = 0,
        applyRotation: Bool = false
    ) -> [Float] {
        precondition(tokenCount >= 0 && headDim > 0 && numHeads > 0)
        precondition(cache.count >= tokenCount * bytesPerToken)
        var output = [Float](repeating: 0, count: tokenCount * numHeads * headDim)

        for token in 0..<tokenCount {
            for head in 0..<numHeads {
                let cacheBase = token * bytesPerToken + head * bytesPerHead
                let scaleBits = UInt16(cache[cacheBase + scaleOffset])
                    | (UInt16(cache[cacheBase + scaleOffset + 1]) << 8)
                let scale = Float(Float16(bitPattern: scaleBits))
                var transformed = [Float](repeating: 0, count: headDim)
                for dimension in 0..<headDim {
                    let byte = cache[cacheBase + packedOffset + dimension / 2]
                    let index = dimension.isMultiple(of: 2)
                        ? Int(byte & 0x0F)
                        : Int(byte >> 4)
                    transformed[dimension] = Float(Float16(codebook[index] * scale))
                }

                let restored = WhtRef.apply(transformed)
                let outputBase = (token * numHeads + head) * headDim
                for dimension in 0..<headDim {
                    let value = applyRotation
                        && rotationSign(layer: layer,
                                        head: UInt32(head),
                                        dimension: UInt32(dimension),
                                        seed: rotationSeed)
                        ? -restored[dimension]
                        : restored[dimension]
                    output[outputBase + dimension] = Float(Float16(value))
                }
            }
        }
        return output
    }

    private static func rotationSign(layer: UInt32,
                                     head: UInt32,
                                     dimension: UInt32,
                                     seed: UInt32) -> Bool {
        var value = UInt64(seed)
        value ^= UInt64(layer) &* 0x9E37_79B9_7F4A_7C15
        value ^= UInt64(head) &* 0xBF58_476D_1CE4_E5B9
        value ^= UInt64(dimension) &* 0x94D0_49BB_1331_11EB
        return mix64(value) >> 63 != 0
    }

    private static func mix64(_ input: UInt64) -> UInt64 {
        var value = input
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}
