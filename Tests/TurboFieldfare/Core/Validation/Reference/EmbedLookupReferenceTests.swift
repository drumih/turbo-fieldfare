import Testing
import Foundation
import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct EmbedLookupReferenceTests {
    @Test("Lookup matches direct dequant of the same row", arguments: [
        (Int(16), 128, UInt64(0x10B)),
        (Int(32), 256, UInt64(0x10C)),
        (Int(64), 512, UInt64(0x10D)),
    ])
    func lookupMatchesDirectDequant(v: Int, d: Int, seed: UInt64) {
        var rng = SeedTree(seed).key("embed-lookup-v\(v)-d\(d)")
        var rows: [[Float]] = []
        rows.reserveCapacity(v)
        for _ in 0..<v {
            rows.append((0..<d).map { _ in rng.uniform(-1.0, 1.0) })
        }

        let groupsPerRow = d / Quantization.groupSize
        var packed = [UInt8]( repeating: 0, count: v * d)
        var scales = [UInt16](repeating: 0, count: v * groupsPerRow)
        var biases = [UInt16](repeating: 0, count: v * groupsPerRow)
        for token in 0..<v {
            let q = Quantization.quantizeInt8Affine(rows[token])
            for i in 0..<d { packed[token * d + i] = q.packed[i] }
            for g in 0..<groupsPerRow {
                scales[token * groupsPerRow + g] = q.scales[g]
                biases[token * groupsPerRow + g] = q.biases[g]
            }
        }

        for token in [0, v / 2, v - 1] {
            let viaRef = EmbedLookupRef.apply(
                tablePacked: packed, tableScales: scales, tableBiases: biases,
                tokenId: token, d: d
            )
            let q = Quantization.quantizeInt8Affine(rows[token])
            let viaDirect = Quantization.dequantizeInt8Affine(q, n: d)
            let rel = RelError.compute(actual: viaRef, reference: viaDirect)
            #expect(rel < Tolerance.identity, "v=\(v) d=\(d) token=\(token) rel=\(rel)")
        }
    }
}
