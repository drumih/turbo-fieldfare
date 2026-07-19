import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct PrefillGroupedRoutedMoETests {
    static func measuredPressureRoutes() throws -> PrefillMoEGroupedRoutes {
        var expertAssignments: [UInt32] = []
        expertAssignments.reserveCapacity(256)
        for expert in 0..<16 {
            let count = expert < 5 ? 15 : 14
            expertAssignments.append(contentsOf: repeatElement(UInt32(expert), count: count))
        }
        expertAssignments.append(contentsOf: repeatElement(UInt32(16), count: 27))
        #expect(expertAssignments.count == 256)

        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(256)
        for i in 0..<256 {
            pairs.append(Self.pair(token: UInt32(i / 8),
                                   expert: expertAssignments[i],
                                   rank: UInt32(i % 8)))
        }
        return try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 32,
            topK: 8,
            numExperts: 128,
            tileExpertCount: 16)
    }

    static func pair(token: UInt32, expert: UInt32, rank: UInt32) -> PrefillTokenExpertPair {
        PrefillTokenExpertPair(token: token,
                               expert: expert,
                               rank: rank,
                               weight: Float16(0.125 + Float(rank) * 0.0625))
    }

    static func fakeTensorViews(device: MTLDevice, count: Int) throws -> [TensorView] {
        guard let buffer = device.makeBuffer(length: max(count, 1) * 64,
                                             options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("fake tensor view buffer")
        }
        return (0..<count).map { index in
            TensorView(buffer: buffer,
                       offset: UInt64(index * 64),
                       length: 64,
                       scaleOffset: 0,
                       scaleLength: 0,
                       biasOffset: 0,
                       biasLength: 0,
                       shape: (0, UInt32(index), 0, 0),
                       dtype: 0)
        }
    }

    static func tileFetchRoutes() throws -> PrefillMoEGroupedRoutes {
        let pairs = [
            Self.pair(token: 0, expert: 3, rank: 0),
            Self.pair(token: 0, expert: 1, rank: 1),
            Self.pair(token: 1, expert: 5, rank: 0),
            Self.pair(token: 1, expert: 3, rank: 1),
            Self.pair(token: 2, expert: 1, rank: 0),
            Self.pair(token: 2, expert: 5, rank: 1),
        ]
        return try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 3,
            topK: 2,
            numExperts: 8,
            tileExpertCount: 3)
    }

    static func byte(_ view: TensorView, at relativeOffset: Int) -> UInt8 {
        view.buffer.contents()
            .advanced(by: Int(view.offset) + relativeOffset)
            .load(as: UInt8.self)
    }

    static func streamedViewsWithNonzeroOffsets(device: MTLDevice,
                                                        pool: SyntheticExpertPool,
                                                        expertIDs: [Int]) throws -> [TensorView] {
        try expertIDs.enumerated().map { index, expertID in
            let start = expertID * pool.stride
            let end = start + pool.stride
            let prefix = 64 + index * 16
            let suffix = 32
            var bytes = [UInt8](repeating: 0xA5, count: prefix)
            bytes.append(contentsOf: pool.bytes[start..<end])
            bytes.append(contentsOf: repeatElement(UInt8(0x5A), count: suffix))
            guard let buffer = device.makeBuffer(bytes: bytes,
                                                 length: bytes.count,
                                                 options: .storageModeShared) else {
                throw PrefillGroupedRoutedMoEError.allocationFailed("streamed expert \(expertID)")
            }
            return TensorView(buffer: buffer,
                              offset: UInt64(prefix),
                              length: UInt64(pool.stride),
                              scaleOffset: 0,
                              scaleLength: 0,
                              biasOffset: 0,
                              biasLength: 0,
                              shape: (0, UInt32(expertID), 0, 0),
                              dtype: 0)
        }
    }

    struct SyntheticExpertPool {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
        let stride: Int
    }

    static func makeSyntheticExpertPool(numExperts: Int, d: Int, f: Int) -> SyntheticExpertPool {
        var allBytes: [UInt8] = []
        var offsets: MoEExpertOffsets?
        var stride = 0
        for expert in 0..<numExperts {
            var bytes: [UInt8] = []
            let gateWOff = UInt32(bytes.count)
            Self.appendProjection(rows: Self.syntheticRows(rows: f, cols: d, expert: expert, role: 0),
                                  to: &bytes,
                                  component: .packed)
            let gateSOff = UInt32(bytes.count)
            Self.appendProjection(rows: Self.syntheticRows(rows: f, cols: d, expert: expert, role: 0),
                                  to: &bytes,
                                  component: .scales)
            let gateBOff = UInt32(bytes.count)
            Self.appendProjection(rows: Self.syntheticRows(rows: f, cols: d, expert: expert, role: 0),
                                  to: &bytes,
                                  component: .biases)

            let upWOff = UInt32(bytes.count)
            Self.appendProjection(rows: Self.syntheticRows(rows: f, cols: d, expert: expert, role: 1),
                                  to: &bytes,
                                  component: .packed)
            let upSOff = UInt32(bytes.count)
            Self.appendProjection(rows: Self.syntheticRows(rows: f, cols: d, expert: expert, role: 1),
                                  to: &bytes,
                                  component: .scales)
            let upBOff = UInt32(bytes.count)
            Self.appendProjection(rows: Self.syntheticRows(rows: f, cols: d, expert: expert, role: 1),
                                  to: &bytes,
                                  component: .biases)

            let downWOff = UInt32(bytes.count)
            Self.appendProjection(rows: Self.syntheticRows(rows: d, cols: f, expert: expert, role: 2),
                                  to: &bytes,
                                  component: .packed)
            let downSOff = UInt32(bytes.count)
            Self.appendProjection(rows: Self.syntheticRows(rows: d, cols: f, expert: expert, role: 2),
                                  to: &bytes,
                                  component: .scales)
            let downBOff = UInt32(bytes.count)
            Self.appendProjection(rows: Self.syntheticRows(rows: d, cols: f, expert: expert, role: 2),
                                  to: &bytes,
                                  component: .biases)

            let currentOffsets = MoEExpertOffsets(gateWOff: gateWOff,
                                                  gateSOff: gateSOff,
                                                  gateBOff: gateBOff,
                                                  upWOff: upWOff,
                                                  upSOff: upSOff,
                                                  upBOff: upBOff,
                                                  downWOff: downWOff,
                                                  downSOff: downSOff,
                                                  downBOff: downBOff)
            if offsets == nil {
                offsets = currentOffsets
                stride = bytes.count
            } else {
                #expect(stride == bytes.count)
                #expect(offsets!.gateWOff == currentOffsets.gateWOff)
                #expect(offsets!.downBOff == currentOffsets.downBOff)
            }
            allBytes.append(contentsOf: bytes)
        }
        return SyntheticExpertPool(bytes: allBytes, offsets: offsets!, stride: stride)
    }

    enum ProjectionComponent {
        case packed
        case scales
        case biases
    }

    static func appendProjection(rows: [[Float]],
                                         to bytes: inout [UInt8],
                                         component: ProjectionComponent) {
        let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
        switch component {
        case .packed:
            for row in quantized {
                bytes.append(contentsOf: row.packed)
            }
        case .scales:
            for row in quantized {
                Self.appendU16(row.scales, to: &bytes)
            }
        case .biases:
            for row in quantized {
                Self.appendU16(row.biases, to: &bytes)
            }
        }
    }

    static func syntheticRows(rows: Int, cols: Int, expert: Int, role: Int) -> [[Float]] {
        (0..<rows).map { row in
            (0..<cols).map { col in
                Float(expert + 1) * 0.001
                    + Float(role + 1) * 0.003
                    + Float((row % 7) - 3) * 0.0004
                    + Float((col % 11) - 5) * 0.0002
            }
        }
    }

    static func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
        for value in values {
            bytes.append(UInt8(truncatingIfNeeded: value))
            bytes.append(UInt8(truncatingIfNeeded: value >> 8))
        }
    }

    static func readU16(_ bytes: [UInt8], _ offset: Int) -> UInt16 {
        UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
    }

    static func cpuSyntheticRoutePartials(routes: PrefillMoEGroupedRoutes,
                                                  hidden: [Float16],
                                                  hiddenStride: Int,
                                                  pool: SyntheticExpertPool,
                                                  topK: Int,
                                                  d: Int,
                                                  f: Int) -> [Float16] {
        var out = [Float16](repeating: -99, count: routes.queryCount * topK * d)
        for pair in routes.sortedPairs {
            let expertBase = Int(pair.expert) * pool.stride
            let xBase = Int(pair.token) * hiddenStride
            let x = (0..<d).map { Float(hidden[xBase + $0]) }
            var act = [Float16](repeating: 0, count: f)
            for row in 0..<f {
                let gate = Self.cpuInt4Dot(bytes: pool.bytes,
                                           base: expertBase,
                                           wOff: Int(pool.offsets.gateWOff),
                                           sOff: Int(pool.offsets.gateSOff),
                                           bOff: Int(pool.offsets.gateBOff),
                                           row: row,
                                           n: d,
                                           x: x)
                let up = Self.cpuInt4Dot(bytes: pool.bytes,
                                         base: expertBase,
                                         wOff: Int(pool.offsets.upWOff),
                                         sOff: Int(pool.offsets.upSOff),
                                         bOff: Int(pool.offsets.upBOff),
                                         row: row,
                                         n: d,
                                         x: x)
                act[row] = Float16(MoeRef.geluTanh([gate])[0] * up)
            }
            let actFloat = act.map { Float($0) }
            let outBase = (Int(pair.token) * topK + Int(pair.rank)) * d
            for row in 0..<d {
                let value = Self.cpuInt4Dot(bytes: pool.bytes,
                                            base: expertBase,
                                            wOff: Int(pool.offsets.downWOff),
                                            sOff: Int(pool.offsets.downSOff),
                                            bOff: Int(pool.offsets.downBOff),
                                            row: row,
                                            n: f,
                                            x: actFloat)
                out[outBase + row] = Float16(value)
            }
        }
        return out
    }

    static func cpuInt4Dot(bytes: [UInt8],
                                   base: Int,
                                   wOff: Int,
                                   sOff: Int,
                                   bOff: Int,
                                   row: Int,
                                   n: Int,
                                   x: [Float]) -> Float {
        let groups = n / Quantization.groupSize
        let rowBytes = n / 2
        let wRow = base + wOff + row * rowBytes
        let sRow = base + sOff + row * groups * MemoryLayout<UInt16>.stride
        let bRow = base + bOff + row * groups * MemoryLayout<UInt16>.stride
        var acc: Float = 0
        for group in 0..<groups {
            let scale = Quantization.bf16ToFloat(Self.readU16(bytes, sRow + group * 2))
            let bias = Quantization.bf16ToFloat(Self.readU16(bytes, bRow + group * 2))
            for k in 0..<Quantization.groupSize {
                let col = group * Quantization.groupSize + k
                let packed = bytes[wRow + col / 2]
                let q = (k & 1) == 0 ? Float(packed & 0x0F) : Float(packed >> 4)
                acc += (q * scale + bias) * x[col]
            }
        }
        return acc
    }
}
