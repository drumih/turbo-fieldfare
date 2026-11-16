import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

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
            pairs.append(
                Self.pair(
                    token: UInt32(i / 8),
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
        PrefillTokenExpertPair(
            token: token,
            expert: expert,
            rank: rank,
            weight: Float16(0.125 + Float(rank) * 0.0625))
    }

    static func fakeTensorViews(device: MTLDevice, count: Int) throws -> [TensorView] {
        guard
            let buffer = device.makeBuffer(
                length: max(count, 1) * 64,
                options: .storageModeShared)
        else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("fake tensor view buffer")
        }
        return (0..<count).map { index in
            TensorView(
                buffer: buffer,
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

    static func streamedViewsWithNonzeroOffsets(
        device: MTLDevice,
        pool: SyntheticExpertPool,
        expertIDs: [Int]
    ) throws -> [TensorView] {
        try expertIDs.enumerated().map { index, expertID in
            let start = expertID * pool.stride
            let end = start + pool.stride
            let prefix = 64 + index * 16
            let suffix = 32
            var bytes = [UInt8](repeating: 0xA5, count: prefix)
            bytes.append(contentsOf: pool.bytes[start..<end])
            bytes.append(contentsOf: repeatElement(UInt8(0x5A), count: suffix))
            guard
                let buffer = device.makeBuffer(
                    bytes: bytes,
                    length: bytes.count,
                    options: .storageModeShared)
            else {
                throw PrefillGroupedRoutedMoEError.allocationFailed("streamed expert \(expertID)")
            }
            return TensorView(
                buffer: buffer,
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
        let weightBits: Int
    }

    static func makeSyntheticExpertPool(
        numExperts: Int,
        d: Int,
        f: Int,
        weightBits: Int = 4
    ) -> SyntheticExpertPool {
        precondition([2, 4, 8].contains(weightBits))
        precondition(numExperts > 0, "a pool needs at least one expert to have offsets")
        var allBytes: [UInt8] = []
        var offsets: MoEExpertOffsets?
        var stride = 0
        for expert in 0..<numExperts {
            var bytes: [UInt8] = []
            let gateWOff = UInt32(bytes.count)
            Self.appendProjection(
                rows: Self.syntheticRows(rows: f, cols: d, expert: expert, role: 0),
                to: &bytes,
                component: .packed,
                weightBits: weightBits)
            let gateSOff = UInt32(bytes.count)
            Self.appendProjection(
                rows: Self.syntheticRows(rows: f, cols: d, expert: expert, role: 0),
                to: &bytes,
                component: .scales,
                weightBits: weightBits)
            let gateBOff = UInt32(bytes.count)
            Self.appendProjection(
                rows: Self.syntheticRows(rows: f, cols: d, expert: expert, role: 0),
                to: &bytes,
                component: .biases,
                weightBits: weightBits)

            let upWOff = UInt32(bytes.count)
            Self.appendProjection(
                rows: Self.syntheticRows(rows: f, cols: d, expert: expert, role: 1),
                to: &bytes,
                component: .packed,
                weightBits: weightBits)
            let upSOff = UInt32(bytes.count)
            Self.appendProjection(
                rows: Self.syntheticRows(rows: f, cols: d, expert: expert, role: 1),
                to: &bytes,
                component: .scales,
                weightBits: weightBits)
            let upBOff = UInt32(bytes.count)
            Self.appendProjection(
                rows: Self.syntheticRows(rows: f, cols: d, expert: expert, role: 1),
                to: &bytes,
                component: .biases,
                weightBits: weightBits)

            let downWOff = UInt32(bytes.count)
            Self.appendProjection(
                rows: Self.syntheticRows(rows: d, cols: f, expert: expert, role: 2),
                to: &bytes,
                component: .packed,
                weightBits: weightBits)
            let downSOff = UInt32(bytes.count)
            Self.appendProjection(
                rows: Self.syntheticRows(rows: d, cols: f, expert: expert, role: 2),
                to: &bytes,
                component: .scales,
                weightBits: weightBits)
            let downBOff = UInt32(bytes.count)
            Self.appendProjection(
                rows: Self.syntheticRows(rows: d, cols: f, expert: expert, role: 2),
                to: &bytes,
                component: .biases,
                weightBits: weightBits)

            let currentOffsets = MoEExpertOffsets(
                gateWOff: gateWOff,
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
            } else if let existing = offsets {
                #expect(stride == bytes.count)
                #expect(existing.gateWOff == currentOffsets.gateWOff)
                #expect(existing.downBOff == currentOffsets.downBOff)
            }
            allBytes.append(contentsOf: bytes)
        }
        guard let poolOffsets = offsets else {
            preconditionFailure("numExperts > 0 was checked above")
        }
        return SyntheticExpertPool(
            bytes: allBytes,
            offsets: poolOffsets,
            stride: stride,
            weightBits: weightBits)
    }

    enum ProjectionComponent {
        case packed
        case scales
        case biases
    }

    static func appendProjection(
        rows: [[Float]],
        to bytes: inout [UInt8],
        component: ProjectionComponent,
        weightBits: Int = 4
    ) {
        let quantized = rows.map { Self.quantizeAffine($0, bits: weightBits) }
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

    struct SyntheticAffineRow {
        let packed: [UInt8]
        let scales: [UInt16]
        let biases: [UInt16]
    }

    static func quantizeAffine(_ row: [Float], bits: Int) -> SyntheticAffineRow {
        precondition(row.count.isMultiple(of: Quantization.groupSize))
        let groups = row.count / Quantization.groupSize
        let levels = Float((1 << bits) - 1)
        var packed = [UInt8](repeating: 0, count: row.count * bits / 8)
        var scales = [UInt16](repeating: 0, count: groups)
        var biases = [UInt16](repeating: 0, count: groups)
        for group in 0..<groups {
            let start = group * Quantization.groupSize
            let values = row[start..<(start + Quantization.groupSize)]
            // A group slice is never empty (`row.count` is a multiple of
            // `groupSize` and `groupSize` is positive), so the reductions below
            // always see at least one value.
            let minimum = values.reduce(Float.greatestFiniteMagnitude) { Swift.min($0, $1) }
            let maximum = values.reduce(-Float.greatestFiniteMagnitude) { Swift.max($0, $1) }
            let scaleBits = Quantization.bf16Bits(
                maximum == minimum ? 1 : (maximum - minimum) / levels)
            let biasBits = Quantization.bf16Bits(minimum)
            scales[group] = scaleBits
            biases[group] = biasBits
            let scale = Quantization.bf16ToFloat(scaleBits)
            let bias = Quantization.bf16ToFloat(biasBits)
            for local in 0..<Quantization.groupSize {
                let column = start + local
                let q = max(
                    0,
                    min(
                        Int(levels),
                        Int(((row[column] - bias) / scale).rounded())))
                let bitOffset = column * bits
                let byteOffset = bitOffset / 8
                let shift = bitOffset % 8
                var word = UInt32(q) << shift
                var remaining = bits + shift
                var index = byteOffset
                while remaining > 0 {
                    packed[index] |= UInt8(truncatingIfNeeded: word)
                    word >>= 8
                    remaining -= 8
                    index += 1
                }
            }
        }
        return SyntheticAffineRow(packed: packed, scales: scales, biases: biases)
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

    static func cpuSyntheticRoutePartials(
        routes: PrefillMoEGroupedRoutes,
        hidden: [Float16],
        hiddenStride: Int,
        pool: SyntheticExpertPool,
        topK: Int,
        d: Int,
        f: Int
    ) -> [Float16] {
        var out = [Float16](repeating: -99, count: routes.queryCount * topK * d)
        for pair in routes.sortedPairs {
            let expertBase = Int(pair.expert) * pool.stride
            let xBase = Int(pair.token) * hiddenStride
            let x = (0..<d).map { Float(hidden[xBase + $0]) }
            var act = [Float16](repeating: 0, count: f)
            for row in 0..<f {
                let gate = Self.cpuAffineDot(
                    bytes: pool.bytes,
                    base: expertBase,
                    wOff: Int(pool.offsets.gateWOff),
                    sOff: Int(pool.offsets.gateSOff),
                    bOff: Int(pool.offsets.gateBOff),
                    row: row,
                    n: d,
                    x: x,
                    bits: pool.weightBits)
                let up = Self.cpuAffineDot(
                    bytes: pool.bytes,
                    base: expertBase,
                    wOff: Int(pool.offsets.upWOff),
                    sOff: Int(pool.offsets.upSOff),
                    bOff: Int(pool.offsets.upBOff),
                    row: row,
                    n: d,
                    x: x,
                    bits: pool.weightBits)
                act[row] = Float16(MoeRef.geluTanh([gate])[0] * up)
            }
            let actFloat = act.map { Float($0) }
            let outBase = (Int(pair.token) * topK + Int(pair.rank)) * d
            for row in 0..<d {
                let value = Self.cpuAffineDot(
                    bytes: pool.bytes,
                    base: expertBase,
                    wOff: Int(pool.offsets.downWOff),
                    sOff: Int(pool.offsets.downSOff),
                    bOff: Int(pool.offsets.downBOff),
                    row: row,
                    n: f,
                    x: actFloat,
                    bits: pool.weightBits)
                out[outBase + row] = Float16(value)
            }
        }
        return out
    }

    static func cpuAffineDot(
        bytes: [UInt8],
        base: Int,
        wOff: Int,
        sOff: Int,
        bOff: Int,
        row: Int,
        n: Int,
        x: [Float],
        bits: Int
    ) -> Float {
        let groups = n / Quantization.groupSize
        let rowBytes = n * bits / 8
        let mask = UInt32((1 << bits) - 1)
        let wRow = base + wOff + row * rowBytes
        let sRow = base + sOff + row * groups * MemoryLayout<UInt16>.stride
        let bRow = base + bOff + row * groups * MemoryLayout<UInt16>.stride
        var acc: Float = 0
        for group in 0..<groups {
            let scale = Quantization.bf16ToFloat(Self.readU16(bytes, sRow + group * 2))
            let bias = Quantization.bf16ToFloat(Self.readU16(bytes, bRow + group * 2))
            for k in 0..<Quantization.groupSize {
                let col = group * Quantization.groupSize + k
                let bitOffset = col * bits
                let byteOffset = wRow + bitOffset / 8
                let shift = bitOffset % 8
                var word = UInt32(bytes[byteOffset])
                if shift + bits > 8 {
                    word |= UInt32(bytes[byteOffset + 1]) << 8
                }
                let q = Float((word >> shift) & mask)
                acc += (q * scale + bias) * x[col]
            }
        }
        return acc
    }
}
