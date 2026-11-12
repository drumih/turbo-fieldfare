import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// Rolls a W-row Gated-DeltaNet pass back to k accepted rows and checks the
/// state and conv tail bit for bit against a run of only those k rows from the
/// same starting state -- the guarantee the wide verify's partial accept rests
/// on. Two linear layers on slot 1 of a two-slot store, so per-layer and
/// per-slot addressing are exercised and slot 0 must come through untouched.
@Suite struct WideVerifyRollbackTests {
    static let cfg = LinearAttentionConfig(
        numKHeads: 2, numVHeads: 4, keyHeadDim: 32, valueHeadDim: 32,
        convKernelSize: 4)
    static let layers = [1, 4]
    static let prefixRows = 3
    static let passRows = 6
    static let slot = 1

    /// One independent GDN store plus the inputs every rig shares by seed.
    final class Rig {
        let ctx: MetalContext
        let gdn: GDN
        var state: [Int: MTLBuffer] = [:]
        var tail: [Int: MTLBuffer] = [:]
        var convW: [Int: MTLBuffer] = [:]
        var aLog: [Int: MTLBuffer] = [:]
        var dtBias: [Int: MTLBuffer] = [:]
        var qkv: [Int: MTLBuffer] = [:]
        var a: [Int: MTLBuffer] = [:]
        var b: [Int: MTLBuffer] = [:]
        let convOut: MTLBuffer
        let y: MTLBuffer

        let stateBytes: Int
        let tailBytes: Int
        static let half = MemoryLayout<Float16>.stride
        var qkvRowBytes: Int { cfg.qkvDim * Self.half }
        var headRowBytes: Int { cfg.numVHeads * Self.half }

        init() throws {
            let cfg = WideVerifyRollbackTests.cfg
            ctx = try MetalContext()
            gdn = try GDN(context: ctx, config: cfg)
            stateBytes = cfg.numVHeads * cfg.valueHeadDim * cfg.keyHeadDim * 4
            tailBytes = (cfg.convKernelSize - 1) * cfg.qkvDim * Self.half
            let rows = WideVerifyRollbackTests.prefixRows + WideVerifyRollbackTests.passRows
            convOut = try Self.buffer(ctx, bytes: rows * cfg.qkvDim * Self.half)
            y = try Self.buffer(ctx, bytes: rows * cfg.valueDim * Self.half)
            for layer in WideVerifyRollbackTests.layers {
                var rng = SeedTree(0x5EED_0009).key("wide-verify-rollback-\(layer)")
                func halves(_ count: Int, _ range: ClosedRange<Float>) -> [Float16] {
                    (0..<count).map { _ in Float16(rng.uniform(range.lowerBound, range.upperBound))
                    }
                }
                func bf16(_ count: Int, _ range: ClosedRange<Float>) -> [UInt16] {
                    (0..<count).map { _ in
                        UInt16(
                            truncatingIfNeeded:
                                rng.uniform(range.lowerBound, range.upperBound).bitPattern >> 16)
                    }
                }
                convW[layer] = try Self.buffer(
                    ctx, bf16(cfg.qkvDim * cfg.convKernelSize, -0.4...0.4))
                aLog[layer] = try Self.buffer(ctx, bf16(cfg.numVHeads, -1.0...1.5))
                dtBias[layer] = try Self.buffer(ctx, bf16(cfg.numVHeads, -0.5...0.5))
                qkv[layer] = try Self.buffer(ctx, halves(rows * cfg.qkvDim, -1...1))
                a[layer] = try Self.buffer(ctx, halves(rows * cfg.numVHeads, -1...1))
                b[layer] = try Self.buffer(ctx, halves(rows * cfg.numVHeads, -1...1))
                // Slot 0 holds a sentinel pattern no pass may touch.
                let stateBuffer = try Self.buffer(ctx, bytes: 2 * stateBytes)
                let tailBuffer = try Self.buffer(ctx, bytes: 2 * tailBytes)
                memset(stateBuffer.contents(), 0x5A, stateBytes)
                memset(tailBuffer.contents(), 0x3C, tailBytes)
                state[layer] = stateBuffer
                tail[layer] = tailBuffer
            }
        }

        private static func buffer(_ ctx: MetalContext, bytes: Int) throws -> MTLBuffer {
            guard let buffer = ctx.device.makeBuffer(length: bytes, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed("test") }
            memset(buffer.contents(), 0, bytes)
            return buffer
        }

        private static func buffer<T>(_ ctx: MetalContext, _ values: [T]) throws -> MTLBuffer {
            let made = values.withUnsafeBytes { raw in
                raw.baseAddress.flatMap {
                    ctx.device.makeBuffer(bytes: $0, length: raw.count, options: .storageModeShared)
                }
            }
            guard let made else { throw MetalError.bufferAllocationFailed("test") }
            return made
        }

        func live(_ layer: Int) throws -> WideVerifyGDNRollback.LayerState {
            guard let state = state[layer], let tail = tail[layer] else {
                throw WideVerifyRollbackError.unknownLayer(layer)
            }
            return .init(
                layer: layer,
                state: .init(state, offset: WideVerifyRollbackTests.slot * stateBytes),
                tail: .init(tail, offset: WideVerifyRollbackTests.slot * tailBytes))
        }

        func replay(_ layer: Int) throws -> WideVerifyGDNRollback.LayerReplay {
            guard let convW = convW[layer], let aLog = aLog[layer], let dtBias = dtBias[layer]
            else { throw WideVerifyRollbackError.unknownLayer(layer) }
            return .init(
                live: try live(layer), convWeight: .init(convW),
                aLog: .init(aLog), dtBias: .init(dtBias))
        }

        /// The chunked-prefill chain over rows `first..<first + count`.
        func encodeChunk(_ cb: MTLCommandBuffer, layer: Int, first: Int, count: Int) throws {
            let live = try live(layer)
            let replay = try replay(layer)
            guard let qkv = qkv[layer], let a = a[layer], let b = b[layer] else { return }
            try gdn.encodeConvPrefill(
                commandBuffer: cb, tail: live.tail.buffer, tailOffset: live.tail.offset,
                qkvRows: qkv, qkvRowsOffset: first * qkvRowBytes,
                convWeight: replay.convWeight.buffer, convWeightOffset: 0,
                out: convOut, rows: count)
            try gdn.encodeConvTailUpdate(
                commandBuffer: cb, tail: live.tail.buffer, tailOffset: live.tail.offset,
                qkvRows: qkv, qkvRowsOffset: first * qkvRowBytes, rows: count)
            try gdn.encodeQKNorm(commandBuffer: cb, convOut: convOut, rows: count)
            try gdn.encodeDeltaStepPrefill(
                commandBuffer: cb, convOut: convOut,
                aProj: a, aProjOffset: first * headRowBytes,
                bProj: b, bProjOffset: first * headRowBytes,
                aLog: replay.aLog.buffer, aLogOffset: 0,
                dtBias: replay.dtBias.buffer, dtBiasOffset: 0,
                state: live.state.buffer, stateOffset: live.state.offset,
                y: y, rows: count)
        }

        /// Scalar decode's chain for one row.
        func encodeDecodeRow(_ cb: MTLCommandBuffer, layer: Int, row: Int) throws {
            let live = try live(layer)
            let replay = try replay(layer)
            guard let qkv = qkv[layer], let a = a[layer], let b = b[layer] else { return }
            try gdn.encodeConvDecode(
                commandBuffer: cb, tail: live.tail.buffer, tailOffset: live.tail.offset,
                qkv: qkv, qkvOffset: row * qkvRowBytes,
                convWeight: replay.convWeight.buffer, convWeightOffset: 0, out: convOut)
            try gdn.encodeQKNorm(commandBuffer: cb, convOut: convOut)
            try gdn.encodeDeltaStepDecode(
                commandBuffer: cb, convOut: convOut,
                aProj: a, aProjOffset: row * headRowBytes,
                bProj: b, bProjOffset: row * headRowBytes,
                aLog: replay.aLog.buffer, aLogOffset: 0,
                dtBias: replay.dtBias.buffer, dtBiasOffset: 0,
                state: live.state.buffer, stateOffset: live.state.offset, y: y)
        }

        func run(_ body: (MTLCommandBuffer) throws -> Void) throws {
            guard let cb = ctx.queue.makeCommandBuffer() else { throw MetalError.noQueue }
            try body(cb)
            cb.commit()
            cb.waitUntilCompleted()
            #expect(
                cb.status == .completed, "command buffer failed: \(String(describing: cb.error))")
        }

        /// Both slots of every layer's state and tail, byte for byte.
        func bytes() -> [Int: [UInt8]] {
            var out: [Int: [UInt8]] = [:]
            for layer in WideVerifyRollbackTests.layers {
                guard let state = state[layer], let tail = tail[layer] else { continue }
                out[layer] =
                    Array(UnsafeRawBufferPointer(start: state.contents(), count: state.length))
                    + Array(UnsafeRawBufferPointer(start: tail.contents(), count: tail.length))
            }
            return out
        }

        func window(rows: Int) throws -> WideVerifyGDNRollback {
            let window = try WideVerifyGDNRollback(
                device: ctx.device, config: WideVerifyRollbackTests.cfg,
                layers: WideVerifyRollbackTests.layers, capacityRows: rows)
            try window.open(
                rows: rows, slot: WideVerifyRollbackTests.slot,
                startPosition: WideVerifyRollbackTests.prefixRows, pleContext: [])
            return window
        }
    }

    /// A rig after the shared prefix, so the pass starts from a non-zero state.
    static func primedRig() throws -> Rig {
        let rig = try Rig()
        try rig.run { cb in
            for layer in layers {
                try rig.encodeChunk(cb, layer: layer, first: 0, count: prefixRows)
            }
        }
        return rig
    }

    static func firstMismatch(_ lhs: [Int: [UInt8]], _ rhs: [Int: [UInt8]]) -> String? {
        for layer in layers {
            let left = lhs[layer] ?? []
            let right = rhs[layer] ?? []
            guard left.count == right.count else { return "layer \(layer): sizes differ" }
            if let index = left.indices.first(where: { left[$0] != right[$0] }) {
                return "layer \(layer): first differing byte \(index) of \(left.count)"
            }
        }
        return nil
    }

    @Test(arguments: [1, passRows - 1, passRows])
    func chunkedPassRollsBackToAcceptedRows(accepted: Int) throws {
        let rig = try Self.primedRig()
        let window = try rig.window(rows: Self.passRows)
        try rig.run { cb in
            try window.encodeSnapshot(commandBuffer: cb, layers: Self.layers.map(rig.live))
            for layer in Self.layers {
                try rig.encodeChunk(cb, layer: layer, first: Self.prefixRows, count: Self.passRows)
                guard let qkv = rig.qkv[layer], let a = rig.a[layer], let b = rig.b[layer] else {
                    return
                }
                try window.encodeRetain(
                    commandBuffer: cb, layer: layer,
                    qkv: .init(qkv, offset: Self.prefixRows * rig.qkvRowBytes),
                    a: .init(a, offset: Self.prefixRows * rig.headRowBytes),
                    b: .init(b, offset: Self.prefixRows * rig.headRowBytes),
                    rows: Self.passRows)
                #expect(!window.isRetaining(layer: layer))
            }
        }
        let afterPass = rig.bytes()
        var encoded = false
        try rig.run { cb in
            encoded = try window.encodeRollback(
                commandBuffer: cb, gdn: rig.gdn, acceptedRows: accepted,
                layers: Self.layers.map(rig.replay))
        }
        #expect(encoded == (accepted < Self.passRows))

        let reference = try Self.primedRig()
        try reference.run { cb in
            for layer in Self.layers {
                try reference.encodeChunk(cb, layer: layer, first: Self.prefixRows, count: accepted)
            }
        }
        let want = reference.bytes()
        #expect(
            Self.firstMismatch(rig.bytes(), want) == nil,
            "\(Self.firstMismatch(rig.bytes(), want) ?? "")")
        if accepted < Self.passRows {
            // The pass really did move the state past the accepted rows.
            #expect(Self.firstMismatch(afterPass, want) != nil)
        }
    }

    /// The decode-order verify runs decode's kernels one row at a time and
    /// retains each row as it goes; the prefill-kernel replay must land on the
    /// state k decode steps leave.
    @Test(arguments: [1, 4])
    func perRowRetentionMatchesDecodeSteps(accepted: Int) throws {
        let rig = try Self.primedRig()
        let window = try rig.window(rows: Self.passRows)
        try rig.run { cb in
            try window.encodeSnapshot(commandBuffer: cb, layers: Self.layers.map(rig.live))
            for row in Self.prefixRows..<(Self.prefixRows + Self.passRows) {
                for layer in Self.layers {
                    try rig.encodeDecodeRow(cb, layer: layer, row: row)
                    guard let qkv = rig.qkv[layer], let a = rig.a[layer], let b = rig.b[layer]
                    else { return }
                    try window.encodeRetain(
                        commandBuffer: cb, layer: layer,
                        qkv: .init(qkv, offset: row * rig.qkvRowBytes),
                        a: .init(a, offset: row * rig.headRowBytes),
                        b: .init(b, offset: row * rig.headRowBytes),
                        rows: 1)
                }
            }
        }
        try rig.run { cb in
            try window.encodeRollback(
                commandBuffer: cb, gdn: rig.gdn, acceptedRows: accepted,
                layers: Self.layers.map(rig.replay))
        }
        let reference = try Self.primedRig()
        try reference.run { cb in
            for row in Self.prefixRows..<(Self.prefixRows + accepted) {
                for layer in Self.layers {
                    try reference.encodeDecodeRow(cb, layer: layer, row: row)
                }
            }
        }
        let want = reference.bytes()
        #expect(
            Self.firstMismatch(rig.bytes(), want) == nil,
            "\(Self.firstMismatch(rig.bytes(), want) ?? "")")
    }

    @Test func misuseIsRefused() throws {
        let rig = try Self.primedRig()
        let window = try rig.window(rows: 2)
        guard let qkv = rig.qkv[1], let a = rig.a[1], let b = rig.b[1] else {
            Issue.record("rig has no layer 1")
            return
        }
        try rig.run { cb in
            #expect(
                throws: WideVerifyRollbackError.retentionOverflow(
                    layer: 1, retained: 0, adding: 3, rows: 2)
            ) {
                try window.encodeRetain(
                    commandBuffer: cb, layer: 1, qkv: .init(qkv), a: .init(a), b: .init(b),
                    rows: 3)
            }
            #expect(throws: WideVerifyRollbackError.unknownLayer(2)) {
                try window.encodeRetain(
                    commandBuffer: cb, layer: 2, qkv: .init(qkv), a: .init(a), b: .init(b),
                    rows: 1)
            }
            #expect(
                throws: WideVerifyRollbackError.retentionIncomplete(
                    layer: 1, retained: 0, needed: 1)
            ) {
                try window.encodeRollback(
                    commandBuffer: cb, gdn: rig.gdn, acceptedRows: 1,
                    layers: Self.layers.map(rig.replay))
            }
            #expect(throws: WideVerifyRollbackError.missingLayer(4)) {
                try window.encodeRollback(
                    commandBuffer: cb, gdn: rig.gdn, acceptedRows: 1,
                    layers: [try rig.replay(1)])
            }
            #expect(throws: WideVerifyRollbackError.acceptedRowsOutOfRange(accepted: 3, rows: 2)) {
                try window.encodeRollback(
                    commandBuffer: cb, gdn: rig.gdn, acceptedRows: 3,
                    layers: Self.layers.map(rig.replay))
            }
            #expect(throws: WideVerifyRollbackError.rowsOutOfRange(rows: 3, capacity: 2)) {
                try window.open(rows: 3, slot: 1, startPosition: 0, pleContext: [])
            }
            window.close()
            #expect(!window.isRetaining(layer: 1))
            #expect(throws: WideVerifyRollbackError.windowNotOpen) {
                try window.encodeRetain(
                    commandBuffer: cb, layer: 1, qkv: .init(qkv), a: .init(a), b: .init(b),
                    rows: 1)
            }
        }
    }
}
