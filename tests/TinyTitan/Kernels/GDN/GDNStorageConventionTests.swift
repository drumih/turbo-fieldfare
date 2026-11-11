import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The delta step under the two storage conventions.
///
/// A llama.cpp GGUF carries the same layer as a Hugging Face checkpoint with
/// two differences: `ssm_a` is `-exp(A_log)` rather than `A_log`, and every
/// per-value-head tensor is permuted from grouped order (value heads grouped by
/// key head) to tiled order. So the same layer written both ways, run through
/// the matching kernel variants, must produce the same readout once the tiled
/// output is permuted back - and the engine variant run on the tiled layout
/// must not, which is the defect that made the GGUF route's GDN layers wrong.
@Suite struct GDNStorageConventionTests {
    private static let config = LinearAttentionConfig(
        numKHeads: 2, numVHeads: 6, keyHeadDim: 32, valueHeadDim: 32,
        convKernelSize: 4, outputGate: .sigmoid)

    /// Tiled position j holds grouped head `(j % Hk) * (Hv / Hk) + j / Hk`.
    private static func groupedHead(ofTiled j: Int) -> Int {
        let ratio = config.numVHeads / config.numKHeads
        return (j % config.numKHeads) * ratio + j / config.numKHeads
    }

    private struct Layer {
        var convRows: [[Float]]
        var a: [[Float]]
        var b: [[Float]]
        var aLog: [Float]
        var dtBias: [Float]
    }

    private static func bf16Bits(_ value: Float) -> UInt16 {
        Quantization.bf16Bits(value)
    }

    private static func bf16(_ value: Float) -> Float {
        Float(bitPattern: UInt32(bf16Bits(value)) << 16)
    }

    private static func makeLayer() -> Layer {
        // A fixed LCG: the fixture is the same on every run.
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Double(state >> 11) / Double(1 << 53)) * 2 - 1
        }
        let c = config
        let channels = 2 * c.numKHeads * c.keyHeadDim + c.numVHeads * c.valueHeadDim
        func row() -> [Float] {
            var values = (0..<channels).map { _ in next() * 0.5 }
            // q and k arrive l2-normalised per head, as the qk-norm kernel leaves them.
            for head in 0..<(2 * c.numKHeads) {
                let range = (head * c.keyHeadDim)..<((head + 1) * c.keyHeadDim)
                let norm = sqrt(range.reduce(Float(0)) { $0 + values[$1] * values[$1] })
                for index in range { values[index] /= norm }
            }
            return values.map { Float(Float16($0)) }
        }
        return Layer(
            convRows: [row(), row(), row()],
            a: (0..<3).map { _ in (0..<c.numVHeads).map { _ in Float(Float16(next())) } },
            b: (0..<3).map { _ in (0..<c.numVHeads).map { _ in Float(Float16(next())) } },
            aLog: (0..<c.numVHeads).map { _ in bf16(next()) },
            dtBias: (0..<c.numVHeads).map { _ in bf16(next() * 0.5) })
    }

    /// The same layer in GGUF layout: value heads tiled, `ssm_a = -exp(A_log)`.
    private static func tiled(_ layer: Layer) -> Layer {
        let c = config
        let valueStart = 2 * c.numKHeads * c.keyHeadDim
        func permuteHeads(_ values: [Float]) -> [Float] {
            (0..<c.numVHeads).map { values[groupedHead(ofTiled: $0)] }
        }
        func permuteRow(_ row: [Float]) -> [Float] {
            var out = Array(row[0..<valueStart])
            for j in 0..<c.numVHeads {
                let g = groupedHead(ofTiled: j)
                let source = valueStart + g * c.valueHeadDim
                out += row[source..<(source + c.valueHeadDim)]
            }
            return out
        }
        return Layer(
            convRows: layer.convRows.map(permuteRow),
            a: layer.a.map(permuteHeads),
            b: layer.b.map(permuteHeads),
            aLog: permuteHeads(layer.aLog).map { bf16(-exp($0)) },
            dtBias: permuteHeads(layer.dtBias))
    }

    /// Runs the layer's rows through the delta step and returns every row's
    /// readout, `[Hv * Dv]` per row.
    private static func run(_ layer: Layer, storage: GDNStorageConvention) throws -> [[Float]] {
        let context = try MetalContext()
        let gdn = try GDN(context: context, config: config, storage: storage)
        let device = context.device
        func half(_ values: [Float]) throws -> MTLBuffer {
            let bits = values.map { Float16($0).bitPattern }
            guard
                let buffer = device.makeBuffer(
                    bytes: bits, length: bits.count * 2, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed("half") }
            return buffer
        }
        func bfloat(_ values: [Float]) throws -> MTLBuffer {
            let bits = values.map(bf16Bits)
            guard
                let buffer = device.makeBuffer(
                    bytes: bits, length: bits.count * 2, options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed("bf16") }
            return buffer
        }
        let c = config
        let valueWidth = c.numVHeads * c.valueHeadDim
        guard
            let state = device.makeBuffer(
                length: c.numVHeads * c.valueHeadDim * c.keyHeadDim * 4,
                options: .storageModeShared),
            let y = device.makeBuffer(length: valueWidth * 2, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("state") }
        memset(state.contents(), 0, state.length)
        let aLog = try bfloat(layer.aLog)
        let dtBias = try bfloat(layer.dtBias)
        var readouts: [[Float]] = []
        for index in layer.convRows.indices {
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try gdn.encodeDeltaStepDecode(
                commandBuffer: commandBuffer, convOut: try half(layer.convRows[index]),
                aProj: try half(layer.a[index]), bProj: try half(layer.b[index]),
                aLog: aLog, aLogOffset: 0, dtBias: dtBias, dtBiasOffset: 0,
                state: state, y: y)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let pointer = y.contents().assumingMemoryBound(to: Float16.self)
            readouts.append((0..<valueWidth).map { Float(pointer[$0]) })
        }
        return readouts
    }

    /// Tiled readout back to grouped order, head by head.
    private static func ungroup(_ readout: [Float]) -> [Float] {
        let c = config
        var out = [Float](repeating: 0, count: readout.count)
        for j in 0..<c.numVHeads {
            let g = groupedHead(ofTiled: j)
            for d in 0..<c.valueHeadDim {
                out[g * c.valueHeadDim + d] = readout[j * c.valueHeadDim + d]
            }
        }
        return out
    }

    private static func relativeError(_ a: [Float], _ b: [Float]) -> Float {
        let difference = zip(a, b).reduce(Float(0)) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }
        let norm = b.reduce(Float(0)) { $0 + $1 * $1 }
        return sqrt(difference / max(norm, 1e-20))
    }

    @Test func conventionFlagsMatchTheStorage() {
        #expect(!GDNStorageConvention.engine.aLogPreexponentiated)
        #expect(!GDNStorageConvention.engine.valueHeadsTiled)
        #expect(GDNStorageConvention.llamaCppGGUF.aLogPreexponentiated)
        #expect(GDNStorageConvention.llamaCppGGUF.valueHeadsTiled)
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil))
    func ggufLayoutMatchesEngineLayoutOverThreeSteps() throws {
        let layer = Self.makeLayer()
        let engine = try Self.run(layer, storage: .engine)
        let gguf = try Self.run(Self.tiled(layer), storage: .llamaCppGGUF)
        for (step, (grouped, tiled)) in zip(engine, gguf).enumerated() {
            // The only difference left is `ssm_a` rounded to bf16 after the
            // exp, against `A_log` rounded before it.
            let error = Self.relativeError(Self.ungroup(tiled), grouped)
            #expect(error < 1e-2, "step \(step): relative error \(error)")
        }
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil))
    func engineKernelsMisreadTheGGUFLayout() throws {
        let layer = Self.makeLayer()
        let engine = try Self.run(layer, storage: .engine)
        let misread = try Self.run(Self.tiled(layer), storage: .engine)
        let error = Self.relativeError(Self.ungroup(misread[2]), engine[2])
        #expect(error > 0.1, "the wrong convention should not agree: \(error)")
    }
}
