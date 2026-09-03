import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct MoEFusedFFNTests {
    private static let dimension = 128
    private static let intermediate = 64
    private static let topK = 8

    private struct RoutedBlob {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    @Test func productionRoutedPipelineAndHitSplitMatchReference() throws {
        var rng = SeedTree(0x2D3).key("production-routed-moe")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }

        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<Self.topK {
            gates.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            ups.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            downs.append(matrix(rows: Self.dimension, columns: Self.intermediate))
        }
        let x = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let residual = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let routingWeights = (0..<Self.topK).map {
            Float(Float16(0.04 + Float($0) * 0.015))
        }
        let expected = MoeRef.applyStreamedRouted(
            x: x,
            residual: residual,
            routedGate: gates.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedUp: ups.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedDown: downs.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            indices: Array(0..<Self.topK),
            routingWeights: routingWeights,
            d: Self.dimension,
            f: Self.intermediate)
        let blobs = (0..<Self.topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
        }

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == Self.topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let fullActs = Fp16Buffer.make(
                context.device, count: Self.topK * Self.intermediate),
              let splitActs = Fp16Buffer.make(
                context.device, count: Self.topK * Self.intermediate),
              let fullOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let splitOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let lowSlots = context.device.makeBuffer(
                bytes: [UInt32](0...3),
                length: 4 * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let highSlots = context.device.makeBuffer(
                bytes: [UInt32](4...7),
                length: 4 * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers,
                topK: UInt32(Self.topK)) else {
            Issue.record("buffer allocation failed")
            return
        }

        let fullCommand = context.queue.makeCommandBuffer()!
        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: fullActs,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: fullActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: fullOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        fullCommand.commit()
        fullCommand.waitUntilCompleted()
        #expect(fullCommand.error == nil)

        let splitCommand = context.queue.makeCommandBuffer()!
        for (slots, activeSlots) in [([UInt32](0...3), lowSlots),
                                     ([UInt32](4...7), highSlots)] {
            kernel.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: splitCommand,
                routedArgBuffer: argumentBuffer,
                routedBlobs: routedBuffers,
                routedOffsets: blobs[0].offsets,
                x: xBuffer,
                acts: splitActs,
                activeSlots: activeSlots,
                activeSlotIndices: slots,
                activeCount: UInt32(slots.count),
                d: UInt32(Self.dimension),
                f: UInt32(Self.intermediate),
                topK: UInt32(Self.topK))
        }
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: splitCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: splitActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: splitOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        splitCommand.commit()
        splitCommand.waitUntilCompleted()
        #expect(splitCommand.error == nil)

        let full = Fp16Buffer.read(fullOutput, count: Self.dimension)
        let split = Fp16Buffer.read(splitOutput, count: Self.dimension)
        #expect(full == split)
        #expect(RelError.compute(actual: full, reference: expected)
            < Tolerance.fp16ChainedReduction)
    }

    // MARK: - Routed width

    private struct WidthFixture {
        let blobs: [RoutedBlob]
        let gates: [[[Float]]]
        let ups: [[[Float]]]
        let downs: [[[Float]]]
        let x: [Float]
        let residual: [Float]
        let routingWeights: [Float]
    }

    /// Eight routed slots with descending routing weights, from which each
    /// width under test takes its first K. Deterministic, so the K=8 row is a
    /// fixture rather than a fresh random draw each run.
    private static func widthFixture() -> WidthFixture {
        var rng = SeedTree(0x167).key("routed-width")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }
        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<Self.topK {
            gates.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            ups.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            downs.append(matrix(rows: Self.dimension, columns: Self.intermediate))
        }
        return WidthFixture(
            blobs: (0..<Self.topK).map {
                Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
            },
            gates: gates,
            ups: ups,
            downs: downs,
            x: (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) },
            residual: (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) },
            routingWeights: (0..<Self.topK).map {
                Float(Float16(0.20 - Float($0) * 0.02))
            })
    }

    /// Recorded from `moe_phase2_down_reduce_k8` as it stands in private `main`
    /// at `680aa478`, where `public/.../Metal/MoE/moe.metal` is still the
    /// pre-generalisation file; check out that commit's `public/` tree to re-run
    /// the capture. At the checkpoint's width the generalised kernel sums the
    /// same eight partials in the same order, so it must reproduce these FP16
    /// bits exactly. Regenerate only alongside a deliberate change to the
    /// reduction's arithmetic.
    private static let widthFixtureOutputHexAtEight =
        "b90d3188bac23a0b361fb80c264abb43b961280b2c52b2f9b03b3517ba722c3a" +
        "39332f37ad7b2f3db295a708b8f5b7f035463362a0bdb308b731acf732bc3187" +
        "2791b3ed2e50b373b8a0375935e838ab34103259304c3795b5fb2b26b5841e6f" +
        "2d80348f2beeb5293764b49f29f735b835e329b9302fb45cb8a5ab1d25b2b313" +
        "2bfe215831aeb3683705b773b34d2c40ad11366f382da80ab1aeb20938a4b50c" +
        "b810b03b32b830feb8aab1c7af1136b0afd6b9a73918b8a635f62d79355b399a" +
        "ba33b38a343838efac48ac0236ed9dc826e8b40e2e3db852b55134552d51350f" +
        "b4492b651e85b320b8712cffab973602ad5cb54ab96b38c03a06b3aab661349b"

    @Test func phase2ReduceAtTheCheckpointWidthReproducesThePreGeneralisationFixture() throws {
        let fixture = Self.widthFixture()
        let output = try Self.runRoutedWidth(fixture: fixture, topK: Self.topK)
        #expect(Self.hex(output) == Self.widthFixtureOutputHexAtEight)
    }

    /// Phase 1 writes `K * F` rows and phase 2 reduces `K` partials, so a
    /// narrower width changes both the grid and the number of SIMD groups in
    /// the reduce threadgroup. Every group has to reach the barrier, which is
    /// why the dispatch is exactly `32 * K` threads rather than a fixed 256
    /// with an early return.
    @Test(arguments: [4, 6, 8])
    func routedPipelineMatchesReferenceAtEveryRoutedWidth(_ topK: Int) throws {
        let fixture = Self.widthFixture()
        let output = try Self.runRoutedWidth(fixture: fixture, topK: topK)
        let actual = output.map { Float(Float16(bitPattern: $0)) }
        let expected = MoeRef.applyStreamedRouted(
            x: fixture.x,
            residual: fixture.residual,
            routedGate: fixture.gates.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedUp: fixture.ups.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedDown: fixture.downs.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            indices: Array(0..<topK),
            routingWeights: Array(fixture.routingWeights.prefix(topK)),
            d: Self.dimension,
            f: Self.intermediate)
        #expect(RelError.compute(actual: actual, reference: expected)
            < Tolerance.fp16ChainedReduction)
    }

    private static func runRoutedWidth(fixture: WidthFixture,
                                       topK: Int) throws -> [UInt16] {
        let context = try MetalContext()
        let kernel = try MoE(context: context)
        let routedBuffers = fixture.blobs.prefix(topK).compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == topK,
              let xBuffer = Fp16Buffer.make(context.device, values: fixture.x),
              let residualBuffer = Fp16Buffer.make(context.device, values: fixture.residual),
              let routingBuffer = Fp16Buffer.make(
                context.device, values: Array(fixture.routingWeights.prefix(topK))),
              let acts = Fp16Buffer.make(context.device, count: topK * Self.intermediate),
              let output = Fp16Buffer.make(context.device, count: Self.dimension),
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers,
                topK: UInt32(topK)),
              let command = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: command,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: fixture.blobs[0].offsets,
            x: xBuffer,
            acts: acts,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(topK))
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: command,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: fixture.blobs[0].offsets,
            acts: acts,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: output,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(topK))
        command.commit()
        command.waitUntilCompleted()
        #expect(command.error == nil)
        let pointer = output.contents().bindMemory(
            to: UInt16.self, capacity: Self.dimension)
        return (0..<Self.dimension).map { pointer[$0] }
    }

    private static func hex(_ values: [UInt16]) -> String {
        values.map { String(format: "%04x", $0) }.joined()
    }

    private static func makeBlob(gate: [[Float]],
                                 up: [[Float]],
                                 down: [[Float]]) -> RoutedBlob {
        func packed(_ rows: [[Float]])
            -> (weights: [UInt8], scales: [UInt16], biases: [UInt16]) {
            let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
            return (quantized.flatMap(\.packed),
                    quantized.flatMap(\.scales),
                    quantized.flatMap(\.biases))
        }
        var bytes = [UInt8]()
        func append(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func append(_ values: [UInt16]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        let gateValues = packed(gate)
        let upValues = packed(up)
        let downValues = packed(down)
        let gateW = UInt32(bytes.count); append(gateValues.weights)
        let gateS = UInt32(bytes.count); append(gateValues.scales)
        let gateB = UInt32(bytes.count); append(gateValues.biases)
        let upW = UInt32(bytes.count); append(upValues.weights)
        let upS = UInt32(bytes.count); append(upValues.scales)
        let upB = UInt32(bytes.count); append(upValues.biases)
        let downW = UInt32(bytes.count); append(downValues.weights)
        let downS = UInt32(bytes.count); append(downValues.scales)
        let downB = UInt32(bytes.count); append(downValues.biases)
        return RoutedBlob(
            bytes: bytes,
            offsets: MoEExpertOffsets(
                gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                upWOff: upW, upSOff: upS, upBOff: upB,
                downWOff: downW, downSOff: downS, downBOff: downB))
    }
}
