import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct MoEFusedFFNTests {
    private static let dimension = 128
    private static let intermediate = 64
    private static let topK = 8

    private struct RoutedBlob {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    @Test(arguments: [2, 4, 8])
    func affineRoutedPipelineSupportsQwenBits(bits: Int) throws {
        let blobs = (0..<Self.topK).map { _ in Self.makeConstantBlob(bits: bits) }
        let context = try MetalContext()
        let kernel = try MoE(
            context: context, siluActivation: true,
            routedWeightBits: bits,
            specializedD: UInt32(Self.dimension),
            specializedF: UInt32(Self.intermediate),
            specializedNumExperts: 256)
        let routed = try blobs.map {
            try #require(context.device.makeBuffer(bytes: $0.bytes, length: $0.bytes.count))
        }
        let x = try #require(
            Fp16Buffer.make(
                context.device, halves: [Float16](repeating: 1, count: Self.dimension)))
        let acts = try #require(
            Fp16Buffer.make(
                context.device, count: Self.topK * Self.intermediate))
        let weights = try #require(
            Fp16Buffer.make(
                context.device, halves: [Float16](repeating: 0.125, count: Self.topK)))
        let residual = try #require(Fp16Buffer.make(context.device, count: Self.dimension))
        let output = try #require(Fp16Buffer.make(context.device, count: Self.dimension))
        memset(residual.contents(), 0, residual.length)
        let args = try #require(
            kernel.makeRoutedArgumentBuffer(routedBlobs: routed, topK: 8))
        let cb = try #require(context.queue.makeCommandBuffer())
        try kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: cb, routedArgBuffer: args, routedBlobs: routed,
            routedOffsets: blobs[0].offsets, x: x, acts: acts,
            d: 128, f: 64, topK: 8)
        try kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: cb, routedArgBuffer: args, routedBlobs: routed,
            routedOffsets: blobs[0].offsets, acts: acts,
            routingWeights: weights, residual: residual, y: output,
            d: 128, f: 64, topK: 8)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let expected = Float(2) / (1 + exp(-Float(2)))
        let actual = Fp16Buffer.read(output, count: Self.dimension)
        #expect(actual.allSatisfy { abs($0 - expected) < 0.01 })
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
            context.device.makeBuffer(
                bytes: $0.bytes,
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
                topK: UInt32(Self.topK))
        else {
            Issue.record("buffer allocation failed")
            return
        }

        let fullCommand = try #require(context.queue.makeCommandBuffer())
        try kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: fullActs,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        try kernel.encodeRoutedPersistentPhase2Reduce(
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

        let splitCommand = try #require(context.queue.makeCommandBuffer())
        for (slots, activeSlots) in [
            ([UInt32](0...3), lowSlots),
            ([UInt32](4...7), highSlots),
        ] {
            try kernel.encodeRoutedPersistentPhase1SubsetU16Load(
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
        try kernel.encodeRoutedPersistentPhase2Reduce(
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
        #expect(
            RelError.compute(actual: full, reference: expected)
                < Tolerance.fp16ChainedReduction)
    }

    @Test("Specialized decode pipelines honour a top-k other than 8")
    func specializedPipelineComputesEveryExpertSlot() throws {
        // The specialized decode pipelines bake D, F and k in as function
        // constants. k was a hardcoded 8 while D and F came from the model,
        // so a family that shares another family's hidden dimensions but
        // routes to a different number of experts took the specialized path
        // and silently computed only the first 8: phase 1 never wrote the
        // remaining activation slots and the reduce summed zeros for them.
        // The output stayed smooth and confident, which is why this needs a
        // test rather than a review.
        let k = 10
        var rng = SeedTree(0x5E1).key("specialized-topk")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in (0..<columns).map { _ in rng.uniform(-0.4, 0.4) } }
        }
        let gates = (0..<k).map { _ in
            matrix(rows: Self.intermediate, columns: Self.dimension)
        }
        let ups = (0..<k).map { _ in
            matrix(rows: Self.intermediate, columns: Self.dimension)
        }
        let downs = (0..<k).map { _ in
            matrix(rows: Self.dimension, columns: Self.intermediate)
        }
        let x = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = [Float](repeating: 0, count: Self.dimension)
        // Descending weights, so dropping the last slots is a small, plausible
        // error rather than an obvious one -- the shape of the real bug.
        let routingWeights = (0..<k).map { Float(Float16(0.2 - Float($0) * 0.015)) }
        let expected = MoeRef.applyStreamedRouted(
            x: x, residual: residual,
            routedGate: gates.map { $0.map { Quantization.quantizeInt4Affine($0) } },
            routedUp: ups.map { $0.map { Quantization.quantizeInt4Affine($0) } },
            routedDown: downs.map { $0.map { Quantization.quantizeInt4Affine($0) } },
            indices: Array(0..<k), routingWeights: routingWeights,
            d: Self.dimension, f: Self.intermediate)

        let blobs = (0..<k).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
        }
        let context = try MetalContext()
        // Specializing on this test's own shape is what forces the
        // constant-folded pipelines to be the ones exercised.
        let kernel = try MoE(
            context: context,
            specializedD: UInt32(Self.dimension),
            specializedF: UInt32(Self.intermediate),
            topKExperts: k)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(
                bytes: $0.bytes, length: $0.bytes.count,
                options: .storageModeShared)
        }
        guard routedBuffers.count == k,
            let xBuffer = Fp16Buffer.make(context.device, values: x),
            let residualBuffer = Fp16Buffer.make(context.device, values: residual),
            let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
            let acts = Fp16Buffer.make(context.device, count: k * Self.intermediate),
            let output = Fp16Buffer.make(context.device, count: Self.dimension),
            let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers, topK: UInt32(k))
        else {
            Issue.record("buffer allocation failed")
            return
        }
        let command = try #require(context.queue.makeCommandBuffer())
        try kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: command, routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers, routedOffsets: blobs[0].offsets,
            x: xBuffer, acts: acts,
            d: UInt32(Self.dimension), f: UInt32(Self.intermediate),
            topK: UInt32(k))
        try kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: command, routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers, routedOffsets: blobs[0].offsets,
            acts: acts, routingWeights: routingBuffer, residual: residualBuffer,
            y: output, d: UInt32(Self.dimension), f: UInt32(Self.intermediate),
            topK: UInt32(k))
        command.commit()
        command.waitUntilCompleted()
        #expect(command.error == nil)

        // Every slot must carry real activations. Slots 8 and 9 were exactly
        // zero before the fix, so this names the failure directly.
        let allActs = Fp16Buffer.read(acts, count: k * Self.intermediate)
        for slot in 0..<k {
            let slice = allActs[slot * Self.intermediate..<(slot + 1) * Self.intermediate]
            #expect(
                slice.contains { $0 != 0 },
                "expert slot \(slot) of \(k) was never written")
        }
        #expect(
            RelError.compute(
                actual: Fp16Buffer.read(output, count: Self.dimension),
                reference: expected)
                < Tolerance.fp16ChainedReduction)
    }

    private static func makeBlob(
        gate: [[Float]],
        up: [[Float]],
        down: [[Float]]
    ) -> RoutedBlob {
        func packed(_ rows: [[Float]])
            -> (weights: [UInt8], scales: [UInt16], biases: [UInt16])
        {
            let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
            return (
                quantized.flatMap(\.packed),
                quantized.flatMap(\.scales),
                quantized.flatMap(\.biases)
            )
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
        let gateW = UInt32(bytes.count)
        append(gateValues.weights)
        let gateS = UInt32(bytes.count)
        append(gateValues.scales)
        let gateB = UInt32(bytes.count)
        append(gateValues.biases)
        let upW = UInt32(bytes.count)
        append(upValues.weights)
        let upS = UInt32(bytes.count)
        append(upValues.scales)
        let upB = UInt32(bytes.count)
        append(upValues.biases)
        let downW = UInt32(bytes.count)
        append(downValues.weights)
        let downS = UInt32(bytes.count)
        append(downValues.scales)
        let downB = UInt32(bytes.count)
        append(downValues.biases)
        return RoutedBlob(
            bytes: bytes,
            offsets: MoEExpertOffsets(
                gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                upWOff: upW, upSOff: upS, upBOff: upB,
                downWOff: downW, downSOff: downS, downBOff: downB))
    }

    private static func makeConstantBlob(bits: Int) -> RoutedBlob {
        func bf16(_ value: Float) -> UInt16 {
            UInt16(truncatingIfNeeded: value.bitPattern >> 16)
        }
        var bytes = [UInt8]()
        func appendZeros(rows: Int, cols: Int) -> UInt32 {
            let offset = UInt32(bytes.count)
            bytes += [UInt8](repeating: 0, count: rows * cols * bits / 8)
            return offset
        }
        func appendBF16(_ value: Float, count: Int) -> UInt32 {
            let offset = UInt32(bytes.count)
            let raw = bf16(value)
            for _ in 0..<count {
                bytes.append(UInt8(truncatingIfNeeded: raw))
                bytes.append(UInt8(truncatingIfNeeded: raw >> 8))
            }
            return offset
        }
        let gateW = appendZeros(rows: intermediate, cols: dimension)
        let gateS = appendBF16(0, count: intermediate * dimension / 64)
        let gateB = appendBF16(1.0 / 64.0, count: intermediate * dimension / 64)
        let upW = appendZeros(rows: intermediate, cols: dimension)
        let upS = appendBF16(0, count: intermediate * dimension / 64)
        let upB = appendBF16(0.5 / 64.0, count: intermediate * dimension / 64)
        let downW = appendZeros(rows: dimension, cols: intermediate)
        let downS = appendBF16(0, count: dimension * intermediate / 64)
        let downB = appendBF16(1.0 / 64.0, count: dimension * intermediate / 64)
        return RoutedBlob(
            bytes: bytes,
            offsets: MoEExpertOffsets(
                gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                upWOff: upW, upSOff: upS, upBOff: upB,
                downWOff: downW, downSOff: downS, downBOff: downB))
    }
}
