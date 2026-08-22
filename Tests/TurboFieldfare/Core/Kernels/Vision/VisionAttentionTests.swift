import Metal
import Testing
@testable import TurboFieldfare

private let visionMPPTensorOpsAvailable = MTLCreateSystemDefaultDevice()?
    .supportsFamily(.apple8) == true

@Suite struct VisionAttentionTests {
    /// The layout is the caller's decision, signalled by the row stride: the
    /// runtime fuses only when the linear path can also write the padded
    /// head-major store, so an attention that *could* fuse still receives
    /// row-major Q/K/V when it cannot. Reading those as 80-stride head-major
    /// used to fail a precondition on every image whenever the register MLP or
    /// the MLX GEMM override was set on its own.
    @Test(.enabled(if: visionMPPTensorOpsAvailable,
                   "vision MPP tensor shapes require Apple8 or newer"))
    func rowMajorInputsAreHonouredEvenWhenTheFusedLayoutIsAvailable() throws {
        let context = try MetalContext()
        let attention = try VisionAttention(
            context: context,
            environment: ["TURBO_FIELDFARE_VISION_ATTENTION_MPP": "1"])
        try #require(attention.usesFusedPaddedLayout,
                     "this configuration does not fuse, so there is nothing to test")

        // 200 spans four 64-key tiles; the reference test validates both paths
        // against FP32 at this length, so any divergence here is the layout.
        let length = 200
        let heads = 2
        let dimension = VisionAttention.headDimension
        var q = [Float](repeating: 0, count: length * heads * dimension)
        var k = q
        var v = q
        for index in q.indices {
            q[index] = Float((index * 13) % 17 - 8) / 32
            k[index] = Float((index * 7) % 19 - 9) / 32
            v[index] = Float((index * 5) % 23 - 11) / 16
        }
        func make(_ values: [Float]) throws -> MTLBuffer {
            let bits = values.map(Quantization.bf16Bits)
            return try #require(context.device.makeBuffer(
                bytes: bits, length: bits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
        }
        // The same values laid out head-major with the 80-lane stride, which is
        // what the fused branch expects.
        let padded = VisionAttention.paddedHeadDimension
        func makePadded(_ values: [Float]) throws -> MTLBuffer {
            var laid = [Float](repeating: 0, count: heads * length * padded)
            for row in 0..<length {
                for head in 0..<heads {
                    for d in 0..<dimension {
                        laid[(head * length + row) * padded + d] =
                            values[(row * heads + head) * dimension + d]
                    }
                }
            }
            return try make(laid)
        }

        func run(rowMajor: Bool) throws -> [Float] {
            let output = try #require(context.device.makeBuffer(
                length: q.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            attention.encode(
                commandBuffer: commandBuffer,
                q: rowMajor ? try make(q) : try makePadded(q),
                k: rowMajor ? try make(k) : try makePadded(k),
                v: rowMajor ? try make(v) : try makePadded(v),
                output: output,
                sequenceLength: length, numHeads: heads,
                inputRowStride: rowMajor ? 0 : length)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)
            let bits = output.contents().bindMemory(to: UInt16.self, capacity: q.count)
            return (0..<q.count).map { Quantization.bf16ToFloat(bits[$0]) }
        }

        // Same attention, same values, two layouts. A layout misread is a gross
        // error, so this compares the two paths against each other rather than
        // against a tolerance on an FP32 reference.
        let fallback = try run(rowMajor: true)
        let fused = try run(rowMajor: false)
        // The tolerance below is wider than the signal in this fixture, so it
        // cannot tell a correct output from an empty one. It let a fallback
        // that wrote only half its rows pass. Every row must be written.
        for row in 0..<length {
            let start = row * heads * dimension
            let slice = fallback[start..<(start + heads * dimension)]
            #expect(slice.contains { $0 != 0 },
                    "row \(row) of the fallback output was never written")
        }
        for index in fallback.indices {
            let detail = "row-major fallback diverged at \(index): "
                + "\(fallback[index]) vs fused \(fused[index])"
            #expect(abs(fallback[index] - fused[index]) <= 0.02, "\(detail)")
        }
    }

    /// When the linear path cannot write the padded head-major store, the
    /// runtime constructs attention with `allowFusedPaddedLayout: false`. The
    /// MPP kernel the environment asked for must still run — through the
    /// pad/unpad roundtrip — rather than silently degrading to the native
    /// kernel while `variant` reported otherwise.
    @Test(.enabled(if: visionMPPTensorOpsAvailable,
                   "vision MPP tensor shapes require Apple8 or newer"))
    func mppWithoutThePaddedStoreStillRunsTheMPPKernel() throws {
        let context = try MetalContext()
        let attention = try VisionAttention(
            context: context,
            environment: ["TURBO_FIELDFARE_VISION_ATTENTION_MPP": "1"],
            allowFusedPaddedLayout: false)
        #expect(attention.variant == .mppTensorOps72)
        #expect(!attention.usesFusedPaddedLayout)
        #expect(attention.fusedQ == nil,
                "a row-major caller must not be handed fused destination buffers")

        let length = 200
        let heads = 2
        let dimension = VisionAttention.headDimension
        var q = [Float](repeating: 0, count: length * heads * dimension)
        var k = q
        var v = q
        for index in q.indices {
            q[index] = Float((index * 13) % 17 - 8) / 32
            k[index] = Float((index * 7) % 19 - 9) / 32
            v[index] = Float((index * 5) % 23 - 11) / 16
        }
        func make(_ values: [Float]) throws -> MTLBuffer {
            let bits = values.map(Quantization.bf16Bits)
            return try #require(context.device.makeBuffer(
                bytes: bits, length: bits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
        }
        let qBuffer = try make(q), kBuffer = try make(k), vBuffer = try make(v)

        func run(_ attention: VisionAttention) throws -> ([Float], [VisionAttention.Phase]) {
            let output = try #require(context.device.makeBuffer(
                length: q.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
            var phases: [VisionAttention.Phase] = []
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            attention.encodePhased(
                q: qBuffer, k: kBuffer, v: vBuffer, output: output,
                sequenceLength: length, numHeads: heads,
                inputRowStride: 0) { phase, body in
                phases.append(phase)
                body(commandBuffer)
            }
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)
            let bits = output.contents().bindMemory(to: UInt16.self, capacity: q.count)
            return ((0..<q.count).map { Quantization.bf16ToFloat(bits[$0]) }, phases)
        }

        // Pad and unpad running is the proof the padded MPP kernel executed;
        // the native fallback this used to hit runs `.core` alone.
        let (roundtrip, phases) = try run(attention)
        #expect(phases == [.pad, .core, .unpad],
                "the MPP roundtrip did not run; phases were \(phases)")

        // Cross-checked against the native kernel, which shares no layout code.
        let (native, nativePhases) = try run(
            try VisionAttention(context: context, environment: [:]))
        #expect(nativePhases == [.core])
        for index in roundtrip.indices {
            #expect(abs(roundtrip[index] - native[index]) <= 0.02,
                    "roundtrip diverged at \(index): \(roundtrip[index]) vs native \(native[index])")
        }
    }

    /// 5 fits one 64-key tile; 200 spans four tiles and is the only case that
    /// exercises the online rescale between tiles.
    @Test(arguments: [5, 200])
    func onlineAttentionMatchesFP32Reference(length: Int) throws {
        let context = try MetalContext()
        let heads = 2
        let dimension = VisionAttention.headDimension
        var q = [Float](repeating: 0, count: length * heads * dimension)
        var k = q
        var v = q
        for index in q.indices {
            q[index] = Float((index * 13) % 17 - 8) / 32
            k[index] = Float((index * 7) % 19 - 9) / 32
            v[index] = Float((index * 5) % 23 - 11) / 16
        }
        func make(_ values: [Float]) throws -> MTLBuffer {
            let bits = values.map(Quantization.bf16Bits)
            return try #require(context.device.makeBuffer(
                bytes: bits, length: bits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
        }
        let qBuffer = try make(q)
        let kBuffer = try make(k)
        let vBuffer = try make(v)
        // Head-major [head][row][80] copies for the fused layout, which reads the
        // padding the projections would otherwise have written.
        let padded = VisionAttention.paddedHeadDimension
        func makePadded(_ values: [Float]) throws -> MTLBuffer {
            var laid = [Float](repeating: 0, count: heads * length * padded)
            for row in 0..<length {
                for head in 0..<heads {
                    for d in 0..<dimension {
                        laid[(head * length + row) * padded + d] =
                            values[(row * heads + head) * dimension + d]
                    }
                }
            }
            return try make(laid)
        }
        let paddedQBuffer = try makePadded(q)
        let paddedKBuffer = try makePadded(k)
        let paddedVBuffer = try makePadded(v)
        let output = try #require(context.device.makeBuffer(
            length: q.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))

        var environments: [[String: String]] = [
            ["TURBO_FIELDFARE_VISION_ATTENTION_Q8": "1"],
            [:],
            ["TURBO_FIELDFARE_VISION_ATTENTION_PAD80": "1"],
        ]
        if visionMPPTensorOpsAvailable {
            environments.append(contentsOf: [
                ["TURBO_FIELDFARE_VISION_ATTENTION_MPP": "1"],
                ["TURBO_FIELDFARE_VISION_ATTENTION_MPP": "1",
                 "TURBO_FIELDFARE_VISION_PAD_ROUNDTRIP": "1"],
                ["TURBO_FIELDFARE_VISION_ATTENTION_MPP": "1",
                 "TURBO_FIELDFARE_VISION_ATTENTION_PARALLEL_SOFTMAX": "1"],
                ["TURBO_FIELDFARE_VISION_ATTENTION_MPP": "1",
                 "TURBO_FIELDFARE_VISION_ATTENTION_SERIAL_SOFTMAX": "1"],
            ])
        }
        for environment in environments {
            let attention = try VisionAttention(
                context: context,
                environment: environment)
            let fused = attention.usesFusedPaddedLayout
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            attention.encode(commandBuffer: commandBuffer,
                             q: fused ? paddedQBuffer : qBuffer,
                             k: fused ? paddedKBuffer : kBuffer,
                             v: fused ? paddedVBuffer : vBuffer,
                             output: output,
                             sequenceLength: length, numHeads: heads,
                             inputRowStride: fused ? length : 0)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)

            let actualBits = output.contents().bindMemory(
                to: UInt16.self, capacity: q.count)
            for query in 0..<length {
                for head in 0..<heads {
                    var scores = [Float](repeating: 0, count: length)
                    for key in 0..<length {
                        for d in 0..<dimension {
                            scores[key] += q[(query * heads + head) * dimension + d]
                                * k[(key * heads + head) * dimension + d]
                        }
                    }
                    let maximum = scores.max()!
                    let weights = scores.map { exp($0 - maximum) }
                    let denominator = weights.reduce(0, +)
                    for d in 0..<dimension {
                        var expected: Float = 0
                        for key in 0..<length {
                            expected += weights[key]
                                * v[(key * heads + head) * dimension + d]
                        }
                        expected /= denominator
                        let index = (query * heads + head) * dimension + d
                        let actual = Quantization.bf16ToFloat(actualBits[index])
                        #expect(abs(actual - expected) <= 0.02,
                                "variant=\(attention.variant) index=\(index) actual=\(actual) expected=\(expected)")
                    }
                }
            }
        }
    }

    /// The parallel softmax phase changes only the tile-sum reduction order, so at
    /// production shape it must stay within BF16 rounding of the serial phase.
    /// A layout or lane-grouping mistake would show here as a gross difference.
    @Test(.enabled(if: visionMPPTensorOpsAvailable,
                   "vision MPP tensor shapes require Apple8 or newer"))
    func parallelSoftmaxStaysWithinRoundingOfSerialPhase() throws {
        let context = try MetalContext()
        let length = 512
        let heads = 4
        let dimension = VisionAttention.headDimension
        let padded = VisionAttention.paddedHeadDimension
        var values = [Float](repeating: 0, count: heads * length * padded)
        var state: UInt64 = 0x9E3779B97F4A7C15
        for head in 0..<heads {
            for row in 0..<length {
                for d in 0..<dimension {
                    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    let unit = Float(state >> 40) / Float(1 << 24) - 0.5
                    values[(head * length + row) * padded + d] = unit
                }
            }
        }
        func make() throws -> MTLBuffer {
            let bits = values.map(Quantization.bf16Bits)
            return try #require(context.device.makeBuffer(
                bytes: bits, length: bits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
        }
        let q = try make(), k = try make(), v = try make()

        func run(parallel: Bool) throws -> [Float] {
            var environment = ["TURBO_FIELDFARE_VISION_ATTENTION_MPP": "1"]
            environment[parallel
                ? "TURBO_FIELDFARE_VISION_ATTENTION_PARALLEL_SOFTMAX"
                : "TURBO_FIELDFARE_VISION_ATTENTION_SERIAL_SOFTMAX"] = "1"
            let attention = try VisionAttention(context: context, environment: environment)
            #expect(attention.usesParallelSoftmax == parallel)
            let count = length * heads * dimension
            let output = try #require(context.device.makeBuffer(
                length: count * MemoryLayout<UInt16>.stride, options: .storageModeShared))
            let commandBuffer = try #require(context.queue.makeCommandBuffer())
            attention.encode(commandBuffer: commandBuffer, q: q, k: k, v: v,
                             output: output, sequenceLength: length, numHeads: heads,
                             inputRowStride: length)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            #expect(commandBuffer.error == nil)
            let bits = output.contents().bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { Quantization.bf16ToFloat(bits[$0]) }
        }

        let serial = try run(parallel: false)
        let parallel = try run(parallel: true)
        var maximumDifference: Float = 0
        var magnitude: Float = 0
        for index in serial.indices {
            maximumDifference = max(maximumDifference, abs(serial[index] - parallel[index]))
            magnitude = max(magnitude, abs(serial[index]))
        }
        #expect(magnitude > 0)
        #expect(maximumDifference <= 0.01 * magnitude,
                "max difference \(maximumDifference) against magnitude \(magnitude)")
    }
}
