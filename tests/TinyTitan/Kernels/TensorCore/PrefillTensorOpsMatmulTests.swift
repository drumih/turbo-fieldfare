import Foundation
import Metal
import Testing
import TinyTitanValidationSupport
import TinyTitanCore

@testable import TinyTitan

// MPP (matrix-product pipeline) tensor operations are an Apple10 hardware
// feature. The selection tests below never depend on the host: they drive the
// same TensorOpsPolicy seam from synthetic capabilities and a stub pipeline
// builder. The one arithmetic comparison is gated, because an M3 cannot
// dispatch the kernel at all; a silent pass there would claim coverage for a
// path that never ran.
private let mppTensorOpsAvailable: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPPrefillInt4QMM(context: context).isAvailable
}()

@Suite struct PrefillTensorOpsMatmulTests {
    private struct Matmul {
        let name: String
        let rows: Int
        let columns: Int
    }

    /// The shapes the two call sites feed the kernel: the qualified attention
    /// geometries from the existing MPP suite plus the golden dense model's
    /// FFN geometry (D=2048, F=6144).
    private static let matmuls: [Matmul] = [
        Matmul(name: "q", rows: 4096, columns: 2816),
        Matmul(name: "k", rows: 2048, columns: 2816),
        Matmul(name: "v", rows: 2048, columns: 2816),
        Matmul(name: "o", rows: 2816, columns: 4096),
        Matmul(name: "ffn-gate", rows: 6144, columns: 2048),
        Matmul(name: "ffn-up", rows: 6144, columns: 2048),
        Matmul(name: "ffn-down", rows: 2048, columns: 6144),
    ]

    private static func shape(
        _ matmul: Matmul,
        tokens: Int,
        bits: Int = 4
    ) -> MPPPrefillInt4QMM.Shape {
        MPPPrefillInt4QMM.Shape(
            tokenCount: tokens,
            rows: matmul.rows,
            columns: matmul.columns,
            weightBits: bits,
            xStrideElements: matmul.columns,
            yStrideElements: matmul.rows)
    }

    private static func capabilities(apple10: Bool) -> HardwareCapabilities {
        HardwareCapabilities(
            chipModel: "MacTest,1",
            gpuName: "Synthetic GPU",
            gpuFamily: apple10 ? .apple10 : .apple9,
            supportsApple10TensorOps: apple10,
            metal4Available: apple10,
            anePresent: true,
            physicalMemoryBytes: 24 << 30,
            osVersion: "26.0.0")
    }

    /// Stands in for the `tensorops` pipeline so `isAvailable` is true on every
    /// host. These tests never dispatch it; it only proves the policy reached
    /// the build step with the right function constant.
    private static func stubBuilder(
        _ context: MetalContext
    ) -> (String, [MetalFunctionConstant]) throws -> MTLComputePipelineState {
        { _, constants in
            try context.pipeline(
                "prefill_dequant_affine_qmm_f16_block", constants: constants)
        }
    }

    // MARK: - K5 selection

    /// `off` must not even compile the shader: the decision is fallback before
    /// any pipeline build, and every matmul stays scalar.
    @Test func offNeverBuildsTheMatmulPipeline() throws {
        let context = try MetalContext()
        var builds = 0
        let candidate = MPPPrefillInt4QMM(
            context: context,
            weightBits: 4,
            capabilities: Self.capabilities(apple10: true),
            tensorOpsMode: .off,
            tensorOpsPipelineBuilder: { _, constants in
                builds += 1
                return try context.pipeline(
                    "prefill_dequant_affine_qmm_f16_block", constants: constants)
            })
        #expect(builds == 0)
        #expect(!candidate.isAvailable)
        #expect(candidate.tensorOpsDecision.path == .fallback)
        #expect(candidate.tensorOpsDecision.reason.contains("TINYTITAN_TENSOROPS=off"))
        for matmul in Self.matmuls {
            let selection = candidate.select(Self.shape(matmul, tokens: 64))
            #expect(!selection.selected, "\(matmul.name) selected with tensor ops off")
            #expect(selection.reason.contains("off"), "\(matmul.name): \(selection.reason)")
        }
    }

    /// The capability `auto` decision is per matmul: once the pipeline exists,
    /// every production shape above the row threshold is selected.
    @Test func syntheticApple10SelectsEveryProductionMatmul() throws {
        let context = try MetalContext()
        var builds = 0
        let candidate = MPPPrefillInt4QMM(
            context: context,
            weightBits: 4,
            capabilities: Self.capabilities(apple10: true),
            tensorOpsMode: .auto,
            tensorOpsPipelineBuilder: { name, constants in
                builds += 1
                #expect(name == "mpp_prefill_affine_threadgroup_f16")
                #expect(
                    constants
                        == [MetalFunctionConstant(index: 78, value: .uint32(4))])
                return try context.pipeline(
                    "prefill_dequant_affine_qmm_f16_block", constants: constants)
            })
        #expect(builds == 1)
        #expect(candidate.isAvailable)
        #expect(candidate.tensorOpsDecision.path == .tensorOps)
        #expect(candidate.tensorOpsDecision.reason.contains("Apple10"))
        // 32 is the threshold; 33 and 100 are a partial last row tile each.
        for tokens in [32, 33, 100] {
            for matmul in Self.matmuls {
                let selection = candidate.select(Self.shape(matmul, tokens: tokens))
                #expect(
                    selection.selected,
                    "\(matmul.name) tokens=\(tokens): \(selection.reason)")
            }
        }
    }

    /// The M3 contract: no capability, no build, every matmul scalar.
    @Test func autoWithoutTheCapabilityKeepsEveryMatmulScalar() throws {
        let context = try MetalContext()
        var builds = 0
        let candidate = MPPPrefillInt4QMM(
            context: context,
            weightBits: 4,
            capabilities: Self.capabilities(apple10: false),
            tensorOpsMode: .auto,
            tensorOpsPipelineBuilder: { _, constants in
                builds += 1
                return try context.pipeline(
                    "prefill_dequant_affine_qmm_f16_block", constants: constants)
            })
        #expect(builds == 0)
        #expect(!candidate.isAvailable)
        #expect(candidate.tensorOpsDecision.path == .fallback)
        #expect(candidate.tensorOpsDecision.reason.contains("does not support Apple10"))
        for matmul in Self.matmuls {
            #expect(!candidate.select(Self.shape(matmul, tokens: 64)).selected)
        }
    }

    /// Forced `on` attempts the build and records the failure; every matmul
    /// then reports that reason instead of trapping.
    @Test func forcedOnFallsBackWhenPipelineCreationFails() throws {
        struct InjectedFailure: Error {}
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(
            context: context,
            weightBits: 4,
            capabilities: Self.capabilities(apple10: false),
            tensorOpsMode: .on,
            tensorOpsPipelineBuilder: { _, _ in throw InjectedFailure() })
        #expect(!candidate.isAvailable)
        #expect(candidate.tensorOpsDecision.path == .fallback)
        #expect(candidate.tensorOpsDecision.reason.contains("pipeline creation failed"))
        for matmul in Self.matmuls {
            let selection = candidate.select(Self.shape(matmul, tokens: 64))
            #expect(!selection.selected)
            #expect(selection.reason.contains("pipeline creation failed"))
        }
    }

    /// Every kernel-level constraint has a scalar fallback: the row threshold,
    /// a partial K tile, padded strides, a per-role width mismatch and the
    /// bf16 offset alignment of the scales, biases, input and output.
    @Test func shapeAndAlignmentConstraintsKeepTheScalarPath() throws {
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(
            context: context,
            weightBits: 4,
            capabilities: Self.capabilities(apple10: true),
            tensorOpsMode: .auto,
            tensorOpsPipelineBuilder: Self.stubBuilder(context))
        let base = Self.shape(Self.matmuls[0], tokens: 64)
        #expect(candidate.select(base).selected)

        var short = base
        short.tokenCount = 31
        #expect(!candidate.select(short).selected)
        #expect(candidate.select(short).reason.contains("below the tensor-ops"))

        var partialK = base
        partialK.columns = 65
        partialK.xStrideElements = 65
        #expect(!candidate.select(partialK).selected)
        #expect(candidate.select(partialK).reason.contains("multiple of 64"))

        var paddedX = base
        paddedX.xStrideElements = 2817
        #expect(!candidate.select(paddedX).selected)
        #expect(candidate.select(paddedX).reason.contains("x stride"))

        var paddedY = base
        paddedY.yStrideElements = 4097
        #expect(!candidate.select(paddedY).selected)
        #expect(candidate.select(paddedY).reason.contains("y stride"))

        var wrongBits = base
        wrongBits.weightBits = 8
        #expect(!candidate.select(wrongBits).selected)
        #expect(candidate.select(wrongBits).reason.contains("4-bit"))

        var unalignedScales = base
        unalignedScales.scalesOffset = 1
        #expect(!candidate.select(unalignedScales).selected)
        #expect(candidate.select(unalignedScales).reason.contains("unaligned"))

        var unalignedY = base
        unalignedY.yOffset = 1
        #expect(!candidate.select(unalignedY).selected)

        // The same shape is eligible for an 8-bit pipeline; the width is the
        // only difference, and it is the pipeline's function constant.
        let eightBit = MPPPrefillInt4QMM(
            context: context,
            weightBits: 8,
            capabilities: Self.capabilities(apple10: true),
            tensorOpsMode: .auto,
            tensorOpsPipelineBuilder: Self.stubBuilder(context))
        var eightBitShape = base
        eightBitShape.weightBits = 8
        #expect(eightBit.select(eightBitShape).selected)
    }

    /// A candidate that never built still reports `.unavailable` from `encode`,
    /// which is the value every call site already treats as "scalar path".
    @Test func offCandidateLeavesASupportedShapeOnTheScalarPath() throws {
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(
            context: context,
            weightBits: 4,
            capabilities: Self.capabilities(apple10: true),
            tensorOpsMode: .off)
        let buffer = try #require(
            context.device.makeBuffer(length: 4096, options: .storageModeShared))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        let path = try candidate.encode(
            commandBuffer: commandBuffer,
            weights: buffer,
            scales: buffer,
            biases: buffer,
            x: buffer,
            y: buffer,
            m: 64,
            n: 64,
            k: 64)
        #expect(path == .unavailable)
    }

    /// K5 logs one decision per pipeline family, with the subject naming the
    /// family so a fallback line is attributable.
    @Test func decisionLogNamesTheMatmulFamily() {
        let decision = TensorOpsDecision(
            mode: .auto,
            path: .fallback,
            reason: "device does not support Apple10 MPP tensor operations")
        #expect(
            decision.logLine(subject: "prefill projections")
                .contains("tensor-ops prefill projections: fallback"))
        #expect(decision.logLine(subject: "prefill ffn").contains("mode=auto"))
    }

    // MARK: - FFN routing

    /// The FFN block routes all three matmuls or none, and it says why.
    @Test func ffnBlockSelectsTensorOpsOnlyWhenAllThreeMatmulsFit() throws {
        let context = try MetalContext()
        let off = MPPPrefillInt4QMM(
            context: context,
            weightBits: 4,
            capabilities: Self.capabilities(apple10: true),
            tensorOpsMode: .off)
        let scalarBlock = try PrefillSharedExpert(
            context: context, weightBits: 4, tensorOps: off)
        #expect(scalarBlock.tensorOpsDecision?.path == .fallback)
        let scalarSelection = scalarBlock.ffnTensorOpsSelection(
            queryCount: 64, d: 2048, intermediate: 6144)
        #expect(scalarSelection?.selected == false)

        let available = MPPPrefillInt4QMM(
            context: context,
            weightBits: 4,
            capabilities: Self.capabilities(apple10: true),
            tensorOpsMode: .auto,
            tensorOpsPipelineBuilder: Self.stubBuilder(context))
        let tensorBlock = try PrefillSharedExpert(
            context: context, weightBits: 4, tensorOps: available)
        #expect(tensorBlock.tensorOpsDecision?.path == .tensorOps)
        let selected = try #require(
            tensorBlock.ffnTensorOpsSelection(
                queryCount: 64, d: 2048, intermediate: 6144))
        #expect(selected.selected, "\(selected.reason)")
        #expect(
            tensorBlock.ffnTensorOpsSelection(
                queryCount: 31, d: 2048, intermediate: 6144)?.selected == false)
        // gate/up take K=D and down takes K=F, so both widths must be whole
        // tiles; a ragged intermediate keeps the scalar QMM.
        #expect(
            tensorBlock.ffnTensorOpsSelection(
                queryCount: 64, d: 2048, intermediate: 6150)?.selected == false)

        let noCandidate = try PrefillSharedExpert(
            context: context, weightBits: 4, tensorOps: nil)
        #expect(noCandidate.tensorOpsDecision == nil)
        #expect(
            noCandidate.ffnTensorOpsSelection(
                queryCount: 64, d: 2048, intermediate: 6144) == nil)
    }

    /// Requirement 3: `off` reproduces today's path exactly. Both blocks take
    /// the scalar QMM and the FFN output is byte-identical, on every host.
    @Test func offFfnBlockIsByteIdenticalToTheScalarBlock() throws {
        let context = try MetalContext()
        let tokens = 8
        let d = 128
        let f = 64
        let gate = try Self.int8Projection(
            context: context, rows: f, columns: d, salt: 0x11)
        let up = try Self.int8Projection(
            context: context, rows: f, columns: d, salt: 0x5D)
        let down = try Self.int8Projection(
            context: context, rows: d, columns: f, salt: 0xA3)
        let x = Fp16Buffer.make(
            context.device, halves: Self.activation(tokens: tokens, d: d))
        let plainY = Fp16Buffer.make(context.device, count: tokens * d)
        let offY = Fp16Buffer.make(context.device, count: tokens * d)
        let scratchGate = Fp16Buffer.make(context.device, count: tokens * f)
        let scratchUp = Fp16Buffer.make(context.device, count: tokens * f)
        let scratchAct = Fp16Buffer.make(context.device, count: tokens * f)
        let plainX = try #require(x)
        let plain = try PrefillSharedExpert(context: context, weightBits: 8)
        let off = try PrefillSharedExpert(
            context: context,
            weightBits: 8,
            tensorOps: MPPPrefillInt4QMM(
                context: context,
                weightBits: 8,
                capabilities: Self.capabilities(apple10: true),
                tensorOpsMode: .off))

        let plainCB = try #require(context.queue.makeCommandBuffer())
        try plain.encodeBlock(
            commandBuffer: plainCB,
            x: plainX,
            y: try #require(plainY),
            gate: gate,
            up: up,
            down: down,
            scratchGate: try #require(scratchGate),
            scratchUp: try #require(scratchUp),
            scratchAct: try #require(scratchAct),
            queryCount: tokens,
            d: d,
            intermediate: f,
            xStrideElements: d,
            yStrideElements: d)
        plainCB.commit()
        plainCB.waitUntilCompleted()
        #expect(plainCB.error == nil)

        let offCB = try #require(context.queue.makeCommandBuffer())
        try off.encodeBlock(
            commandBuffer: offCB,
            x: plainX,
            y: try #require(offY),
            gate: gate,
            up: up,
            down: down,
            scratchGate: try #require(scratchGate),
            scratchUp: try #require(scratchUp),
            scratchAct: try #require(scratchAct),
            queryCount: tokens,
            d: d,
            intermediate: f,
            xStrideElements: d,
            yStrideElements: d)
        offCB.commit()
        offCB.waitUntilCompleted()
        #expect(offCB.error == nil)

        let baseline = Fp16Buffer.readHalf(try #require(plainY), count: tokens * d)
        let fallback = Fp16Buffer.readHalf(try #require(offY), count: tokens * d)
        #expect(fallback == baseline)
    }

    // MARK: - Arithmetic (Apple10 only)

    /// The MPP kernel needs Apple10 tensor operations; an M3 cannot run it, so
    /// this comparison is a recorded skip there rather than a faked pass. On
    /// Apple10 it pins the tensor path against the existing scalar prefill QMM
    /// on a toy shape with a partial last row tile (40 of 64 rows).
    ///
    /// Tolerance: the two kernels dequantise and reduce in different orders,
    /// so this is not byte equality. The bound is the one the existing MPP
    /// suite measured: absolute error <= 0.03 and relative error <= 1e-3, with
    /// inputs quantised exactly as production weights are.
    @Test(
        .enabled(
            if: mppTensorOpsAvailable,
            "Requires Apple10 MPP tensor operations; this host cannot dispatch the tensor kernel"))
    func tensorOpsMatmulMatchesTheScalarKernelOnAToyShape() throws {
        let context = try MetalContext()
        let m = 40
        let n = 64
        let k = 128
        let tensor = MPPPrefillInt4QMM(
            context: context,
            weightBits: 4,
            capabilities: Self.capabilities(apple10: true),
            tensorOpsMode: .on)
        let scalar = try PrefillInt4QMM(context: context, weightBits: 4)

        let weights = try Self.int4Weights(
            context: context, rows: n, columns: k, salt: 0x29)
        let input = try #require(
            Fp16Buffer.make(
                context.device,
                halves: Self.activation(tokens: m, d: k)))
        let expected = try #require(Fp16Buffer.make(context.device, count: m * n))
        let actual = try #require(Fp16Buffer.make(context.device, count: m * n))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        try scalar.encode(
            commandBuffer: commandBuffer,
            weights: weights.weights,
            weightsOffset: weights.weightsOffset,
            scales: weights.scales,
            scalesOffset: weights.scalesOffset,
            biases: weights.biases,
            biasesOffset: weights.biasesOffset,
            x: input,
            y: expected,
            t: m,
            n: n,
            k: k)
        let path = try tensor.encode(
            commandBuffer: commandBuffer,
            weights: weights.weights,
            weightsOffset: weights.weightsOffset,
            scales: weights.scales,
            scalesOffset: weights.scalesOffset,
            biases: weights.biases,
            biasesOffset: weights.biasesOffset,
            x: input,
            y: actual,
            m: m,
            n: n,
            k: k)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(path == .affineThreadgroupF16)

        let reference = Fp16Buffer.read(expected, count: m * n)
        let candidate = Fp16Buffer.read(actual, count: m * n)
        let maxAbs = RelError.maxAbsDiff(candidate, reference)
        let relative = RelError.compute(actual: candidate, reference: reference)
        #expect(
            maxAbs <= 0.03,
            "tensor maxAbs=\(maxAbs) rel=\(relative)")
        #expect(
            relative <= 1e-3 || maxAbs <= 0.01,
            "tensor rel=\(relative) maxAbs=\(maxAbs)")
    }

    // MARK: - Fixtures

    private static func activation(tokens: Int, d: Int) -> [Float16] {
        (0..<(tokens * d)).map { index in
            Float16(Float((index * 11) % 29 - 14) / 64.0)
        }
    }

    private static func int4Weights(
        context: MetalContext, rows: Int, columns: Int, salt: Int
    ) throws -> SharedExpertInt8Proj {
        let groups = columns / Quantization.groupSize
        let packed = (0..<(rows * columns / 2)).map { index in
            UInt8(truncatingIfNeeded: index &* 37 &+ salt)
        }
        let scales = (0..<(rows * groups)).map { index in
            Quantization.bf16Bits(0.001 + Float(index % 5) * 0.0005)
        }
        let biases = (0..<(rows * groups)).map { index in
            Quantization.bf16Bits(-0.01 + Float(index % 3) * 0.002)
        }
        return try projection(
            context: context, packed: packed, scales: scales, biases: biases,
            rows: rows, columns: columns)
    }

    private static func int8Projection(
        context: MetalContext, rows: Int, columns: Int, salt: Int
    ) throws -> SharedExpertInt8Proj {
        let groups = columns / Quantization.groupSize
        let packed = (0..<(rows * columns)).map { index in
            UInt8(truncatingIfNeeded: index &* 37 &+ salt)
        }
        let scales = (0..<(rows * groups)).map { index in
            Quantization.bf16Bits(0.001 + Float(index % 5) * 0.0005)
        }
        let biases = (0..<(rows * groups)).map { index in
            Quantization.bf16Bits(-0.01 + Float(index % 3) * 0.002)
        }
        return try projection(
            context: context, packed: packed, scales: scales, biases: biases,
            rows: rows, columns: columns)
    }

    private static func projection(
        context: MetalContext,
        packed: [UInt8],
        scales: [UInt16],
        biases: [UInt16],
        rows: Int,
        columns: Int
    ) throws -> SharedExpertInt8Proj {
        let weights = try #require(
            context.device.makeBuffer(
                bytes: packed, length: packed.count, options: .storageModeShared))
        let scalesBuffer = try #require(
            context.device.makeBuffer(
                bytes: scales,
                length: scales.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
        let biasesBuffer = try #require(
            context.device.makeBuffer(
                bytes: biases,
                length: biases.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared))
        return SharedExpertInt8Proj(
            weights: weights,
            scales: scalesBuffer,
            biases: biasesBuffer,
            rows: UInt32(rows),
            cols: UInt32(columns))
    }
}
