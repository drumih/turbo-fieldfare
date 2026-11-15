import Foundation
import Metal
import TinyTitanCore

enum MPPPrefillInt4QMMError: Error, CustomStringConvertible {
    case pipelineUnavailable(reason: String)
    case invalidArguments(String)

    public var description: String {
        switch self {
        case .pipelineUnavailable(let reason):
            return "MPP prefill QMM path unavailable: \(reason)"
        case .invalidArguments(let detail):
            return "MPP prefill QMM invalid arguments: \(detail)"
        }
    }
}

/// The MPP (Metal Performance Primitives) tensor-ops variant of the prefill
/// affine matmul: dequantise a 4/8-bit affine weight tile to fp16 in
/// threadgroup memory, then run the `matmul2d` tensor operation.
///
/// K5: the pipeline is built through the same `TensorOpsPolicy` seam as
/// `PrefillAttention`. `auto` builds it only where the Apple10 capability is
/// present, `off` never builds it, and `on` attempts it and falls back on a
/// pipeline failure. Every shape the kernel cannot take keeps the existing
/// scalar path; `select(_:)` is the pure decision the call sites consult.
final class MPPPrefillInt4QMM {
    enum Path: String, Sendable {
        case affineThreadgroupF16 = "affine-threadgroup-f16"
        case unavailable
    }

    /// One matmul as the kernel sees it. Strides are in elements, offsets in
    /// bytes. The kernel reads X with a K stride and writes Y with an N stride,
    /// so both must be contiguous; the call sites pass their real strides.
    struct Shape: Equatable, Sendable {
        var tokenCount: Int
        var rows: Int
        var columns: Int
        var weightBits: Int
        var xStrideElements: Int
        var yStrideElements: Int
        var xOffset: Int = 0
        var yOffset: Int = 0
        var weightsOffset: Int = 0
        var scalesOffset: Int = 0
        var biasesOffset: Int = 0
    }

    /// Whether the tensor-ops variant serves a shape, and why not when it does
    /// not. Pure: no pipeline is touched, so a test can enumerate every matmul.
    struct Selection: Equatable, Sendable {
        let selected: Bool
        let reason: String
    }

    static let tileM = 64
    static let tileN = 32
    static let tileK = Quantization.groupSize
    /// Below this row count the scalar kernel stays in place: the tensor
    /// operation pays for a 64-row tile and a threadgroup dequant either way,
    /// and K5 is a prefill optimisation, not a decode one.
    static let minimumTokenCount = 32
    static let supportedWeightBits = [4, 8]

    private var pipeline: MTLComputePipelineState?
    /// K6: the compile failure reason, recorded once at init so an explicit
    /// MPP request can throw the real cause instead of silently degrading.
    private let unavailableReason: String
    /// The width this instance's pipeline was compiled for. A per-role
    /// projection can carry a different width, which `select` refuses.
    let weightBits: Int
    private let tensorOpsDecisionValue: TensorOpsDecision

    /// What this candidate decided about tensor operations. Test-visible and
    /// the source of the startup log line.
    var tensorOpsDecision: TensorOpsDecision { tensorOpsDecisionValue }

    /// - Parameters:
    ///   - capabilities: what this machine can do; omitted, the live probe.
    ///   - tensorOpsMode: `TINYTITAN_TENSOROPS`, already parsed; omitted, read
    ///     from the environment here.
    ///   - tensorOpsPipelineBuilder: how the pipeline is compiled. Omitted, the
    ///     `tensorops` Metal module; a test passes a stub or a throwing builder
    ///     to drive every policy branch without an Apple10 host.
    init(
        context: MetalContext,
        weightBits: Int = 4,
        capabilities: HardwareCapabilities? = nil,
        tensorOpsMode: TensorOpsMode? = nil,
        tensorOpsPipelineBuilder:
            ((String, [MetalFunctionConstant]) throws -> MTLComputePipelineState)? = nil
    ) {
        precondition([4, 8].contains(weightBits))
        self.weightBits = weightBits
        let liveCapabilities = capabilities ?? HardwareCapabilities.probe(device: context.device)
        let resolution = TensorOpsMode.resolve()
        let mode = tensorOpsMode ?? resolution.mode
        let note = tensorOpsMode == nil ? resolution.note : nil
        let plan = TensorOpsPolicy.plan(
            capabilities: liveCapabilities, mode: mode, note: note)
        guard plan.attemptsPipeline else {
            self.pipeline = nil
            self.unavailableReason = plan.reason
            self.tensorOpsDecisionValue = TensorOpsDecision(
                mode: mode, path: .fallback, reason: plan.reason)
            return
        }
        let constants = [
            MetalFunctionConstant(index: 78, value: .uint32(UInt32(weightBits)))
        ]
        let build =
            tensorOpsPipelineBuilder
            ?? { name, constants in
                try Self.makePipeline(
                    context: context, name: name, constants: constants)
            }
        do {
            self.pipeline = try build("mpp_prefill_affine_threadgroup_f16", constants)
            self.unavailableReason = ""
            self.tensorOpsDecisionValue = TensorOpsDecision(
                mode: mode, path: .tensorOps, reason: plan.reason)
        } catch {
            // Capability probe: this path is optional on non-Apple10 hardware,
            // so init stays non-throwing. Record the reason so a later
            // explicit request can throw it (K6).
            self.pipeline = nil
            self.unavailableReason = "pipeline creation failed: \(error)"
            self.tensorOpsDecisionValue = TensorOpsDecision(
                mode: mode,
                path: .fallback,
                reason: "pipeline creation failed: \(error)")
        }
    }

    var isAvailable: Bool {
        pipeline != nil
    }

    /// The pure selection rule for one matmul. Shape constraints are checked
    /// after the pipeline, because a fallback candidate is scalar for every
    /// shape and says so once.
    func select(_ shape: Shape) -> Selection {
        guard isAvailable else {
            return Selection(
                selected: false,
                reason: unavailableReason.isEmpty
                    ? "MPP pipeline failed to compile"
                    : unavailableReason)
        }
        guard shape.tokenCount >= Self.minimumTokenCount else {
            return Selection(
                selected: false,
                reason: "\(shape.tokenCount) rows is below the tensor-ops "
                    + "prefill threshold of \(Self.minimumTokenCount)")
        }
        guard shape.weightBits == weightBits else {
            return Selection(
                selected: false,
                reason: "shape is \(shape.weightBits)-bit but the pipeline is "
                    + "\(weightBits)-bit")
        }
        if let rejection = Self.shapeRejection(shape) {
            return Selection(selected: false, reason: rejection)
        }
        return Selection(
            selected: true,
            reason: "MPP \(shape.rows)x\(shape.columns) matmul, "
                + "\(shape.tokenCount) rows, \(weightBits)-bit")
    }

    /// The kernel-level rules, with no policy or row-count threshold: `encode`
    /// applies these too, so a caller that skips `select` still cannot encode
    /// a shape the tensor operation would read out of bounds.
    static func shapeRejection(_ shape: Shape) -> String? {
        guard supportedWeightBits.contains(shape.weightBits) else {
            return "unsupported weight bits \(shape.weightBits)"
        }
        guard shape.tokenCount > 0, shape.rows > 0, shape.columns > 0 else {
            return "empty shape m=\(shape.tokenCount) n=\(shape.rows) "
                + "k=\(shape.columns)"
        }
        // The A operand declares a *static* `tileK` extent and the MPP
        // operation trusts the operand's extents, so a partial K tile would
        // read past the staged weights -- the bug llama.cpp fixed in a3027337
        // ("the op reads those out-of-bounds elements (undefined behavior per
        // the MSL specification, 2.22.2)"). A partial last M tile is the
        // supported case (the operation edge-checks against the operand).
        guard shape.columns.isMultiple(of: tileK) else {
            return "K=\(shape.columns) is not a multiple of \(tileK)"
        }
        guard shape.xStrideElements == shape.columns else {
            return "x stride \(shape.xStrideElements) != K \(shape.columns)"
        }
        guard shape.yStrideElements == shape.rows else {
            return "y stride \(shape.yStrideElements) != N \(shape.rows)"
        }
        guard shape.weightsOffset >= 0, shape.scalesOffset >= 0,
            shape.biasesOffset >= 0, shape.xOffset >= 0, shape.yOffset >= 0
        else {
            return "negative offset"
        }
        guard
            shape.scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
            shape.biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
            shape.xOffset.isMultiple(of: MemoryLayout<Float16>.stride),
            shape.yOffset.isMultiple(of: MemoryLayout<Float16>.stride)
        else {
            return "unaligned offset weights=\(shape.weightsOffset) "
                + "scales=\(shape.scalesOffset) biases=\(shape.biasesOffset) "
                + "x=\(shape.xOffset) y=\(shape.yOffset)"
        }
        return nil
    }

    /// `required: true` makes an unavailable path a thrown error instead of a
    /// silent `.unavailable` fallback — use it when the caller explicitly
    /// requests the MPP path. Auto-selected callers keep `required: false`
    /// and check the returned `Path`.
    ///
    /// `xStrideElements`/`yStrideElements` default to the contiguous strides
    /// the kernel assumes; production callers pass the real ones so a padded
    /// layout stays on the scalar path.
    @discardableResult
    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, yOffset: Int = 0,
        m: Int,
        n: Int,
        k: Int,
        xStrideElements: Int? = nil,
        yStrideElements: Int? = nil,
        required: Bool = false
    ) throws -> Path {
        let shape = Shape(
            tokenCount: m,
            rows: n,
            columns: k,
            weightBits: weightBits,
            xStrideElements: xStrideElements ?? k,
            yStrideElements: yStrideElements ?? n,
            xOffset: xOffset,
            yOffset: yOffset,
            weightsOffset: weightsOffset,
            scalesOffset: scalesOffset,
            biasesOffset: biasesOffset)
        if let rejection = Self.shapeRejection(shape) {
            if required {
                throw MPPPrefillInt4QMMError.invalidArguments(rejection)
            }
            return .unavailable
        }
        guard let pipeline else {
            if required {
                throw MPPPrefillInt4QMMError.pipelineUnavailable(
                    reason: unavailableReason.isEmpty
                        ? "MPP pipeline failed to compile"
                        : unavailableReason)
            }
            return .unavailable
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            if required { throw MetalError.commandEncoderFailed }
            return .unavailable
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (n + Self.tileN - 1) / Self.tileN,
                height: (m + Self.tileM - 1) / Self.tileM,
                depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: pipeline.threadExecutionWidth * 4,
                height: 1,
                depth: 1))
        encoder.endEncoding()
        return .affineThreadgroupF16
    }

    private static func compileTensorOpsLibrary(device: MTLDevice) throws -> MTLLibrary {
        try MetalContext.moduleLibrary(device: device, module: "tensorops")
    }

    private static func makePipeline(
        context: MetalContext,
        name: String,
        constants: [MetalFunctionConstant]
    ) throws -> MTLComputePipelineState {
        let library = try compileTensorOpsLibrary(device: context.device)
        let values = MTLFunctionConstantValues()
        for constant in constants {
            switch constant.value {
            case .uint32(let raw):
                var value = raw
                values.setConstantValue(&value, type: .uint, index: constant.index)
            case .bool(let raw):
                var value = raw
                values.setConstantValue(&value, type: .bool, index: constant.index)
            case .float(let raw):
                var value = raw
                values.setConstantValue(&value, type: .float, index: constant.index)
            }
        }
        let function = try library.makeFunction(name: name, constantValues: values)
        return try context.device.makeComputePipelineState(function: function)
    }
}
