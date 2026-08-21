import Metal

public final class VisionLinearBF16 {
    private enum PathMode {
        case disabled, all, attention, mlp

        func applies(n: Int, k: Int) -> Bool {
            switch self {
            case .disabled: false
            case .all: true
            case .attention: n == 1_152 && k <= 1_152
            case .mlp: n == 4_304 || k == 4_304
            }
        }
    }

    public struct Metadata: Sendable, Equatable {
        public let tileM: Int
        public let tileN: Int
        public let tileK: Int
        public let modelDerivedCPUHeapScratchBytes = 0
        public let expandedWeightBufferBytes = 0

        fileprivate init(tileM: Int, tileN: Int, tileK: Int) {
            self.tileM = tileM
            self.tileN = tileN
            self.tileK = tileK
        }
    }

    /// Head-major padded store for the QKV projections. `rowStride` is the
    /// per-head row count of the destination; lanes in `[headDim, paddedHeadDim)`
    /// are left untouched and must already be zero.
    public struct PaddedHeadStore: Sendable, Equatable {
        public let rowStride: Int
        public let headDim: Int
        public let paddedHeadDim: Int

        public init(rowStride: Int, headDim: Int, paddedHeadDim: Int) {
            self.rowStride = rowStride
            self.headDim = headDim
            self.paddedHeadDim = paddedHeadDim
        }
    }

    private let pipeline: MTLComputePipelineState?
    private let mlxPipelines: [Int: MTLComputePipelineState]
    private let registerPipeline: MTLComputePipelineState?
    private let registerGeGLUPipeline: MTLComputePipelineState?
    private let registerMode: PathMode

    /// The fused MLP epilogue exists only on the register path.
    public func supportsFusedGeGLU(n: Int, k: Int) -> Bool {
        registerGeGLUPipeline != nil && registerMode.applies(n: n, k: k)
    }

    /// Only the register GEMM implements the padded head-major store, so the
    /// fused attention layout stays off whenever another linear path is selected.
    public func supportsPaddedHeadStore(n: Int, k: Int) -> Bool {
        registerPipeline != nil && registerMode.applies(n: n, k: k)
    }

    public init(context: MetalContext,
                environment: [String: String] = ProcessInfo.processInfo.environment) throws {
        let registerAttention = environment[
            "TURBO_FIELDFARE_VISION_REGISTER_ATTENTION"] == "1"
        let registerMLP = environment["TURBO_FIELDFARE_VISION_REGISTER_MLP"] == "1"
        let mode: PathMode
        if environment["TURBO_FIELDFARE_VISION_REGISTER_GEMM"] == "1"
            || (registerAttention && registerMLP) {
            mode = .all
        } else if registerAttention {
            mode = .attention
        } else if registerMLP {
            mode = .mlp
        } else {
            mode = .disabled
        }
        registerMode = mode
        if mode != .disabled {
            let library = try MetalContext.privateLibrary(
                device: context.device, module: "vision_register_gemm")
            guard let function = library.makeFunction(name: "tff_vision_register_gemm"),
                  let geglu = library.makeFunction(
                    name: "tff_vision_register_gemm_geglu") else {
                throw MetalError.missingFunction("tff_vision_register_gemm")
            }
            registerPipeline = try context.device.makeComputePipelineState(function: function)
            registerGeGLUPipeline = try context.device.makeComputePipelineState(
                function: geglu)
        } else {
            registerPipeline = nil
            registerGeGLUPipeline = nil
        }
        var loadedMLXPipelines: [Int: MTLComputePipelineState] = [:]
        if let path = environment["TURBO_FIELDFARE_VISION_MLX_GEMM_METALLIB"] {
            let mlxLibrary = try context.device.makeLibrary(URL: URL(fileURLWithPath: path))
            for alignN in [false, true] {
                for alignK in [false, true] {
                    let constants = MTLFunctionConstantValues()
                    var disabled = false
                    var enabled = true
                    constants.setConstantValue(&disabled, type: .bool, index: 10)
                    constants.setConstantValue(&disabled, type: .bool, index: 100)
                    constants.setConstantValue(&disabled, type: .bool, index: 110)
                    constants.setConstantValue(&enabled, type: .bool, index: 200)
                    var n = alignN, k = alignK
                    constants.setConstantValue(&n, type: .bool, index: 201)
                    constants.setConstantValue(&k, type: .bool, index: 202)
                    let function = try mlxLibrary.makeFunction(
                        name: "steel_gemm_fused_nax_nt_bfloat16_bfloat16_"
                            + "bm64_bn128_bk256_wm2_wn4",
                        constantValues: constants)
                    loadedMLXPipelines[
                        (alignN ? 2 : 0) | (alignK ? 1 : 0)
                    ] = try context.device.makeComputePipelineState(function: function)
                }
            }
        }
        mlxPipelines = loadedMLXPipelines
        if mode == .all || !loadedMLXPipelines.isEmpty {
            pipeline = nil
        } else {
            let library = try MetalContext.privateLibrary(
                device: context.device, module: "tensorops",
                includeVisionTensorOps: true)
            guard let function = library.makeFunction(name: "mpp_vision_linear_bf16") else {
                throw MetalError.missingFunction("mpp_vision_linear_bf16")
            }
            pipeline = try context.device.makeComputePipelineState(function: function)
        }
    }

    @discardableResult
    public func encode(commandBuffer: MTLCommandBuffer,
                       input: MTLBuffer, inputOffset: Int = 0,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       output: MTLBuffer, outputOffset: Int = 0,
                       m: Int, n: Int, k: Int,
                       paddedHeadStore: PaddedHeadStore? = nil,
                       fusesGeGLU: Bool = false) -> Metadata {
        precondition(m > 0 && n > 0 && k > 0)
        precondition(!(fusesGeGLU && paddedHeadStore != nil))
        if let selected = fusesGeGLU ? registerGeGLUPipeline : registerPipeline,
           registerMode.applies(n: n, k: k) {
            encodeRegisterGEMM(
                commandBuffer: commandBuffer,
                input: input, inputOffset: inputOffset,
                weights: weights, weightsOffset: weightsOffset,
                output: output, outputOffset: outputOffset,
                m: m, n: n, k: k,
                paddedHeadStore: paddedHeadStore,
                pipeline: selected)
            return Metadata(tileM: 64, tileN: 128, tileK: 16)
        }
        precondition(!fusesGeGLU, "fused GeGLU requires the register GEMM path")
        precondition(paddedHeadStore == nil,
                     "padded head store requires the register GEMM path")
        if !mlxPipelines.isEmpty {
            encodeExternalMLXGEMM(
                commandBuffer: commandBuffer,
                input: input, inputOffset: inputOffset,
                weights: weights, weightsOffset: weightsOffset,
                output: output, outputOffset: outputOffset,
                m: m, n: n, k: k)
            return Metadata(tileM: 64, tileN: 128, tileK: 256)
        }
        guard let pipeline else {
            preconditionFailure("no vision linear path is available for this shape")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return Metadata(tileM: 64, tileN: 32, tileK: 64)
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        var rows = UInt32(m), columns = UInt32(n), inner = UInt32(k)
        encoder.setBytes(&rows, length: 4, index: 3)
        encoder.setBytes(&columns, length: 4, index: 4)
        encoder.setBytes(&inner, length: 4, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + 31) / 32, height: (m + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: pipeline.threadExecutionWidth * 4, height: 1, depth: 1))
        encoder.endEncoding()
        return Metadata(tileM: 64, tileN: 32, tileK: 64)
    }

    private func encodeRegisterGEMM(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer, inputOffset: Int,
        weights: MTLBuffer, weightsOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        m: Int, n: Int, k: Int,
        paddedHeadStore: PaddedHeadStore?,
        pipeline: MTLComputePipelineState
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        var rows = UInt32(m), columns = UInt32(n), inner = UInt32(k)
        encoder.setBytes(&rows, length: 4, index: 3)
        encoder.setBytes(&columns, length: 4, index: 4)
        encoder.setBytes(&inner, length: 4, index: 5)
        var store = VisionGEMMPaddedStore(
            rowStride: UInt32(paddedHeadStore?.rowStride ?? 0),
            headDim: UInt32(paddedHeadStore?.headDim ?? 0),
            paddedHeadDim: UInt32(paddedHeadStore?.paddedHeadDim ?? 0))
        encoder.setBytes(&store,
                         length: MemoryLayout<VisionGEMMPaddedStore>.stride,
                         index: 6)
        encoder.setComputePipelineState(pipeline)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + 127) / 128, height: (m + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeExternalMLXGEMM(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer, inputOffset: Int,
        weights: MTLBuffer, weightsOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        m: Int, n: Int, k: Int
    ) {
        let key = (n.isMultiple(of: 128) ? 2 : 0) | (k.isMultiple(of: 256) ? 1 : 0)
        guard let pipeline = mlxPipelines[key],
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let tilesN = (n + 127) / 128
        let tilesM = (m + 63) / 64
        var parameters = MLXGEMMParameters(
            m: m, n: n, k: k, tilesN: tilesN, tilesM: tilesM)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(weights, offset: weightsOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        encoder.setBytes(&parameters, length: MemoryLayout<MLXGEMMParameters>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: tilesN * 4, height: (tilesM + 3) / 4, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 2))
        encoder.endEncoding()
    }
}

private struct VisionGEMMPaddedStore {
    var rowStride: UInt32
    var headDim: UInt32
    var paddedHeadDim: UInt32
}

private struct MLXGEMMParameters {
    var m: Int32
    var n: Int32
    var k: Int32
    var lda: Int32
    var ldb: Int32
    var ldd: Int32
    var tilesN: Int32
    var tilesM: Int32
    var batchStrideA: Int64 = 0
    var batchStrideB: Int64 = 0
    var batchStrideD: Int64
    var swizzleLog: Int32 = 2
    var alignedKIterations: Int32
    var batchDimensions: Int32 = 1

    init(m: Int, n: Int, k: Int, tilesN: Int, tilesM: Int) {
        self.m = Int32(m)
        self.n = Int32(n)
        self.k = Int32(k)
        lda = Int32(k)
        ldb = Int32(k)
        ldd = Int32(n)
        self.tilesN = Int32(tilesN)
        self.tilesM = Int32(tilesM)
        batchStrideD = Int64(m * n)
        alignedKIterations = Int32(k / 256)
    }
}
