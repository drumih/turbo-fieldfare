import Foundation
import Metal
import TurboFieldfareFormat

/// Batched trunk projection for Stage-E1 chunked prefill: `Y = X W^T` over a
/// `[tokens x K]` activation against a resident `[N, K]` weight view, i.e. the
/// decode trunk GEMV (`K3TrunkGEMV`) lifted to a matmul.
///
/// Affine int4/int8 views run on the M5 Neural Accelerator via the MSL 4.0
/// `mpp::tensor_ops::matmul2d` kernels in `tensorops_k3.metal` (compiled
/// separately; `__HAVE_TENSOR__` guarded). When the tensor-ops pipelines are
/// unavailable — pre-M5 silicon, or the `K3_PREFILL_FORCE_FALLBACK` env toggle
/// the tests use to exercise the portable path on an M5 — they fall back to
/// the batched simd GEMMs in `prefill_k3.metal`. BF16/FP32 views (FP32 is the
/// router's format) always use the portable kernels, mirroring how the house
/// only routes affine int4 through tensor ops.
///
/// Output dtype matches the decode storage contract: fp16 for int4/int8/bf16,
/// fp32 for fp32. `K` (the view's column count) must be a multiple of the
/// affine group size, as with the decode GEMV.
final class K3PrefillGEMM {
    /// Which code path served a matmul (diagnostics / test assertions).
    enum Path: String, Sendable {
        case tensorOpsInt4 = "tensor-ops-int4"
        case tensorOpsInt8 = "tensor-ops-int8"
        case fallbackInt4 = "fallback-int4"
        case fallbackInt8 = "fallback-int8"
        case fallbackBF16 = "fallback-bf16"
        case fallbackF32 = "fallback-f32"
    }

    static let tileM = 64
    static let tileN = 32
    static let tileK = 64

    /// Environment toggle that forces the portable fallback even when the NAX
    /// tensor-ops kernels compiled (the fallback-path parity test sets this).
    static let forceFallbackEnv = "K3_PREFILL_FORCE_FALLBACK"

    private let tensorInt4: MTLComputePipelineState?
    private let tensorInt8: MTLComputePipelineState?
    private let gemmInt4: MTLComputePipelineState
    private let gemmInt8: MTLComputePipelineState
    private let gemmBF16: MTLComputePipelineState
    private let gemmF32: MTLComputePipelineState

    /// True when the NAX tensor-ops pipelines are live and not force-disabled.
    let usingTensorOps: Bool

    init(context: MetalContext, forceFallback: Bool? = nil) throws {
        let forced = forceFallback
            ?? (ProcessInfo.processInfo.environment[Self.forceFallbackEnv] != nil)
        let library = K3MetalLibrary.shared
        gemmInt4 = try library.pipeline(device: context.device, name: "k3_gemm_int4")
        gemmInt8 = try library.pipeline(device: context.device, name: "k3_gemm_int8")
        gemmBF16 = try library.pipeline(device: context.device, name: "k3_gemm_bf16")
        gemmF32 = try library.pipeline(device: context.device, name: "k3_gemm_f32")

        var t4: MTLComputePipelineState?
        var t8: MTLComputePipelineState?
        if !forced {
            // Compile-and-catch, mirroring MPPPrefillInt4QMM: any failure (no
            // __HAVE_TENSOR__, missing function, PSO error) leaves the tensor
            // path nil and the fallback serves.
            if let tensorLib = try? K3MetalLibrary.separateModuleLibrary(
                device: context.device, module: "tensorops_k3") {
                if let fn = tensorLib.makeFunction(name: "k3_tensorop_affine_int4_qmm") {
                    t4 = try? context.device.makeComputePipelineState(function: fn)
                }
                if let fn = tensorLib.makeFunction(name: "k3_tensorop_affine_int8_qmm") {
                    t8 = try? context.device.makeComputePipelineState(function: fn)
                }
            }
        }
        tensorInt4 = t4
        tensorInt8 = t8
        usingTensorOps = (t4 != nil) && (t8 != nil)
    }

    /// Encode `Y[t][n] = sum_k W[n][k] * X[t][k]` for a resident view.
    /// `tokens` is the chunk's row count M. `x`/`y` dtypes follow the format:
    /// fp16 for int4/int8/bf16, fp32 for fp32. Returns the path taken.
    @discardableResult
    func encode(commandBuffer: MTLCommandBuffer,
                view: TensorView,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                tokens: Int) throws -> Path {
        let format = try K3TrunkGEMV.format(of: view)
        let n = Int(view.shape.0)   // output features (weight rows)
        let k = Int(view.shape.1)   // input features (weight columns)
        precondition(tokens > 0)
        precondition(k > 0 && k % Self.tileK == 0,
                     "K \(k) must be a positive multiple of \(Self.tileK)")
        switch format {
        case .int4G64:
            if let pso = tensorInt4 {
                encodeTensor(commandBuffer: commandBuffer, pipeline: pso,
                             view: view, x: x, xOffset: xOffset,
                             y: y, yOffset: yOffset, m: tokens, n: n, k: k)
                return .tensorOpsInt4
            }
            encodeFallback(commandBuffer: commandBuffer, pipeline: gemmInt4,
                           view: view, x: x, xOffset: xOffset,
                           y: y, yOffset: yOffset, m: tokens, n: n, k: k)
            return .fallbackInt4
        case .int8G64:
            if let pso = tensorInt8 {
                encodeTensor(commandBuffer: commandBuffer, pipeline: pso,
                             view: view, x: x, xOffset: xOffset,
                             y: y, yOffset: yOffset, m: tokens, n: n, k: k)
                return .tensorOpsInt8
            }
            encodeFallback(commandBuffer: commandBuffer, pipeline: gemmInt8,
                           view: view, x: x, xOffset: xOffset,
                           y: y, yOffset: yOffset, m: tokens, n: n, k: k)
            return .fallbackInt8
        case .bf16:
            encodePlain(commandBuffer: commandBuffer, pipeline: gemmBF16,
                        view: view, x: x, xOffset: xOffset,
                        y: y, yOffset: yOffset, m: tokens, n: n, k: k)
            return .fallbackBF16
        case .fp32:
            encodePlain(commandBuffer: commandBuffer, pipeline: gemmF32,
                        view: view, x: x, xOffset: xOffset,
                        y: y, yOffset: yOffset, m: tokens, n: n, k: k)
            return .fallbackF32
        }
    }

    // MARK: - Encoders

    /// NAX tensor-ops path: buffers 0-4 = weights/scales/biases/x/y, dims at
    /// 5-7 (M, N, K) — the house `mpp_prefill_affine_threadgroup_f16` contract.
    private func encodeTensor(commandBuffer: MTLCommandBuffer,
                              pipeline: MTLComputePipelineState,
                              view: TensorView,
                              x: MTLBuffer, xOffset: Int,
                              y: MTLBuffer, yOffset: Int,
                              m: Int, n: Int, k: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(view.buffer, offset: Int(view.offset), index: 0)
        encoder.setBuffer(view.buffer, offset: Int(view.scaleOffset), index: 1)
        encoder.setBuffer(view.buffer, offset: Int(view.biasOffset), index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + Self.tileN - 1) / Self.tileN,
                    height: (m + Self.tileM - 1) / Self.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Affine fallback (int4/int8): same buffer contract as the tensor path.
    private func encodeFallback(commandBuffer: MTLCommandBuffer,
                                pipeline: MTLComputePipelineState,
                                view: TensorView,
                                x: MTLBuffer, xOffset: Int,
                                y: MTLBuffer, yOffset: Int,
                                m: Int, n: Int, k: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(view.buffer, offset: Int(view.offset), index: 0)
        encoder.setBuffer(view.buffer, offset: Int(view.scaleOffset), index: 1)
        encoder.setBuffer(view.buffer, offset: Int(view.biasOffset), index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var tValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        encoder.setBytes(&tValue, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.stride, index: 7)
        dispatchSimdRows(encoder: encoder, rows: n, tokens: m)
        encoder.endEncoding()
    }

    /// Plain bf16/fp32 fallback (no scale/bias planes).
    private func encodePlain(commandBuffer: MTLCommandBuffer,
                             pipeline: MTLComputePipelineState,
                             view: TensorView,
                             x: MTLBuffer, xOffset: Int,
                             y: MTLBuffer, yOffset: Int,
                             m: Int, n: Int, k: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(view.buffer, offset: Int(view.offset), index: 0)
        encoder.setBuffer(x, offset: xOffset, index: 1)
        encoder.setBuffer(y, offset: yOffset, index: 2)
        var tValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        encoder.setBytes(&tValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.stride, index: 5)
        dispatchSimdRows(encoder: encoder, rows: n, tokens: m)
        encoder.endEncoding()
    }

    /// Grid (ceil(rows/8), tokens), one SIMD per output element, 8 per TG.
    private func dispatchSimdRows(encoder: MTLComputeCommandEncoder,
                                  rows: Int, tokens: Int) {
        let rowsPerTG = 8
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + rowsPerTG - 1) / rowsPerTG,
                    height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * rowsPerTG,
                                           height: 1, depth: 1))
    }
}
