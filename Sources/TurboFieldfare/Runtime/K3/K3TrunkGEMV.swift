import Foundation
import Metal
import TurboFieldfareFormat

/// Weight storage formats the K3 trunk GEMV adapter can serve.
public enum K3TrunkWeightFormat: Sendable, Equatable {
    /// MLX-style affine 4-bit, group 64, BF16 scale + bias (dtype u32).
    case int4G64
    /// MLX-style affine 8-bit, group 64, BF16 scale + bias (dtype u32).
    case int8G64
    /// Row-major BF16 matrix.
    case bf16
    /// Row-major FP32 matrix (the K3 router's storage format).
    case fp32
}

enum K3TrunkGEMVError: Error, CustomStringConvertible, Equatable {
    case unsupportedView(detail: String)

    var description: String {
        switch self {
        case .unsupportedView(let detail): return "unsupported trunk GEMV view: \(detail)"
        }
    }
}

/// Uniform GEMV access over resident K3 tensor views for the Stage-C2
/// forward runner. The affine int4/int8 paths delegate to the house
/// `DequantInt4GEMV` / `DequantInt8GEMV` dispatchers (verified shape-generic:
/// M/N are runtime parameters), so K3 trunk numerics match the v1 runtime
/// bit-for-bit. BF16/FP32 views are served by the generic-shape kernels in
/// `trunk_k3.metal` (the house has no unquantized GEMV).
///
/// Format is inferred from the view's dtype + byte sizes, so a mislabeled
/// tensor fails loudly here instead of corrupting the stream. x/y dtypes:
/// fp16 for `int4G64` / `int8G64` / `bf16`, fp32 for `fp32`.
final class K3TrunkGEMV {
    private let int4: DequantInt4GEMV
    private let int8: DequantInt8GEMV
    private let bf16PSO: MTLComputePipelineState
    private let f32PSO: MTLComputePipelineState
    private let f32RouterPSO: MTLComputePipelineState

    private static let rowsPerThreadgroup = 8
    private static let routerShape = (m: UInt32(896), n: UInt32(7_168))

    init(context: MetalContext) throws {
        self.int4 = try DequantInt4GEMV(context: context)
        self.int8 = try DequantInt8GEMV(context: context)
        let library = K3MetalLibrary.shared
        self.bf16PSO = try library.pipeline(
            device: context.device, name: "k3_gemv_bf16")
        self.f32PSO = try library.pipeline(
            device: context.device, name: "k3_gemv_f32")
        self.f32RouterPSO = try library.pipeline(
            device: context.device, name: "k3_gemv_f32",
            constants: [
                MetalFunctionConstant(index: 95, value: .uint32(Self.routerShape.m)),
                MetalFunctionConstant(index: 96, value: .uint32(Self.routerShape.n)),
                MetalFunctionConstant(index: 97, value: .bool(true)),
            ])
    }

    /// Infer the storage format of a 2-D resident view, validating byte
    /// sizes against the logical shape. Throws on anything outside the four
    /// supported layouts.
    static func format(of view: TensorView) throws -> K3TrunkWeightFormat {
        let rows = UInt64(view.shape.0)
        let columns = UInt64(view.shape.1)
        guard rows > 0, columns > 0, view.shape.2 == 0, view.shape.3 == 0 else {
            throw K3TrunkGEMVError.unsupportedView(
                detail: "shape [\(view.shape.0), \(view.shape.1), \(view.shape.2), "
                    + "\(view.shape.3)] is not a 2-D matrix")
        }
        let elements = rows * columns
        let groups = columns / UInt64(Quantization.groupSize)
        let auxBytes = rows * groups * UInt64(MemoryLayout<UInt16>.stride)
        switch view.dtype {
        case GTurboFormatV1.DType.u32.rawValue:
            guard view.scaleLength == auxBytes, view.biasLength == auxBytes else {
                throw K3TrunkGEMVError.unsupportedView(
                    detail: "affine aux sizes \(view.scaleLength)/\(view.biasLength) "
                        + "!= \(auxBytes) for [\(rows), \(columns)]")
            }
            if view.length == elements / 2 { return .int4G64 }
            if view.length == elements { return .int8G64 }
            throw K3TrunkGEMVError.unsupportedView(
                detail: "packed payload \(view.length) bytes fits neither int4 nor "
                    + "int8 for [\(rows), \(columns)]")
        case GTurboFormatV1.DType.bf16.rawValue:
            guard view.length == elements * 2,
                  view.scaleLength == 0, view.biasLength == 0 else {
                throw K3TrunkGEMVError.unsupportedView(
                    detail: "bf16 payload/aux mismatch for [\(rows), \(columns)]")
            }
            return .bf16
        case GTurboFormatV1.DType.fp32.rawValue:
            guard view.length == elements * 4,
                  view.scaleLength == 0, view.biasLength == 0 else {
                throw K3TrunkGEMVError.unsupportedView(
                    detail: "fp32 payload/aux mismatch for [\(rows), \(columns)]")
            }
            return .fp32
        default:
            throw K3TrunkGEMVError.unsupportedView(
                detail: "dtype \(view.dtype) has no trunk GEMV path")
        }
    }

    /// Encode `y = W x` for the given resident view. `x`/`y` are fp16 for the
    /// int4/int8/bf16 formats and fp32 for the fp32 format.
    func encode(commandBuffer: MTLCommandBuffer,
                view: TensorView,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0) throws {
        let format = try Self.format(of: view)
        let m = view.shape.0
        let n = view.shape.1
        switch format {
        case .int4G64:
            int4.encode(commandBuffer: commandBuffer,
                        weights: view.buffer, weightsOffset: Int(view.offset),
                        scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                        biases: view.buffer, biasesOffset: Int(view.biasOffset),
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                        m: m, n: n)
        case .int8G64:
            int8.encode(commandBuffer: commandBuffer,
                        weights: view.buffer, weightsOffset: Int(view.offset),
                        scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                        biases: view.buffer, biasesOffset: Int(view.biasOffset),
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                        m: m, n: n)
        case .bf16:
            encodePlain(commandBuffer: commandBuffer,
                        pipeline: bf16PSO,
                        weights: view.buffer, weightsOffset: Int(view.offset),
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                        m: m, n: n)
        case .fp32:
            encodePlain(commandBuffer: commandBuffer,
                        pipeline: (m, n) == Self.routerShape ? f32RouterPSO : f32PSO,
                        weights: view.buffer, weightsOffset: Int(view.offset),
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                        m: m, n: n)
        }
    }

    private func encodePlain(commandBuffer: MTLCommandBuffer,
                             pipeline: MTLComputePipelineState,
                             weights: MTLBuffer, weightsOffset: Int,
                             x: MTLBuffer, xOffset: Int,
                             y: MTLBuffer, yOffset: Int,
                             m: UInt32, n: UInt32) {
        precondition(m > 0)
        precondition(n > 0 && n % UInt32(Quantization.groupSize) == 0,
                     "N must be a positive multiple of \(Quantization.groupSize)")
        var rows = m
        var cols = n
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(x, offset: xOffset, index: 1)
        encoder.setBuffer(y, offset: yOffset, index: 2)
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&cols, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(m) + Self.rowsPerThreadgroup - 1)
                        / Self.rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }
}
