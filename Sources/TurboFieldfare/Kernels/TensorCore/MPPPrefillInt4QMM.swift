import Foundation
import Metal

public final class MPPPrefillInt4QMM {
    public enum Variant: String, Sendable {
        case control
        case apple10V1 = "apple10-v1"
        case apple10BF16 = "apple10-bf16"
    }

    public enum Path: String, Sendable {
        case affineThreadgroupF16 = "affine-threadgroup-f16"
        case affineThreadgroupF16Apple10V1 = "affine-threadgroup-f16-apple10-v1"
        case affineThreadgroupBF16Apple10 = "affine-threadgroup-bf16-apple10"
        case fallback
    }

    public struct PathMetadata: Sendable, Equatable {
        public let path: Path
        public let tileM: Int
        public let tileN: Int
        public let tileK: Int
        public let modelDerivedCPUHeapScratchBytes: Int
        public let expandedWeightBufferBytes: Int
        public let threadgroupBytes: Int
        public let cooperativeValueBytesEstimate: Int
        public let simultaneousWeightTiles: Int
        public let weightBufferOffsetAlignment: Int
        public let weightBufferOffsetRemainder128: Int
        public let fallbackCount: Int
    }

    public static let tileM = 64
    public static let tileN = 32
    public static let tileK = Quantization.groupSize
    public static let threadgroupBytes = tileN * tileK * MemoryLayout<Float16>.stride
    public static let cooperativeValueBytesEstimate = 2 * tileM * tileN * MemoryLayout<Float>.stride

    private let context: MetalContext
    private var controlPSO: MTLComputePipelineState?
    private var apple10PSO: MTLComputePipelineState?
    private var apple10BF16PSO: MTLComputePipelineState?
    private let variant: Variant
    public let unavailableReason: String?

    public init(context: MetalContext, variant: Variant = .control) {
        self.context = context
        self.variant = variant
        var controlPSO: MTLComputePipelineState?
        var apple10PSO: MTLComputePipelineState?
        var apple10BF16PSO: MTLComputePipelineState?
        var unavailableReason: String?
        let apple10FunctionName: String? = switch variant {
        case .control:
            nil
        case .apple10V1:
            "mpp_prefill_affine_threadgroup_f16_apple10_v1"
        case .apple10BF16:
            "mpp_prefill_affine_threadgroup_bf16_apple10_v1"
        }
        do {
            if let apple10FunctionName {
                guard context.device.supportsFamily(.apple10) else {
                    throw NSError(
                        domain: "MPPPrefillInt4QMM",
                        code: 2,
                        userInfo: [NSLocalizedDescriptionKey:
                            "\(apple10FunctionName) needs an Apple10 GPU family device"])
                }
            }
            let library = try MetalContext.privateLibrary(device: context.device, module: "tensorops")
            guard let controlFunction = library.makeFunction(
                name: "mpp_prefill_affine_threadgroup_f16") else {
                throw NSError(
                    domain: "MPPPrefillInt4QMM",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey:
                        "mpp_prefill_affine_threadgroup_f16 symbol unavailable"])
            }
            controlPSO = try context.device.makeComputePipelineState(
                function: controlFunction)
            if let apple10FunctionName {
                // Name the function this variant actually needs, and say when
                // the device is the reason. The BF16 (vision projector) variant
                // used to report the f16 symbol as missing, blaming a symbol
                // that is present and hiding the real cause.
                guard let apple10Function = library.makeFunction(
                    name: apple10FunctionName) else {
                    throw NSError(
                        domain: "MPPPrefillInt4QMM",
                        code: 2,
                        userInfo: [NSLocalizedDescriptionKey:
                            "\(apple10FunctionName) symbol unavailable"])
                }
                if variant == .apple10BF16 {
                    apple10BF16PSO = try context.device.makeComputePipelineState(
                        function: apple10Function)
                } else {
                    apple10PSO = try context.device.makeComputePipelineState(
                        function: apple10Function)
                }
            }
        } catch {
            unavailableReason = String(describing: error)
        }
        self.controlPSO = controlPSO
        self.apple10PSO = apple10PSO
        self.apple10BF16PSO = apple10BF16PSO
        self.unavailableReason = unavailableReason
    }

    public var isAvailable: Bool {
        controlPSO != nil && (
            variant == .control
                || variant == .apple10V1 && apple10PSO != nil
                || variant == .apple10BF16 && apple10BF16PSO != nil)
    }

    @discardableResult
    public func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       m: Int,
                       n: Int,
                       k: Int) -> PathMetadata {
        let supportedShape = m > 0 && n > 0 && k > 0
            && k.isMultiple(of: Self.tileK)
        let useBF16 = variant == .apple10BF16
        // The BF16 variant is the vision projector and has no control kernel, so
        // a short output previously selected no pipeline at all and fell back
        // without projecting — a 3x3 patch grid pools to a single row. The
        // Apple10 kernels bound-check every store against M and the dispatch
        // rounds up to a whole tile, so a partial tile is correct; the F16 path
        // keeps its 64-row floor because the control kernel serves it better.
        let useApple10 = variant != .control
            && context.device.supportsFamily(.apple10)
            && (useBF16 || m >= 64)
            && k.isMultiple(of: 128)
            && (useBF16 ? apple10BF16PSO != nil : apple10PSO != nil)
        let selectedPSO = useApple10
            ? (useBF16 ? apple10BF16PSO : apple10PSO)
            : useBF16 ? nil : controlPSO
        guard supportedShape,
              weightsOffset >= 0,
              scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              xOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              yOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              let pso = selectedPSO,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return metadata(path: .fallback,
                            weightsOffset: weightsOffset,
                            fallbackCount: 1)
        }

        encoder.setComputePipelineState(pso)
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
            MTLSize(width: (n + (useApple10 ? 64 : Self.tileN) - 1)
                        / (useApple10 ? 64 : Self.tileN),
                    height: (m + Self.tileM - 1) / Self.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pso.threadExecutionWidth * 4,
                                           height: 1,
                                           depth: 1))
        encoder.endEncoding()
        return metadata(path: useBF16 && useApple10
                            ? .affineThreadgroupBF16Apple10
                            : useApple10
                                ? .affineThreadgroupF16Apple10V1
                                : .affineThreadgroupF16,
                        weightsOffset: weightsOffset,
                        fallbackCount: variant != .control && !useApple10 ? 1 : 0)
    }

    private func metadata(path: Path,
                          weightsOffset: Int,
                          fallbackCount: Int) -> PathMetadata {
        let apple10 = path == .affineThreadgroupF16Apple10V1
            || path == .affineThreadgroupBF16Apple10
        return PathMetadata(
            path: path,
            tileM: Self.tileM,
            tileN: apple10 ? 64 : Self.tileN,
            tileK: apple10 ? 128 : Self.tileK,
            modelDerivedCPUHeapScratchBytes: 0,
            expandedWeightBufferBytes: 0,
            threadgroupBytes: apple10 ? 16_384 : Self.threadgroupBytes,
            cooperativeValueBytesEstimate: apple10 ? 32_768 : Self.cooperativeValueBytesEstimate,
            simultaneousWeightTiles: 1,
            weightBufferOffsetAlignment: 1,
            weightBufferOffsetRemainder128: weightsOffset % 128,
            fallbackCount: fallbackCount)
    }
}
