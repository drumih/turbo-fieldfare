import Foundation
import Accelerate
import TurboFieldfareFormat

/// An independently reconstructed activation and its GPU counterpart.  The
/// reference always reads the loaded `TensorView` bytes directly; it never
/// shares the Metal gather or normalization kernels under test.
public struct K3ActivationComparison: Sendable, Equatable {
    public let name: String
    public let elements: Int
    public let maxAbsoluteError: Float
    public let maxRelativeError: Float

    public init(name: String, elements: Int, maxAbsoluteError: Float,
                maxRelativeError: Float) {
        self.name = name
        self.elements = elements
        self.maxAbsoluteError = maxAbsoluteError
        self.maxRelativeError = maxRelativeError
    }
}

/// Result of the cheap real-weight activation probe.  It deliberately stops
/// before K3's streamed MoE stack: a full CPU replay of a 2.78T model would
/// require reading and multiplying every selected expert and is not a useful
/// interactive diagnosis.  These two probes establish that the actual bundle
/// bytes, affine decode, embedding gather, BF16 norm weight, and first GPU
/// activation boundary agree before any KDA/MLA/MoE work begins.
public struct K3ActivationDiagnostics: Sendable, Equatable {
    public let token: Int32
    public let embedding: K3ActivationComparison
    public let layer0InputNorm: K3ActivationComparison
    public let layer0KDAProjections: [K3ActivationComparison]

    /// Both operations write fp16 activations.  This permits normal parallel
    /// reduction differences in RMSNorm while still catching a wrong layout,
    /// quantization mode, tensor offset, or tensor-name mapping decisively.
    public var passed: Bool {
        embedding.maxAbsoluteError <= 0.000_01
            && layer0InputNorm.maxRelativeError <= 0.002
            && layer0InputNorm.maxAbsoluteError <= 0.002
            && layer0KDAProjections.allSatisfy {
                $0.maxAbsoluteError <= 0.02 && $0.maxRelativeError <= 0.01
            }
    }

    public var summaryLine: String {
        let status = passed ? "pass" : "FAIL"
        let kdaStatus = layer0KDAProjections.allSatisfy {
            $0.maxAbsoluteError <= 0.02 && $0.maxRelativeError <= 0.01
        } ? "pass" : "FAIL"
        return "[k3-activation \(status) token=\(token) "
            + "embedding(abs=\(Self.metric(embedding.maxAbsoluteError)) "
            + "rel=\(Self.metric(embedding.maxRelativeError))) "
            + "layer0.input_norm(abs=\(Self.metric(layer0InputNorm.maxAbsoluteError)) "
            + "rel=\(Self.metric(layer0InputNorm.maxRelativeError))) "
            + "kda_proj=\(kdaStatus)]"
    }

    private static func metric(_ value: Float) -> String {
        String(format: "%.3e", value)
    }
}

public struct K3RouterActivationDiagnostics: Sendable, Equatable {
    public let layer: Int
    public let tokenInChunk: Int
    public let scores: K3ActivationComparison
    public let actualExperts: [Int]
    public let referenceExperts: [Int]
    public let maxWeightError: Float

    public var passed: Bool {
        scores.maxAbsoluteError <= 0.002
            && scores.maxRelativeError <= 0.005
            && actualExperts == referenceExperts
            && maxWeightError <= 0.002
    }

    public var summaryLine: String {
        let status = passed ? "pass" : "FAIL"
        let absolute = String(format: "%.3e", scores.maxAbsoluteError)
        let relative = String(format: "%.3e", scores.maxRelativeError)
        let weightError = String(format: "%.3e", maxWeightError)
        return "[k3-router \(status) layer=\(layer) tokenInChunk=\(tokenInChunk) "
            + "scores(abs=\(absolute) rel=\(relative)) "
            + "weights=\(weightError) "
            + "actual=\(actualExperts) reference=\(referenceExperts)]"
    }
}

public struct K3HeadActivationDiagnostics: Sendable, Equatable {
    public let logits: K3ActivationComparison
    public let tokenIDs: [Int]

    public var passed: Bool {
        logits.maxAbsoluteError <= 0.05 && logits.maxRelativeError <= 0.01
    }

    public var summaryLine: String {
        let status = passed ? "pass" : "FAIL"
        let absolute = String(format: "%.3e", logits.maxAbsoluteError)
        let relative = String(format: "%.3e", logits.maxRelativeError)
        return "[k3-head \(status) rows=\(tokenIDs) "
            + "abs=\(absolute) rel=\(relative)]"
    }
}

public enum K3ActivationDiagnosticError: Error, CustomStringConvertible, Equatable {
    case unsupportedAffineView(detail: String)
    case unsupportedVectorView(detail: String)

    public var description: String {
        switch self {
        case .unsupportedAffineView(let detail):
            return "unsupported K3 affine diagnostic view: \(detail)"
        case .unsupportedVectorView(let detail):
            return "unsupported K3 vector diagnostic view: \(detail)"
        }
    }
}

/// Scalar reference implementation for the first real activation boundary.
/// This is intentionally independent from `K3TrunkGEMV` and Metal: it reads
/// packed affine weights plus BF16 auxiliaries directly from the resident
/// model buffer and performs the documented `q * scale + bias` reconstruction
/// on the CPU.
public enum K3ActivationReference {
    /// Reconstruct one affine G64 matrix row from the exact resident bytes.
    public static func affineRow(_ view: TensorView, row: Int) throws -> [Float] {
        let rows = Int(view.shape.0)
        let columns = Int(view.shape.1)
        guard rows > 0, columns > 0, row >= 0, row < rows,
              columns % Quantization.groupSize == 0 else {
            throw K3ActivationDiagnosticError.unsupportedAffineView(
                detail: "shape [\(view.shape.0), \(view.shape.1)] row \(row)")
        }
        let elements = rows * columns
        let packedBytes: Int
        if view.length == UInt64(elements / 2) {
            packedBytes = columns / 2
        } else if view.length == UInt64(elements) {
            packedBytes = columns
        } else {
            throw K3ActivationDiagnosticError.unsupportedAffineView(
                detail: "payload \(view.length) for [\(rows), \(columns)]")
        }
        let groups = columns / Quantization.groupSize
        let expectedAux = UInt64(rows * groups * MemoryLayout<UInt16>.stride)
        guard view.dtype == GTurboFormatV1.DType.u32.rawValue,
              view.scaleLength == expectedAux, view.biasLength == expectedAux else {
            throw K3ActivationDiagnosticError.unsupportedAffineView(
                detail: "dtype/auxiliary lengths do not describe affine G64")
        }

        let base = view.buffer.contents()
        let packed = base.advanced(by: Int(view.offset) + row * packedBytes)
            .bindMemory(to: UInt8.self, capacity: packedBytes)
        let scales = base.advanced(by: Int(view.scaleOffset)
                                  + row * groups * MemoryLayout<UInt16>.stride)
            .bindMemory(to: UInt16.self, capacity: groups)
        let biases = base.advanced(by: Int(view.biasOffset)
                                  + row * groups * MemoryLayout<UInt16>.stride)
            .bindMemory(to: UInt16.self, capacity: groups)
        var result = [Float](repeating: 0, count: columns)
        for group in 0..<groups {
            let scale = Quantization.bf16ToFloat(scales[group])
            let bias = Quantization.bf16ToFloat(biases[group])
            let start = group * Quantization.groupSize
            for lane in 0..<Quantization.groupSize {
                let value: UInt8
                if packedBytes == columns {
                    value = packed[start + lane]
                } else {
                    let byte = packed[(start + lane) >> 1]
                    value = (lane & 1) == 0 ? (byte & 0x0F) : (byte >> 4)
                }
                result[start + lane] = Float(value) * scale + bias
            }
        }
        return result
    }

    /// Read an exact BF16 vector without converting through a GPU buffer.
    public static func bf16Vector(_ view: TensorView) throws -> [Float] {
        let count = Int(view.shape.0)
        guard count > 0, view.shape.1 == 0,
              view.dtype == GTurboFormatV1.DType.bf16.rawValue,
              view.length == UInt64(count * MemoryLayout<UInt16>.stride),
              view.scaleLength == 0, view.biasLength == 0 else {
            throw K3ActivationDiagnosticError.unsupportedVectorView(
                detail: "dtype/shape/length do not describe BF16 vector")
        }
        let ptr = view.buffer.contents().advanced(by: Int(view.offset))
            .bindMemory(to: UInt16.self, capacity: count)
        return (0..<count).map { Quantization.bf16ToFloat(ptr[$0]) }
    }

    /// Kimi RMSNorm, rounded at the fp16 output boundary just like the
    /// runner's `RMSNorm.encodeBF16W` path.
    public static func rmsNormFP16(_ x: [Float], weight: [Float], eps: Float) -> [Float] {
        precondition(x.count == weight.count)
        var sumSquares: Float = 0
        for value in x { sumSquares += value * value }
        let inverse = 1 / (sumSquares / Float(x.count) + eps).squareRoot()
        return (0..<x.count).map { Float(Float16(x[$0] * inverse * weight[$0])) }
    }

    /// CPU scalar affine GEMV for selected output rows.  Sampling rows keeps
    /// the real 7K-wide K3 matrices cheap enough for an interactive probe.
    public static func affineGEMVSamples(_ view: TensorView, x: [Float],
                                         rows: [Int]) throws -> [Float] {
        try affineGEMVSamples(view, x: x, rows: rows, roundToFP16: true)
    }

    static func affineGEMVSamples(_ view: TensorView, x: [Float], rows: [Int],
                                  roundToFP16: Bool) throws -> [Float] {
        let columns = Int(view.shape.1)
        precondition(x.count == columns)
        return try rows.map { row in
            let weights = try affineRow(view, row: row)
            var sum: Float = 0
            for column in 0..<columns { sum += weights[column] * x[column] }
            return roundToFP16 ? Float(Float16(sum)) : sum
        }
    }

    public static func sampledRows(count: Int, samples: Int = 16) -> [Int] {
        guard count > 1 else { return [0] }
        let n = min(samples, count)
        return (0..<n).map { $0 * (count - 1) / max(n - 1, 1) }
    }

    static func routerDiagnostics(model: K3Model, layer: Int, tokenInChunk: Int,
                                  input: [Float], actualScores: [Float]) throws
        -> K3RouterActivationDiagnostics {
        let gate = model.routerGate(layer: layer)
        let experts = model.config.moeNumExperts
        let hidden = model.config.hiddenSize
        guard gate.dtype == GTurboFormatV1.DType.fp32.rawValue,
              gate.length == UInt64(experts * hidden * MemoryLayout<Float>.stride) else {
            throw K3ActivationDiagnosticError.unsupportedVectorView(
                detail: "router gate is not fp32 [experts, hidden]")
        }
        let weights = gate.buffer.contents().advanced(by: Int(gate.offset))
            .bindMemory(to: Float.self, capacity: experts * hidden)
        var logits = [Float](repeating: 0, count: experts)
        input.withUnsafeBufferPointer { x in
            logits.withUnsafeMutableBufferPointer { y in
                for expert in 0..<experts {
                    var dot: Float = 0
                    vDSP_dotpr(weights + expert * hidden, 1, x.baseAddress!, 1,
                               &dot, vDSP_Length(hidden))
                    y[expert] = dot
                }
            }
        }
        let referenceScores = logits.map { 1 / (1 + exp(-$0)) }
        let biasView = model.routerCorrectionBias(layer: layer)
        let biasPtr = biasView.buffer.contents().advanced(by: Int(biasView.offset))
            .bindMemory(to: Float.self, capacity: experts)
        let bias = Array(UnsafeBufferPointer(start: biasPtr, count: experts))
        let actual = K3Router.selectTopK(scores: actualScores, bias: bias,
                                         topK: model.config.moeTopKExperts)
        let reference = K3Router.selectTopK(scores: referenceScores, bias: bias,
                                            topK: model.config.moeTopKExperts)
        var maxWeightError: Float = 0
        for index in actual.weights.indices {
            maxWeightError = max(maxWeightError,
                                 abs(actual.weights[index] - reference.weights[index]))
        }
        return K3RouterActivationDiagnostics(
            layer: layer, tokenInChunk: tokenInChunk,
            scores: compare(name: "router.scores", actual: actualScores,
                            reference: referenceScores),
            actualExperts: actual.indices.map(Int.init),
            referenceExperts: reference.indices.map(Int.init),
            maxWeightError: maxWeightError)
    }

    static func fp16(_ values: [Float]) -> [Float] {
        values.map { Float(Float16($0)) }
    }

    static func compare(name: String, actual: [Float], reference: [Float])
        -> K3ActivationComparison {
        precondition(actual.count == reference.count)
        var maxAbsolute: Float = 0
        var maxReference: Float = 0
        for index in actual.indices {
            maxAbsolute = max(maxAbsolute, abs(actual[index] - reference[index]))
            maxReference = max(maxReference, abs(reference[index]))
        }
        return K3ActivationComparison(
            name: name, elements: actual.count, maxAbsoluteError: maxAbsolute,
            maxRelativeError: maxAbsolute / max(maxReference, 1e-6))
    }
}
