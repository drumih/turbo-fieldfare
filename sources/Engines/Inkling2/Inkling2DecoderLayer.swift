import Foundation

/// The block-level attention residual: softmax-weighted read over completed
/// blocks plus the partial stream.
public struct Inkling2AttentionResidual: Sendable {
    private let query: [Float]
    private let normalizeValues: Bool
    private let hidden: Int
    private let queryCount: Int

    public init(
        tensors: Inkling2DenseTensors, layer: Int, name: String, queries: Int,
        hidden: Int
    ) throws {
        let tensorName = "\(name).query"
        if tensors.hasLayerTensor(layer, tensorName) {
            self.query = try tensors.layerMatrix(layer, tensorName)
            self.queryCount = self.query.count / hidden
        } else {
            self.query = [Float](repeating: 0, count: queries * hidden)
            self.queryCount = queries
        }
        self.normalizeValues = true
        self.hidden = hidden
    }

    public func forward(_ blocks: [[Float]], partial: [Float]) -> [Float] {
        forwardAll(blocks, partial: partial)[0]
    }

    public func forwardAll(_ blocks: [[Float]], partial: [Float]) -> [[Float]] {
        var states = blocks
        states.append(partial)
        let normalized = states.map {
            Inkling2Ops.rmsNorm($0, weight: [Float](repeating: 1, count: hidden))
        }
        var outputs: [[Float]] = []
        for queryIndex in 0..<queryCount {
            var logits: [Float] = []
            for state in normalized {
                var dot: Float = 0
                for index in 0..<hidden {
                    dot += state[index] * query[queryIndex * hidden + index]
                }
                logits.append(dot / Float(hidden).squareRoot())
            }
            let probabilities = Inkling2AttentionMath.softmax(logits)
            var output = [Float](repeating: 0, count: hidden)
            for (stateIndex, probability) in probabilities.enumerated() {
                let value = normalizeValues ? normalized[stateIndex] : states[stateIndex]
                for index in 0..<hidden { output[index] += probability * value[index] }
            }
            outputs.append(output)
        }
        return outputs
    }

    public var parts: Int { queryCount }
}

/// One decoder block for the first loop pass (no messages yet).
public struct Inkling2DecoderLayer: Sendable {
    private let attention: any Inkling2AttentionBlock
    private let mlp: Inkling2MLP
    private let tensors: Inkling2DenseTensors
    private let layer: Int
    private let hidden: Int
    private let isBlockBoundary: Bool
    private let isSliding: Bool
    private let attentionInput: Inkling2AttentionResidual?
    private let mlpInput: Inkling2AttentionResidual
    private let attnConv: [Float]
    private let mlpConv: [Float]
    private let kernelSize: Int

    public init(
        tensors: Inkling2DenseTensors, store: Inkling2ExpertStore, layer: Int,
        hidden: Int, kernelSize: Int, isBlockBoundary: Bool,
        isSliding: Bool, attention: any Inkling2AttentionBlock
    ) throws {
        self.tensors = tensors
        self.layer = layer
        self.hidden = hidden
        self.kernelSize = kernelSize
        self.isBlockBoundary = isBlockBoundary
        self.isSliding = isSliding
        self.attention = attention
        self.mlp = try Inkling2MLP(tensors: tensors, store: store, layer: layer)
        self.attentionInput =
            layer == 0
            ? nil
            : try Inkling2AttentionResidual(
                tensors: tensors, layer: layer, name: "attention_input",
                queries: 3, hidden: hidden)
        self.mlpInput = try Inkling2AttentionResidual(
            tensors: tensors, layer: layer, name: "mlp_input",
            queries: 1, hidden: hidden)
        self.attnConv = try tensors.layerMatrix(layer, "attn_sconv.weight")
        self.mlpConv = try tensors.layerMatrix(layer, "mlp_sconv.weight")
    }

    public func forward(_ x: [Float], blocks: inout [[Float]]) throws -> [Float] {
        let parts = attentionInput?.forwardAll(blocks, partial: x) ?? [x]
        let qPart = parts[0]
        let kPart = parts[min(1, parts.count - 1)]
        let vPart = parts[min(2, parts.count - 1)]
        let qNormName = isSliding ? "attn_norm.weight" : "q_input_norm.weight"
        let kNormName = isSliding ? "attn_norm.weight" : "k_input_norm.weight"
        let vNormName = isSliding ? "attn_norm.weight" : "v_input_norm.weight"
        let qNorm = try tensors.layerMatrix(layer, qNormName)
        let kNorm = try tensors.layerMatrix(layer, kNormName)
        let vNorm = try tensors.layerMatrix(layer, vNormName)
        let attentionOutput = try attention.forward(
            query: Inkling2Ops.rmsNorm(qPart, weight: qNorm),
            key: Inkling2Ops.rmsNorm(kPart, weight: kNorm),
            value: Inkling2Ops.rmsNorm(vPart, weight: vNorm))
        let attnResidual = Inkling2AttentionMath.shortConvolution(
            attentionOutput, length: 1, channels: hidden, kernel: attnConv,
            kernelSize: kernelSize, state: nil
        ).values
        if isBlockBoundary {
            blocks.append(x)
        }
        var residual = attnResidual
        if !isBlockBoundary {
            for index in 0..<hidden { residual[index] += x[index] }
        }
        let mlpInputValues = mlpInput.forward(blocks, partial: residual)
        let postNorm = try tensors.layerMatrix(layer, "mlp_norm.weight")
        let normalized = Inkling2Ops.rmsNorm(mlpInputValues, weight: postNorm)
        let mlpOutput = try mlp.forward(normalized)
        let convOutput = Inkling2AttentionMath.shortConvolution(
            mlpOutput, length: 1, channels: hidden, kernel: mlpConv,
            kernelSize: kernelSize, state: nil
        ).values
        var output = residual
        for index in 0..<hidden { output[index] += convOutput[index] }
        return output
    }
}

public protocol Inkling2AttentionBlock: Sendable {
    func forward(query: [Float], key: [Float], value: [Float]) throws -> [Float]
}
