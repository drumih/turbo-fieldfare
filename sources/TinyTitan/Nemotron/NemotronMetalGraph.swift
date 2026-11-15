import Foundation
import Metal
import TinyTitanCore

/// The Nemotron graph with the hidden state, Mamba state and KV cache resident
/// on the GPU.
///
/// This is the measurement path. `NemotronMetal` (the matvec backend) is
/// correct but launch-latency bound: this model needs a CPU-side norm, conv or
/// scan between a layer's GEMVs, and an M3's GPU downclocks in the few
/// milliseconds it is idle, so each of its ~350 waits costs ~10 ms even though
/// the same GEMV takes 0.8 ms back to back. Here a whole mamba or attention
/// layer is encoded into one command buffer, and the graph stops only at a MoE
/// router, where 9 of 240-496 experts are chosen on the CPU. That is 40 waits a
/// token instead of ~350.
public final class NemotronMetalGraph {
    public let install: NemotronInstall
    public let geometry: NemotronGeometry

    let context: MetalContext
    let int4: MTLComputePipelineState
    let int8: MTLComputePipelineState
    let rmsnorm: MTLComputePipelineState
    let rmsnormGated: MTLComputePipelineState
    let convSilu: MTLComputePipelineState
    let ssm: MTLComputePipelineState
    let relu2: MTLComputePipelineState
    let add: MTLComputePipelineState
    let moeCombine: MTLComputePipelineState
    let embedding: MTLComputePipelineState
    let attention: MTLComputePipelineState
    let bf16Gemv: MTLComputePipelineState
    let kvAppend: MTLComputePipelineState

    let weights: MTLBuffer
    var layers: [Int: MTLBuffer] = [:]

    let hidden: MTLBuffer
    let normalized: MTLBuffer
    let projection: MTLBuffer
    let convOutput: MTLBuffer
    let ssmOutput: MTLBuffer
    let mambaNormalized: MTLBuffer
    let layerOutput: MTLBuffer
    let query: MTLBuffer
    let key: MTLBuffer
    let value: MTLBuffer
    let attentionOutput: MTLBuffer
    let latent: MTLBuffer
    let latentAccumulator: MTLBuffer
    let expertUp: MTLBuffer
    let expertDown: MTLBuffer
    let expertWeights: MTLBuffer
    let expertScratch: MTLBuffer
    let routerLogits: MTLBuffer
    let sharedHidden: MTLBuffer
    let sharedOutput: MTLBuffer
    let moeOutput: MTLBuffer
    let logitsBuffer: MTLBuffer
    let mambaStates: MTLBuffer
    let convStates: MTLBuffer
    let keyValues: MTLBuffer
    let scoreScratch: MTLBuffer

    let mambaIndex: [Int]
    let convDimension: Int
    let maxSequence: Int
    var position = 0
    var normBuffers: [Int: MTLBuffer] = [:]
    var mambaNormBuffers: [Int: MTLBuffer] = [:]
    var finalNormBuffer: MTLBuffer

    public init(
        install: NemotronInstall, context: MetalContext, maxSequence: Int
    ) throws {
        self.install = install
        self.geometry = install.geometry
        self.context = context
        self.maxSequence = maxSequence
        let g = install.geometry
        convDimension = g.mambaIntermediate + 2 * g.mambaGroups * g.ssmStateSize
        var index: [Int] = []
        var counter = 0
        for kind in g.layerKinds {
            if kind == .mamba {
                index.append(counter)
                counter += 1
            } else {
                index.append(-1)
            }
        }
        mambaIndex = index

        func constant(_ index: Int) -> MetalFunctionConstant {
            MetalFunctionConstant(index: index, value: .bool(false))
        }
        int4 = try context.pipeline(
            "dequant_int4_gemv_simd",
            constants: [constant(22), constant(26)])
        int8 = try context.pipeline(
            "dequant_int8_gemv_simd", constants: [constant(72)])
        rmsnorm = try context.pipeline("nemotron_rmsnorm_f16")
        rmsnormGated = try context.pipeline("nemotron_rmsnorm_gated_f16")
        convSilu = try context.pipeline("nemotron_conv_silu_f16")
        ssm = try context.pipeline("nemotron_ssm_f16")
        relu2 = try context.pipeline("nemotron_relu2_f16")
        add = try context.pipeline("nemotron_add_f16")
        moeCombine = try context.pipeline("nemotron_moe_combine_f16")
        embedding = try context.pipeline("nemotron_embedding_f16")
        attention = try context.pipeline("nemotron_attention_f16")
        bf16Gemv = try context.pipeline("nemotron_bf16_gemv")
        kvAppend = try context.pipeline("nemotron_kv_append_f16")

        let page = Int(getpagesize())
        func rounded(_ value: Int) -> Int { ((value + page - 1) / page) * page }
        func wrap(_ mapping: (base: UnsafeRawPointer, size: Int), name: String) throws
            -> MTLBuffer
        {
            guard
                let buffer = context.device.makeBuffer(
                    bytesNoCopy: UnsafeMutableRawPointer(mutating: mapping.base),
                    length: rounded(mapping.size),
                    options: .storageModeShared, deallocator: nil)
            else {
                throw NemotronInstallError.unsupported("cannot wrap \(name) in Metal")
            }
            return buffer
        }
        let residentMapping = install.weightsMapping
        if ProcessInfo.processInfo.environment["TINYTITAN_NEMOTRON_NOCOPY"] == "1",
            let buffer = context.device.makeBuffer(
                bytesNoCopy: UnsafeMutableRawPointer(mutating: residentMapping.base),
                length: rounded(residentMapping.size),
                options: .storageModeShared, deallocator: nil)
        {
            weights = buffer
        } else if let buffer = context.device.makeBuffer(
            length: residentMapping.size, options: .storageModeShared)
        {
            // A GPU read of a file-backed mmap faults a page at a time on the
            // GPU's path, and that is what made the first graph version slower
            // than the CPU. The resident tensors are 5-9 GB, so they are
            // copied once into wired shared memory at load.
            memcpy(buffer.contents(), residentMapping.base, residentMapping.size)
            weights = buffer
        } else {
            throw NemotronInstallError.unsupported("cannot allocate the resident weight buffer")
        }
        for (layer, _) in install.experts {
            guard let mapping = install.layerMapping(layer) else { continue }
            layers[layer] = try wrap(mapping, name: "layer \(layer) experts")
        }

        func half(_ count: Int) -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: max(count, 1) * MemoryLayout<Float16>.size,
                    options: .storageModeShared)
            else { preconditionFailure("Metal refused an FP16 scratch buffer") }
            return buffer
        }
        func float(_ count: Int) -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: max(count, 1) * MemoryLayout<Float>.size,
                    options: .storageModeShared)
            else { preconditionFailure("Metal refused an FP32 scratch buffer") }
            return buffer
        }
        hidden = half(g.hiddenSize)
        normalized = half(g.hiddenSize)
        projection = half(
            2 * g.mambaIntermediate + convDimension + g.mambaHeads + g.hiddenSize)
        convOutput = half(convDimension)
        ssmOutput = half(g.mambaIntermediate)
        mambaNormalized = half(g.mambaIntermediate)
        layerOutput = half(g.hiddenSize)
        query = half(g.attentionHeads * g.attentionHeadDim)
        key = half(g.attentionKVHeads * g.attentionHeadDim)
        value = half(g.attentionKVHeads * g.attentionHeadDim)
        attentionOutput = half(g.attentionHeads * g.attentionHeadDim)
        latent = half(g.moeLatentSize)
        latentAccumulator = half(g.moeLatentSize)
        expertUp = half(g.topKExperts * g.maxMoeIntermediate)
        expertDown = half(g.topKExperts * g.moeLatentSize)
        expertWeights = half(g.topKExperts)
        let expertStride = install.experts.values.map(\.stride).max() ?? 0
        expertScratch = half(max(1, g.topKExperts * expertStride / 2))
        routerLogits = float(g.maxExperts)
        sharedHidden = half(g.moeSharedIntermediate)
        sharedOutput = half(g.hiddenSize)
        moeOutput = half(g.hiddenSize)
        logitsBuffer = half(g.vocabSize)
        mambaStates = float(
            max(1, counter * g.mambaHeads * g.mambaHeadDim * g.ssmStateSize))
        convStates = float(max(1, counter * convDimension * g.convKernel))
        keyValues = half(
            max(
                1,
                g.attentionLayers.count * maxSequence * 2 * g.attentionKVHeads
                    * g.attentionHeadDim))
        scoreScratch = float(g.attentionHeads * maxSequence)
        func bf16Buffer(_ values: [Float]) -> MTLBuffer {
            guard
                let buffer = context.device.makeBuffer(
                    length: values.count * 2, options: .storageModeShared)
            else { preconditionFailure("Metal refused a norm-weight buffer") }
            let pointer = buffer.contents().assumingMemoryBound(to: UInt16.self)
            for index in values.indices {
                pointer[index] = UInt16(truncatingIfNeeded: values[index].bitPattern >> 16)
            }
            return buffer
        }
        for layer in 0..<g.numLayers {
            normBuffers[layer] = bf16Buffer(
                try install.floatVector(
                    "language_model.backbone.layers.\(layer).norm.weight"))
            if g.layerKinds[layer] == .mamba {
                mambaNormBuffers[layer] = bf16Buffer(
                    try install.floatVector(
                        "language_model.backbone.layers.\(layer).mixer.norm.weight"))
            }
        }
        finalNormBuffer = bf16Buffer(
            try install.floatVector("language_model.backbone.norm_f.weight"))
    }

    /// A command buffer, or a clear failure. A Metal device that refuses one is
    /// not a condition any caller can recover from, and force-unwrapping it is
    /// what the lint gate exists to keep out.
    func newCommandBuffer() -> MTLCommandBuffer {
        guard let command = context.queue.makeCommandBuffer() else {
            preconditionFailure("Metal refused a command buffer")
        }
        return command
    }

    func computeEncoder(_ command: MTLCommandBuffer) -> MTLComputeCommandEncoder {
        guard let encoder = command.makeComputeCommandEncoder() else {
            preconditionFailure("Metal refused a compute encoder")
        }
        return encoder
    }

    public func reset() {
        position = 0
        memset(hidden.contents(), 0, hidden.length)
        memset(mambaStates.contents(), 0, mambaStates.length)
        memset(convStates.contents(), 0, convStates.length)
        memset(keyValues.contents(), 0, keyValues.length)
    }

    /// Run one token. Returns FP32 logits.
    public func step(token: Int) throws -> [Float] {
        guard position < maxSequence else {
            throw NemotronInstallError.unsupported("sequence exceeds the KV cache")
        }
        let g = geometry
        var command = newCommandBuffer()
        try encodeEmbedding(command, token: token)
        let requested =
            ProcessInfo.processInfo.environment["TINYTITAN_NEMOTRON_LAYERS"]
            .flatMap { Int($0) } ?? g.numLayers
        for layer in 0..<min(g.numLayers, max(0, requested)) {
            switch g.layerKinds[layer] {
            case .mamba:
                try encodeMambaLayer(command, layer: layer)
            case .attention:
                try encodeAttentionLayer(command, layer: layer)
            case .moe:
                // The router is the one point the CPU has to see: 9 of
                // 240-496 experts are chosen from 496 logits.
                try encodeMoERouter(command, layer: layer)
                command.commit()
                command.waitUntilCompleted()
                let selected = try selectExperts(layer: layer)
                command = newCommandBuffer()
                try encodeMoEExperts(
                    command, layer: layer, experts: selected.experts,
                    weights: selected.weights)
            }
        }
        try encodeHead(command)
        command.commit()
        command.waitUntilCompleted()
        position += 1
        return try readLogits()
    }

    // MARK: - Kernel helpers

    func encodeNorm(
        _ command: MTLCommandBuffer, layer: Int, from source: MTLBuffer,
        into destination: MTLBuffer
    ) throws {
        guard let weight = normBuffers[layer] else { return }
        var n = UInt32(geometry.hiddenSize)
        var eps = geometry.normEpsilon
        let encoder = computeEncoder(command)
        encoder.setComputePipelineState(rmsnorm)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(destination, offset: 0, index: 2)
        encoder.setBytes(&n, length: 4, index: 3)
        encoder.setBytes(&eps, length: 4, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeNorm(
        _ command: MTLCommandBuffer, weight: MTLBuffer, from source: MTLBuffer,
        into destination: MTLBuffer, count: Int
    ) throws {
        var n = UInt32(count)
        var eps = geometry.normEpsilon
        let encoder = computeEncoder(command)
        encoder.setComputePipelineState(rmsnorm)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(destination, offset: 0, index: 2)
        encoder.setBytes(&n, length: 4, index: 3)
        encoder.setBytes(&eps, length: 4, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeGatedNorm(
        _ command: MTLCommandBuffer, layer: Int
    ) throws {
        guard let buffer = mambaNormBuffers[layer] else { return }
        let g = geometry
        var group = UInt32(g.mambaIntermediate / g.mambaGroups)
        var eps = g.mambaEpsilon
        let encoder = computeEncoder(command)
        encoder.setComputePipelineState(rmsnormGated)
        encoder.setBuffer(ssmOutput, offset: 0, index: 0)
        // `proj` starts with `gate`, the same layout the reference splits.
        encoder.setBuffer(projection, offset: 0, index: 1)
        encoder.setBuffer(buffer, offset: 0, index: 2)
        encoder.setBuffer(mambaNormalized, offset: 0, index: 3)
        encoder.setBytes(&group, length: 4, index: 4)
        encoder.setBytes(&eps, length: 4, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: g.mambaGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeGEMV(
        _ command: MTLCommandBuffer,
        buffer: MTLBuffer, weight: Int, scales: Int, biases: Int, bits: Int,
        rows: Int, columns: Int,
        x: MTLBuffer, xOffset: Int, y: MTLBuffer, yOffset: Int
    ) {
        let encoder = computeEncoder(command)
        encoder.setComputePipelineState(bits == 4 ? int4 : int8)
        encoder.setBuffer(buffer, offset: weight, index: 0)
        encoder.setBuffer(buffer, offset: scales, index: 1)
        encoder.setBuffer(buffer, offset: biases, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var m = UInt32(rows)
        var n = UInt32(columns)
        encoder.setBytes(&m, length: 4, index: 5)
        encoder.setBytes(&n, length: 4, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * 8, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRelu2(_ command: MTLCommandBuffer, buffer: MTLBuffer, count: Int) {
        var n = UInt32(count)
        let encoder = computeEncoder(command)
        encoder.setComputePipelineState(relu2)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBuffer(buffer, offset: 0, index: 1)
        encoder.setBytes(&n, length: 4, index: 2)
        dispatch(encoder, count: count)
        encoder.endEncoding()
    }

    func encodeAdd(
        _ command: MTLCommandBuffer, a: MTLBuffer, b: MTLBuffer, out: MTLBuffer,
        count: Int
    ) {
        var n = UInt32(count)
        let encoder = computeEncoder(command)
        encoder.setComputePipelineState(add)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        encoder.setBytes(&n, length: 4, index: 3)
        dispatch(encoder, count: count)
        encoder.endEncoding()
    }

    func dispatch(_ encoder: MTLComputeCommandEncoder, count: Int) {
        let threads = 256
        let groups = (count + threads - 1) / threads
        encoder.dispatchThreadgroups(
            MTLSize(width: max(groups, 1), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    func readLogits() throws -> [Float] {
        let count = geometry.vocabSize
        let source = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        var out = [Float](repeating: 0, count: count)
        for index in 0..<count { out[index] = Float(source[index]) }
        return out
    }
}
