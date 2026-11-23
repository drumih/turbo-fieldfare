import Foundation
import Metal
import TinyTitanCore
import TinyTitanFormat

/// One tensor the graph reads: its bytes on the GPU and the GEMV shape
/// (`columns` is GGML's fast axis ne0, `rows` the product of the rest). For a
/// routed-expert tensor the operand is expert 0's matrix and `byteCount` is
/// `experts * expertStride`, so the kernels step from expert to expert by the
/// install's per-expert blob stride.
struct InklingQ4Tensor {
    let operand: GGUFGemvIds.Operand
    let type: GGUFType
    let rows: Int
    let columns: Int
    let byteCount: Int
}

/// Every tensor of one stored layer the decode graph reads.
struct InklingQ4LayerWeights {
    let index: Int
    let isSliding: Bool
    let attnQ: InklingQ4Tensor
    let attnK: InklingQ4Tensor
    let attnV: InklingQ4Tensor
    let attnOutput: InklingQ4Tensor
    let attnGate: InklingQ4Tensor
    let attnRel: InklingQ4Tensor
    let attnSink: InklingQ4Tensor
    let qClip: InklingQ4Tensor
    let relBias: InklingQ4Tensor
    let relClip: InklingQ4Tensor
    let kConv: InklingQ4Tensor
    let vConv: InklingQ4Tensor
    let attnPostConv: InklingQ4Tensor
    let ffnPostConv: InklingQ4Tensor
    /// Sliding layers: one input norm. Full layers: Q, K and V input norms.
    let inputNorms: [InklingQ4Tensor]
    /// Full layers only: input-dependent K and V convolution taps.
    let kConvProj: InklingQ4Tensor?
    let vConvProj: InklingQ4Tensor?
    let attnResQuery: InklingQ4Tensor?
    let ffnResQuery: InklingQ4Tensor
    let ffnNorm: InklingQ4Tensor
    let router: InklingQ4Tensor
    let routerBias: InklingQ4Tensor
    let gateUp: InklingQ4Tensor
    let down: InklingQ4Tensor
    let gateUpScale: InklingQ4Tensor?
    let downScale: InklingQ4Tensor?
    /// Loop message controllers (attn, k, v, mlp): down, gain, forget.
    let loopDown: [InklingQ4Tensor]
    let loopGain: [InklingQ4Tensor]
    let loopForget: [InklingQ4Tensor]
}

/// The weights of an Inkling 2 Q4 install. The resident tensors are no-copy
/// Metal buffers over `model_weights.bin`; the routed experts are no-copy
/// buffers over each `packed_experts/layer_NN.bin`, read in place at the
/// install's per-expert stride. The per-expert F32 scales the NVFP4 layers
/// carry sit inside each expert's blob, so they are gathered once into one
/// contiguous array per layer, which is the shape the kernels index by expert.
final class InklingQ4Weights {
    let manifest: InklingQ4Manifest
    let layers: [InklingQ4LayerWeights]
    let tokenEmbedding: InklingQ4Tensor
    let tokenNorm: InklingQ4Tensor
    let output: InklingQ4Tensor
    let outputNorm: InklingQ4Tensor
    let outputResQuery: InklingQ4Tensor
    let loopResQuery: InklingQ4Tensor?
    let loopKVResQuery: InklingQ4Tensor?
    /// Bytes of every mapped file: the resident weights and the expert layers.
    let mappedBytes: Int
    private let mappings: [MappedFile]

    /// With `cache`, the routed-expert tensors are its slot buffers and the
    /// layer files are not mapped; without one they are read in place.
    init(
        install: InklingQ4Install, manifest: InklingQ4Manifest, device: MTLDevice,
        cache: InklingQ4ExpertCache? = nil
    ) throws {
        self.manifest = manifest
        let residentFile = try MappedFile(
            url: install.directory.appendingPathComponent(InklingQ4Install.residentFile),
            device: device)
        var mappings = [residentFile]
        let entries = install.resident.entries
        func tensor(_ name: String) throws -> InklingQ4Tensor {
            guard let entry = entries[name] else {
                throw InklingQ4Manifest.Error.missingTensor(name)
            }
            guard entry.fileOffset + entry.sizeBytes <= UInt64(residentFile.length) else {
                throw InklingQ4Manifest.Error.invalidGeometry("\(name) lies past the file")
            }
            let (rows, columns) = Self.matrix(entry.shape)
            return InklingQ4Tensor(
                operand: .init(residentFile.buffer, offset: Int(entry.fileOffset)),
                type: try InklingQ4Manifest.type(indexCode: entry.dtype, tensor: name),
                rows: rows, columns: columns, byteCount: Int(entry.sizeBytes))
        }
        func optional(_ name: String) throws -> InklingQ4Tensor? {
            entries[name] == nil ? nil : try tensor(name)
        }
        tokenEmbedding = try tensor("token_embd.weight")
        tokenNorm = try tensor("token_embd_norm.weight")
        output = try tensor("output.weight")
        outputNorm = try tensor("output_norm.weight")
        outputResQuery = try tensor("output_res_query.weight")
        loopResQuery = try optional("loop_res_query.weight")
        loopKVResQuery = try optional("loop_kv_res_query.weight")
        var layers: [InklingQ4LayerWeights] = []
        for layer in manifest.layers {
            let p = "blk.\(layer.index)."
            let looped =
                manifest.loopPasses > 1
                && layer.index >= manifest.loopStart && layer.index < manifest.loopEnd
            let modules = looped ? ["attn", "k", "v", "mlp"] : []
            let experts: ExpertTensors
            if let cache {
                let slots = cache.tensors(layer: layer, hidden: manifest.hidden)
                experts = ExpertTensors(
                    file: nil, gateUp: slots.gateUp, down: slots.down,
                    gateUpScale: slots.gateUpScale, downScale: slots.downScale)
            } else {
                experts = try ExpertTensors(
                    install: install, layer: layer, manifest: manifest, device: device)
            }
            if let file = experts.file { mappings.append(file) }
            layers.append(
                InklingQ4LayerWeights(
                    index: layer.index, isSliding: layer.isSliding,
                    attnQ: try tensor(p + "attn_q.weight"),
                    attnK: try tensor(p + "attn_k.weight"),
                    attnV: try tensor(p + "attn_v.weight"),
                    attnOutput: try tensor(p + "attn_output.weight"),
                    attnGate: try tensor(p + "attn_gate.weight"),
                    attnRel: try tensor(p + "attn_rel.weight"),
                    attnSink: try tensor(p + "attn_sink_proj.weight"),
                    qClip: try tensor(p + "attn_q_clip.weight"),
                    relBias: try tensor(p + "attn_rel_bias.weight"),
                    relClip: try tensor(p + "attn_rel_clip.weight"),
                    kConv: try tensor(p + "attn_k_conv.weight"),
                    vConv: try tensor(p + "attn_v_conv.weight"),
                    attnPostConv: try tensor(p + "attn_post_conv.weight"),
                    ffnPostConv: try tensor(p + "ffn_post_conv.weight"),
                    inputNorms: layer.isSliding
                        ? [try tensor(p + "attn_norm.weight")]
                        : try ["q", "k", "v"].map { try tensor(p + "attn_\($0)_inp_norm.weight") },
                    kConvProj: layer.isSliding ? nil : try tensor(p + "attn_k_conv_proj.weight"),
                    vConvProj: layer.isSliding ? nil : try tensor(p + "attn_v_conv_proj.weight"),
                    attnResQuery: try optional(p + "attn_res_query.weight"),
                    ffnResQuery: try tensor(p + "ffn_res_query.weight"),
                    ffnNorm: try tensor(p + "ffn_norm.weight"),
                    router: try tensor(p + "ffn_gate_inp.weight"),
                    routerBias: try tensor(p + "exp_probs_b.bias"),
                    gateUp: experts.gateUp, down: experts.down,
                    gateUpScale: experts.gateUpScale, downScale: experts.downScale,
                    loopDown: try modules.map { try tensor(p + "loop_\($0)_down.weight") },
                    loopGain: try modules.map { try tensor(p + "loop_\($0)_gain.weight") },
                    loopForget: try modules.map { try tensor(p + "loop_\($0)_forget.weight") }))
        }
        self.layers = layers
        self.mappings = mappings
        mappedBytes = mappings.reduce(0) { $0 + $1.length }
    }

    /// (rows, columns) of an index shape: the shape is the ggml dimensions
    /// reversed and zero-padded, so the last non-zero entry is ne0.
    static func matrix(_ shape: (UInt32, UInt32, UInt32, UInt32)) -> (rows: Int, columns: Int) {
        let dimensions = [shape.0, shape.1, shape.2, shape.3].filter { $0 != 0 }.map(Int.init)
        guard let columns = dimensions.last else { return (0, 0) }
        return (dimensions.dropLast().reduce(1, *), columns)
    }
}

/// One layer's routed experts over its mapped `layer_NN.bin`.
private struct ExpertTensors {
    let file: MappedFile?
    let gateUp: InklingQ4Tensor
    let down: InklingQ4Tensor
    let gateUpScale: InklingQ4Tensor?
    let downScale: InklingQ4Tensor?

    init(
        file: MappedFile?, gateUp: InklingQ4Tensor, down: InklingQ4Tensor,
        gateUpScale: InklingQ4Tensor?, downScale: InklingQ4Tensor?
    ) {
        self.file = file
        self.gateUp = gateUp
        self.down = down
        self.gateUpScale = gateUpScale
        self.downScale = downScale
    }

    init(
        install: InklingQ4Install, layer: InklingQ4Manifest.Layer,
        manifest: InklingQ4Manifest, device: MTLDevice
    ) throws {
        let experts = manifest.expertCount
        guard let plan = install.layout.layers.first(where: { $0.layer == layer.index }),
            plan.experts.count == experts
        else {
            throw InklingQ4Install.Error.layout("layer \(layer.index) has no full expert list")
        }
        // Every expert of a layer sits at one stride, with the same sub-tensor
        // offsets; the kernels rely on both.
        let first = plan.experts[0]
        let stride = experts > 1 ? plan.experts[1].offset - first.offset : first.size
        for (rank, expert) in plan.experts.enumerated() {
            guard expert.expert ?? rank == rank, expert.offset == UInt64(rank) * stride,
                expert.tensors == first.tensors
            else {
                throw InklingQ4Install.Error.layout(
                    "layer \(layer.index) expert \(rank) breaks the uniform stride")
            }
        }
        let file = try MappedFile(
            url: install.directory.appendingPathComponent("packed_experts/" + plan.file),
            device: device)
        self.file = file
        guard UInt64(file.length) >= UInt64(experts) * stride else {
            throw InklingQ4Install.Error.layout("\(plan.file) is shorter than its experts")
        }
        let feedForward = layer.expertFeedForward
        let hidden = manifest.hidden
        guard let gateUpSlice = first.tensors["gate_up"], let downSlice = first.tensors["down"]
        else { throw InklingQ4Install.Error.layout("layer \(layer.index) lacks gate_up/down") }
        let span = experts * Int(stride)
        gateUp = InklingQ4Tensor(
            operand: .init(file.buffer, offset: Int(gateUpSlice.offset)),
            type: layer.gateUpType, rows: 2 * feedForward, columns: hidden, byteCount: span)
        down = InklingQ4Tensor(
            operand: .init(file.buffer, offset: Int(downSlice.offset)),
            type: layer.downType, rows: hidden, columns: feedForward, byteCount: span)
        gateUpScale = try Self.gathered(
            "gate_up_scale", from: first, file: file, stride: Int(stride), experts: experts,
            layer: layer.index, device: device)
        downScale = try Self.gathered(
            "down_scale", from: first, file: file, stride: Int(stride), experts: experts,
            layer: layer.index, device: device)
    }

    /// One F32 per expert, read out of every blob into one contiguous array.
    private static func gathered(
        _ key: String, from first: SSDAIExpertV1, file: MappedFile, stride: Int, experts: Int,
        layer: Int, device: MTLDevice
    ) throws -> InklingQ4Tensor? {
        guard let slice = first.tensors[key] else { return nil }
        guard slice.size == 4, slice.dtype == "F32" else {
            throw InklingQ4Install.Error.layout("layer \(layer) \(key) is not one F32")
        }
        guard let buffer = device.makeBuffer(length: experts * 4, options: .storageModeShared)
        else { throw MetalError.bufferAllocationFailed("Inkling Q4 \(key)") }
        let values = buffer.contents().bindMemory(to: Float.self, capacity: experts)
        for expert in 0..<experts {
            values[expert] = file.buffer.contents().loadUnaligned(
                fromByteOffset: expert * stride + Int(slice.offset), as: Float.self)
        }
        return InklingQ4Tensor(
            operand: .init(buffer), type: .f32, rows: experts, columns: 1,
            byteCount: experts * 4)
    }
}

/// A whole file mapped read-only and wrapped in a no-copy Metal buffer.
/// unchecked-invariant: the mapping is created once in `init`, never written,
/// and unmapped only in `deinit`, after the last reference (and so the last
/// GPU use the owner encoded and waited on) is gone.
final class MappedFile: @unchecked Sendable {
    let buffer: MTLBuffer
    let length: Int
    private let base: UnsafeMutableRawPointer
    private let mappedLength: Int

    init(url: URL, device: MTLDevice) throws {
        let fd = open(url.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else { throw ModelError.posixFailed(call: "open(\(url.path))", errno: errno) }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG, info.st_size > 0 else {
            throw ModelError.posixFailed(call: "fstat(\(url.path))", errno: errno)
        }
        let page = Int(getpagesize())
        length = Int(info.st_size)
        mappedLength = (length + page - 1) / page * page
        guard let mapped = mmap(nil, mappedLength, PROT_READ, MAP_PRIVATE, fd, 0),
            mapped != MAP_FAILED
        else { throw ModelError.posixFailed(call: "mmap(\(url.path))", errno: errno) }
        base = mapped
        guard
            let buffer = device.makeBuffer(
                bytesNoCopy: mapped, length: mappedLength, options: .storageModeShared,
                deallocator: nil)
        else {
            munmap(mapped, mappedLength)
            throw ModelError.residentBufferWrapFailed
        }
        self.buffer = buffer
    }

    deinit { munmap(base, mappedLength) }
}
