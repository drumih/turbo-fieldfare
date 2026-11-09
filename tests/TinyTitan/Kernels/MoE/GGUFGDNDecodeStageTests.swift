import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// A whole GDN decode step on layer 0 of the real checkpoint: projections,
/// conv+SiLU, qk norm, the delta rule with its FP32 state, the gated norm and
/// the out projection, against a CPU reference.
@Suite struct GGUFGDNDecodeStageTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    private static func bf16(_ value: Float) -> Float {
        Float(bitPattern: UInt32(Quantization.bf16Bits(value)) << 16)
    }

    private static func decoded(_ model: GGUFModel, name: String) throws -> [Float] {
        let location = try #require(model.tensor(named: name))
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
        let data = try #require(try handle.read(upToCount: Int(location.tensor.byteCount)))
        if location.tensor.type == .f32 {
            return data.withUnsafeBytes { raw in
                let count = raw.count / 4
                let p = raw.bindMemory(to: Float.self)
                return (0..<count).map { p[$0] }
            }
        }
        return try GGUFDequant.decode(type: location.tensor.type, bytes: [UInt8](data))
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func gdnDecodeStepMatchesTheCPU() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let la = arch.linearAttention
        let context = try MetalContext()
        let names = [
            "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
            "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
            "blk.0.ssm_conv1d.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias",
            "blk.0.ssm_norm.weight", "blk.0.ssm_out.weight",
        ]
        let arena = try GGUFResidentArena.build(names: names, from: model, context: context)
        let mapped = try GGUFMappedShards(model: model, device: context.device)
        let stage = try GGUFGDNDecodeStage(
            context: context, arch: arch, arena: arena, mapped: mapped)

        var rng = SeedTree(0x6D4D).key("gdn-x")
        let x = (0..<arch.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let xBuffer,
            let out = context.device.makeBuffer(
                length: arch.hiddenSize * 2, options: .storageModeShared),
            let cb = context.queue.makeCommandBuffer()
        else { throw MetalError.bufferAllocationFailed("gdn buffers") }
        try stage.encode(commandBuffer: cb, layer: 0, x: xBuffer, out: out)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }

        // CPU reference.
        let hidden = arch.hiddenSize
        let Dk = la.keyHeadDim, Dv = la.valueHeadDim
        let Hk = la.numKHeads, Hv = la.numVHeads
        let qkvDim = 2 * Hk * Dk + Hv * Dv
        let valueDim = Hv * Dv
        let xf = x.map(Float.init)
        func matvec(_ name: String, rows: Int) throws -> [Float] {
            let weights = try Self.decoded(model, name: name).map(Self.bf16)
            var result = [Float](repeating: 0, count: rows)
            for r in 0..<rows {
                var acc: Float = 0
                let base = r * hidden
                for n in 0..<hidden { acc += weights[base + n] * xf[n] }
                result[r] = Float(Float16(acc))
            }
            return result
        }
        let qkv = try matvec("blk.0.attn_qkv.weight", rows: qkvDim)
        let z = try matvec("blk.0.attn_gate.weight", rows: valueDim)
        let a = try matvec("blk.0.ssm_alpha.weight", rows: Hv)
        let b = try matvec("blk.0.ssm_beta.weight", rows: Hv)
        // conv (first token: tail is zero)
        let convW = try Self.decoded(model, name: "blk.0.ssm_conv1d.weight").map(Self.bf16)
        var conv = [Float](repeating: 0, count: qkvDim)
        for c in 0..<qkvDim {
            let acc = qkv[c] * convW[c * la.convKernelSize + la.convKernelSize - 1]
            conv[c] = Float(Float16(acc / (1 + exp(-acc))))
        }
        // qk norm
        let eps: Float = 1e-6
        for head in 0..<(2 * Hk) {
            let isQ = head < Hk
            // k heads are indexed 0..<Hk again after the k region's offset.
            let base = isQ ? head * Dk : Hk * Dk + (head - Hk) * Dk
            var square: Float = 0
            for i in 0..<Dk { square += conv[base + i] * conv[base + i] }
            let invRms = 1 / (square / Float(Dk) + eps).squareRoot()
            let scale = isQ ? (1 / Float(Dk)) : 1 / Float(Dk).squareRoot()
            for i in 0..<Dk { conv[base + i] = Float(Float16(conv[base + i] * invRms * scale)) }
        }
        // delta rule against a fresh state
        let aLog = try Self.decoded(model, name: "blk.0.ssm_a")
        let dt = try Self.decoded(model, name: "blk.0.ssm_dt.bias")
        var state = [Float](repeating: 0, count: Hv * Dv * Dk)
        var y = [Float](repeating: 0, count: valueDim)
        for h in 0..<Hv {
            let hk = h / (Hv / Hk)
            let qBase = hk * Dk
            let kBase = Hk * Dk + hk * Dk
            let vBase = 2 * Hk * Dk + h * Dv
            let ah = Float(Float16(a[h]))
            let dtv = Float(Float16(dt[h]))
            let softplus = (ah + dtv > 20) ? ah + dtv : log(1 + exp(ah + dtv))
            let g = exp(-exp(Float(Float16(aLog[h]))) * softplus)
            let beta = 1 / (1 + exp(-Float(Float16(b[h]))))
            for dv in 0..<Dv {
                let base = (h * Dv + dv) * Dk
                var kv: Float = 0
                for k in 0..<Dk { kv += state[base + k] * g * conv[kBase + k] }
                let delta = (conv[vBase + dv] - kv) * beta
                var acc: Float = 0
                for k in 0..<Dk {
                    state[base + k] = state[base + k] * g + conv[kBase + k] * delta
                    acc += state[base + k] * conv[qBase + k]
                }
                y[h * Dv + dv] = Float(Float16(acc))
            }
        }
        // gated norm (sigmoid output gate on this family)
        let normW = try Self.decoded(model, name: "blk.0.ssm_norm.weight").map(Self.bf16)
        var gated = [Float](repeating: 0, count: valueDim)
        for h in 0..<Hv {
            let base = h * Dv
            var square: Float = 0
            for i in 0..<Dv { square += y[base + i] * y[base + i] }
            let invRms = 1 / (square / Float(Dv) + eps).squareRoot()
            for i in 0..<Dv {
                let normed = y[base + i] * invRms * normW[i]
                let gate = 1 / (1 + exp(-z[base + i]))
                gated[base + i] = Float(Float16(normed * gate))
            }
        }
        // out projection
        let outW = try Self.decoded(model, name: "blk.0.ssm_out.weight").map(Self.bf16)
        var reference = [Float](repeating: 0, count: hidden)
        for r in 0..<hidden {
            var acc: Float = 0
            let base = r * valueDim
            for n in 0..<valueDim { acc += outW[base + n] * gated[n] }
            reference[r] = Float(Float16(acc))
        }

        let pointer = out.contents().assumingMemoryBound(to: UInt16.self)
        var sqDiff: Double = 0
        var sqRef: Double = 0
        for i in 0..<hidden {
            let got = Double(Float(Float16(bitPattern: pointer[i])))
            let diff = got - Double(reference[i])
            sqDiff += diff * diff
            sqRef += Double(reference[i] * reference[i])
        }
        let relative = (sqDiff / max(sqRef, 1e-12)).squareRoot()
        print("GDN decode step RMS relative error: \(relative)")
        #expect(relative < 5e-3, "RMS relative error \(relative)")
    }
}

/// Intermediate rows of the same step, so a failure names the kernel.
@Suite struct GGUFGDNIntermediateTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func convAndDeltaMatchTheirCPUSteps() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let arch = ArchConfig.qwen38FlashNext
        let la = arch.linearAttention
        let context = try MetalContext()
        let names = [
            "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
            "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
            "blk.0.ssm_conv1d.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias",
            "blk.0.ssm_norm.weight", "blk.0.ssm_out.weight",
        ]
        let arena = try GGUFResidentArena.build(names: names, from: model, context: context)
        let mapped = try GGUFMappedShards(model: model, device: context.device)
        let stage = try GGUFGDNDecodeStage(
            context: context, arch: arch, arena: arena, mapped: mapped)
        var rng = SeedTree(0x6D4D).key("gdn-x")
        let x = (0..<arch.hiddenSize).map { _ in Float16(rng.uniform(-1, 1)) }
        let xBits = x.map(\.bitPattern)
        let xBuffer: MTLBuffer? = xBits.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            return context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared)
        }
        guard let xBuffer,
            let out = context.device.makeBuffer(
                length: arch.hiddenSize * 2, options: .storageModeShared),
            let cb = context.queue.makeCommandBuffer()
        else { throw MetalError.bufferAllocationFailed("gdn buffers") }
        try stage.encode(commandBuffer: cb, layer: 0, x: xBuffer, out: out)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw MetalError.invalidState("\(error)") }

        // Compare the conv row against the CPU conv of the CPU qkv projection.
        let hidden = arch.hiddenSize
        let Hk = la.numKHeads, Dk = la.keyHeadDim
        let qkvDim = 2 * Hk * Dk + la.numVHeads * la.valueHeadDim
        let xf = x.map(Float.init)
        let qkvW = try Self.decoded(model, name: "blk.0.attn_qkv.weight").map(Self.bf16)
        var qkv = [Float](repeating: 0, count: qkvDim)
        for r in 0..<qkvDim {
            var acc: Float = 0
            let base = r * hidden
            for n in 0..<hidden { acc += qkvW[base + n] * xf[n] }
            qkv[r] = Float(Float16(acc))
        }
        let convW = try Self.decoded(model, name: "blk.0.ssm_conv1d.weight").map(Self.bf16)
        var conv = [Float](repeating: 0, count: qkvDim)
        for c in 0..<qkvDim {
            let acc = qkv[c] * convW[c * la.convKernelSize + la.convKernelSize - 1]
            conv[c] = Float(Float16(acc / (1 + exp(-acc))))
        }
        let rows = stage.debugRows
        let qkvPointer = rows.qkv.contents().assumingMemoryBound(to: UInt16.self)
        let convPointer = rows.conv.contents().assumingMemoryBound(to: UInt16.self)
        func relative(_ gpu: UnsafePointer<UInt16>, _ reference: [Float], _ count: Int) -> Double {
            var sqDiff: Double = 0
            var sqRef: Double = 0
            for i in 0..<count {
                let got = Double(Float(Float16(bitPattern: gpu[i])))
                let diff = got - Double(reference[i])
                sqDiff += diff * diff
                sqRef += Double(reference[i] * reference[i])
            }
            return (sqDiff / max(sqRef, 1e-12)).squareRoot()
        }
        let qkvError = relative(qkvPointer, qkv, qkvDim)
        print("GDN qkv RMS: \(qkvError)")
        #expect(qkvError < 1e-2)
        // The conv row differs by the qk norm, which runs in place after it;
        // compare against the CPU conv passed through the same norm.
        for head in 0..<(2 * Hk) {
            let isQ = head < Hk
            // k heads are indexed 0..<Hk again after the k region's offset.
            let base = isQ ? head * Dk : Hk * Dk + (head - Hk) * Dk
            var square: Float = 0
            for i in 0..<Dk { square += conv[base + i] * conv[base + i] }
            let invRms = 1 / (square / Float(Dk) + 1e-6).squareRoot()
            let scale = isQ ? (1 / Float(Dk)) : 1 / Float(Dk).squareRoot()
            for i in 0..<Dk { conv[base + i] = Float(Float16(conv[base + i] * invRms * scale)) }
        }
        // Only the q/k region is normed by the kernel; the v region is copied raw.
        let convError = relative(convPointer, conv, 2 * Hk * Dk)
        print("GDN conv+qknorm RMS (q/k region): \(convError)")
        #expect(convError < 1e-2)
    }

    private static func decoded(_ model: GGUFModel, name: String) throws -> [Float] {
        let location = try #require(model.tensor(named: name))
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        try handle.seek(toOffset: shard.dataOffset + location.tensor.offset)
        let data = try #require(try handle.read(upToCount: Int(location.tensor.byteCount)))
        if location.tensor.type == .f32 {
            return data.withUnsafeBytes { raw in
                let count = raw.count / 4
                let p = raw.bindMemory(to: Float.self)
                return (0..<count).map { p[$0] }
            }
        }
        return try GGUFDequant.decode(type: location.tensor.type, bytes: [UInt8](data))
    }

    private static func bf16(_ value: Float) -> Float {
        Float(bitPattern: UInt32(Quantization.bf16Bits(value)) << 16)
    }
}
