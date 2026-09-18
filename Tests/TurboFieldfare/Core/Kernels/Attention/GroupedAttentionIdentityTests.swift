import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized)
struct GroupedAttentionIdentityTests {
    // The old contiguous reduction passes tolerance checks while changing scores.
    // One-key chunks expose scores directly as m; longer chunks pin recurrence state.
    @Test(arguments: [1, 3, 16, 17, 33, 65, 257, 16_384, 65_536])
    func exactOrder(seqLen: Int) throws {
        try compare(seqLen: seqLen, pattern: "random")
    }

    @Test(arguments: ["cancellation", "lateMaximum", "equal", "wide"])
    func adversarial(pattern: String) throws {
        try compare(seqLen: 257, pattern: pattern)
    }

    // Fusing weight * V instead of old * alpha changes component 3 by one ULP.
    @Test func twoKeyRecurrence() throws {
        try compare(seqLen: 2, pattern: "random", chunkLength: 2)
    }

    @Test(arguments: [UInt64(0), 1, 0x12345678, UInt64.max])
    func variedSeeds(seed: UInt64) throws {
        try compare(seqLen: 257, pattern: "random", seed: seed)
    }

    private func compare(seqLen: Int, pattern: String, chunkLength: Int? = nil,
                         seed: UInt64 = 0x9E571) throws {
        let ctx = try MetalContext()
        func buffer(_ count: Int, _ stride: Int) throws -> MTLBuffer {
            try #require(ctx.device.makeBuffer(length: count * stride, options: .storageModeShared))
        }
        let q = try buffer(8192, 2)
        let k = try buffer(seqLen * 1024, 2)
        let v = try buffer(seqLen * 1024, 2)
        var state = seed
        func random() -> Float16 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float16(Float(Int32(truncatingIfNeeded: state >> 32)) / Float(Int32.max) * 0.5)
        }
        let qp = q.contents().bindMemory(to: Float16.self, capacity: 8192)
        for i in 0..<8192 {
            qp[i] = pattern == "wide" ? Float16(Float(random()) * 6)
                : (pattern == "random" ? random() : 1)
        }
        let kp = k.contents().bindMemory(to: Float16.self, capacity: seqLen * 1024)
        let vp = v.contents().bindMemory(to: Float16.self, capacity: seqLen * 1024)
        for i in 0..<(seqLen * 1024) {
            switch pattern {
            case "cancellation":
                kp[i] = [Float16(32), Float16(0.03125), Float16(-32), Float16(-0.015625)][i % 4]
            case "lateMaximum":
                // Every chunk sees a new maximum after weights have underflowed.
                kp[i] = Float16((i / 1024) % 17 == 16 ? 2 : -2)
            case "equal": kp[i] = 0.125
            case "wide": kp[i] = Float16(Float(random()) * 6)
            default: kp[i] = random()
            }
            vp[i] = random()
        }
        let constants = [
            MetalFunctionConstant(index: 60, value: .uint32(512)),
            MetalFunctionConstant(index: 61, value: .uint32(16)),
            MetalFunctionConstant(index: 62, value: .uint32(2)),
            MetalFunctionConstant(index: 63, value: .bool(true)),
        ]
        let exact = try ctx.pipeline("attention_decode_partial", constants: constants + [
            MetalFunctionConstant(index: 65, value: .uint32(16))
        ])
        let grouped = try ctx.pipeline("attention_decode_full_grouped_vec_partial", constants: constants)
        let combine = try ctx.pipeline("attention_decode_combine", constants: constants + [
            MetalFunctionConstant(index: 65, value: .uint32(16))
        ])
        func run(_ pso: MTLComputePipelineState, grouped: Bool) throws -> [MTLBuffer] {
            let outputs = try [buffer(256, 4), buffer(256, 4), buffer(256 * 512, 4), buffer(8192, 2)]
            let cb = try #require(ctx.queue.makeCommandBuffer())
            let enc = try #require(cb.makeComputeCommandEncoder())
            enc.setComputePipelineState(pso)
            for (index, b) in ([q, k, v] + Array(outputs.prefix(3))).enumerated() {
                enc.setBuffer(b, offset: 0, index: index)
            }
            for (offset, value) in [512, 16, 2, seqLen, 0, chunkLength ?? (seqLen + 15) / 16, 16].enumerated() {
                var value = UInt32(value)
                enc.setBytes(&value, length: 4, index: 6 + offset)
            }
            var scale: Float = 1
            enc.setBytes(&scale, length: 4, index: 13)
            enc.dispatchThreadgroups(MTLSize(width: grouped ? 32 : 256, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
            let merge = try #require(cb.makeComputeCommandEncoder())
            merge.setComputePipelineState(combine)
            for (index, b) in outputs.enumerated() { merge.setBuffer(b, offset: 0, index: index) }
            var hd: UInt32 = 512, nc: UInt32 = 16
            merge.setBytes(&hd, length: 4, index: 4)
            merge.setBytes(&nc, length: 4, index: 5)
            merge.dispatchThreadgroups(MTLSize(width: 16, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            merge.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw error }
            try #require(cb.status == .completed)
            return outputs
        }
        if seqLen >= 17, chunkLength == nil {
            let outputs = try [buffer(8192, 2), buffer(8192, 2)]
            for index in 0..<2 {
                let attention = try Attention(context: ctx)
                let cb = try #require(ctx.queue.makeCommandBuffer())
                attention.encodeFull(commandBuffer: cb, q: q, k: k, v: v, out: outputs[index],
                                     headDim: 512, numQHeads: 16, numKVHeads: 2,
                                     seqLen: UInt32(seqLen), scale: 1, useGroupedVector: index == 1)
                cb.commit()
                cb.waitUntilCompleted()
                if let error = cb.error { throw error }
                try #require(cb.status == .completed)
            }
            #expect(memcmp(outputs[0].contents(), outputs[1].contents(), outputs[0].length) == 0,
                    "runtime output mismatch length=\(seqLen) pattern=\(pattern) seed=\(seed)")
        }
        let control = try run(exact, grouped: false)
        for repetition in 0..<2 {
            let candidate = try run(grouped, grouped: true)
            for index in 0..<4 {
                let names = ["m/score", "d", "o", "FP16"]
                let stride = index == 3 ? 2 : 4
                let a = control[index].contents()
                let b = candidate[index].contents()
                var changed = 0
                var first = -1
                var maxError: Float = 0
                for element in 0..<(control[index].length / stride) {
                    let differs: Bool
                    if stride == 4 {
                        differs = a.load(fromByteOffset: element * 4, as: UInt32.self) != b.load(fromByteOffset: element * 4, as: UInt32.self)
                        if differs {
                            maxError = max(maxError, abs(a.load(fromByteOffset: element * 4, as: Float.self) - b.load(fromByteOffset: element * 4, as: Float.self)))
                        }
                    } else {
                        differs = a.load(fromByteOffset: element * 2, as: UInt16.self) != b.load(fromByteOffset: element * 2, as: UInt16.self)
                        if differs {
                            maxError = max(maxError, abs(Float(a.load(fromByteOffset: element * 2, as: Float16.self)) - Float(b.load(fromByteOffset: element * 2, as: Float16.self))))
                        }
                    }
                    if differs { changed += 1; if first < 0 { first = element } }
                }
                print("D1a length=\(seqLen) pattern=\(pattern) seed=\(seed) repetition=\(repetition) state=\(names[index]) changed=\(changed) first=\(first) maxAbs=\(maxError)")
                if first >= 0 {
                    if stride == 4 {
                        print("D1a first exact=\(a.load(fromByteOffset: first * 4, as: Float.self)) candidate=\(b.load(fromByteOffset: first * 4, as: Float.self)) exactBits=\(a.load(fromByteOffset: first * 4, as: UInt32.self)) candidateBits=\(b.load(fromByteOffset: first * 4, as: UInt32.self))")
                    } else {
                        print("D1a first exactBits=\(a.load(fromByteOffset: first * 2, as: UInt16.self)) candidateBits=\(b.load(fromByteOffset: first * 2, as: UInt16.self))")
                    }
                }
                #expect(changed == 0, "length=\(seqLen) pattern=\(pattern) state=\(names[index]) first=\(first) changed=\(changed) maxAbs=\(maxError)")
            }
        }
    }
}
