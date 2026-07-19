import Metal

/// Make-/read- helpers for FP16 MTLBuffers. Replaces the byte-identical
/// inline copies that lived in 12+ test files.
public enum Fp16Buffer {
    public static func make(_ device: MTLDevice, values: [Float]) -> MTLBuffer? {
        let halves = values.map { Float16($0) }
        return halves.withUnsafeBufferPointer { ptr in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: halves.count * MemoryLayout<Float16>.size,
                options: .storageModeShared
            )
        }
    }

    public static func make(_ device: MTLDevice, halves: [Float16]) -> MTLBuffer? {
        halves.withUnsafeBufferPointer { ptr in
            device.makeBuffer(
                bytes: ptr.baseAddress!,
                length: halves.count * MemoryLayout<Float16>.size,
                options: .storageModeShared
            )
        }
    }

    public static func make(_ device: MTLDevice, count: Int) -> MTLBuffer? {
        device.makeBuffer(
            length: count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        )
    }

    public static func read(_ buf: MTLBuffer, count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        let base = buf.contents()
        for i in 0..<count {
            let h = base.load(
                fromByteOffset: i * MemoryLayout<Float16>.size,
                as: Float16.self
            )
            out[i] = Float(h)
        }
        return out
    }

    public static func readHalf(_ buf: MTLBuffer, count: Int) -> [Float16] {
        var out = [Float16](repeating: 0, count: count)
        let base = buf.contents()
        for i in 0..<count {
            out[i] = base.load(
                fromByteOffset: i * MemoryLayout<Float16>.size,
                as: Float16.self
            )
        }
        return out
    }
}
