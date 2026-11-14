import Foundation
import Metal
import Testing
import TinyTitanFormat

@testable import TinyTitan
@testable import TinyTitanRepackCore
import TinyTitanCore

// The whole compressed path on a toy install: convert a real raw install with
// the repacker's writer, load it with the runtime's loader, and compare every
// routed expert's bytes with the raw file they came from.
//
// The pieces are each tested on their own -- the codec, the index, the writer,
// the streamer -- but this is the only place where the file the writer produces
// is handed to the loader that decides whether a layer is compressed. A naming
// or geometry disagreement between the two is silent weight corruption, so the
// comparison here is byte for byte.
extension ModelLoaderTests {
    @Test func compressedInstallLoadsAndStreamsEveryExpert() throws {
        let raw = try Self.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: raw) }
        let compressed = raw.deletingLastPathComponent()
            .appendingPathComponent(raw.lastPathComponent + "-compressed")
        defer { try? FileManager.default.removeItem(at: compressed) }

        let result = try CompressExpertsTool.run(
            options: CompressExpertsOptions(
                inputSSDAI: raw.path, output: compressed.path, overwrite: false))
        #expect(result.layerCount > 0)
        #expect(result.compressedBytes < result.rawBytes)

        let layoutData = try Data(
            contentsOf: compressed.appendingPathComponent("packed_experts/layout.json"))
        let layout = try SSDAIPackedExpertsLayoutCodec.decode(layoutData)
        #expect(layout.compression == .lzfse)

        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(
            directoryURL: compressed, device: device, expecting: .qwenToy())
        let stride = Int(layout.expertStride)
        for layer in layout.layers {
            let rawLayer = try Data(
                contentsOf: raw.appendingPathComponent("packed_experts/\(layer.file)"))
            for expert in 0..<layout.expertsPerLayer {
                let view = try model.routedExpert(layer: layer.layer, expert: expert)
                let got = Self.copyBytes(of: view.buffer, offset: view.offset, count: stride)
                let start = expert * stride
                let want = [UInt8](rawLayer[start..<start + stride])
                #expect(
                    got == want,
                    "layer \(layer.layer) expert \(expert) did not decode to its raw bytes")
            }
        }
    }

    /// A slot's bytes, copied out of the Metal buffer the streamer handed back.
    private static func copyBytes(
        of buffer: MTLBuffer, offset: UInt64, count: Int
    ) -> [UInt8] {
        let start = buffer.contents().advanced(by: Int(offset))
            .assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: start, count: count))
    }
}
