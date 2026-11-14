import Foundation
import Testing
import TinyTitanFormat

@testable import TinyTitanRepackCore
import TinyTitanCore

/// The rANS layer writer and its sidecar.
///
/// The GPU kernel is verified separately against a converted install; this pins
/// the format contract the kernel relies on — one model per block index, blocks
/// tiling an expert, one contiguous payload per expert — and decodes the payload
/// with a reference rANS loop, so a writer change cannot silently produce bytes
/// only the kernel's assumptions happen to accept.
@Suite("rANS expert layer writer")
struct RANSExpertLayerWriterTests {
    private static let stride = 4096
    private static let experts = 2
    private static let blocks = [
        RANSBlockSpec(rawOffset: 0, symbolCount: 2048, stride: 1),
        RANSBlockSpec(rawOffset: 2048, symbolCount: 1024, stride: 2),
        RANSBlockSpec(rawOffset: 2049, symbolCount: 1024, stride: 2),
    ]

    /// Compressible: a small alphabet, so the writer has something to model.
    private static func expertBytes(_ expert: Int) -> [UInt8] {
        (0..<stride).map { index in
            UInt8(truncatingIfNeeded: (index / 7 + expert * 13) % 5)
        }
    }

    private static func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rans-writer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test("A layer round-trips through the writer, the sidecar and a reference decoder")
    func layerRoundTrips() throws {
        let directory = try Self.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("layer_00.bin")
        var raw = Data()
        for expert in 0..<Self.experts { raw.append(contentsOf: Self.expertBytes(expert)) }
        try raw.write(to: source)
        let destination = directory.appendingPathComponent("out.bin")
        let index = directory.appendingPathComponent("out.ridx")

        let result = try RANSExpertLayerWriter.write(
            sourcePath: source.path, destinationPath: destination.path,
            indexPath: index.path,
            geometry: RANSExpertLayerGeometry(
                expertStride: UInt64(Self.stride), expertsPerLayer: Self.experts,
                blocks: Self.blocks))
        #expect(result.blockCount == Self.blocks.count)
        #expect(result.compressedBytes > 0)
        #expect(result.indexBytes > 0)

        let payload = try Data(contentsOf: destination)
        #expect(UInt64(payload.count) == result.compressedBytes)
        let layout = try ExpertRANSIndex.decode(
            try Data(contentsOf: index),
            expectedExperts: Self.experts,
            expertStride: UInt64(Self.stride),
            payloadSize: UInt64(payload.count))
        #expect(layout.models.count == Self.blocks.count)
        #expect(layout.blocks.count == Self.experts)

        for expert in 0..<Self.experts {
            var restored = [UInt8](repeating: 0, count: Self.stride)
            for (index, block) in layout.blocks[expert].enumerated() {
                let spec = Self.blocks[index]
                let decoded = try Self.decodeBlock(
                    payload, block: block, model: layout.models[Int(block.role)],
                    streams: RANSExpertLayerWriter.streams(count: Int(spec.symbolCount)))
                #expect(decoded.count == Int(spec.symbolCount))
                for (offset, byte) in decoded.enumerated() {
                    restored[Int(spec.rawOffset) + offset * Int(spec.stride)] = byte
                }
            }
            #expect(restored == Self.expertBytes(expert))
        }
    }

    @Test("A source that is not the layout's shape is refused")
    func refusesWrongSourceSize() throws {
        let directory = try Self.directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("layer_00.bin")
        try Data(count: 100).write(to: source)
        #expect(throws: RepackError.self) {
            _ = try RANSExpertLayerWriter.write(
                sourcePath: source.path,
                destinationPath: directory.appendingPathComponent("out.bin").path,
                indexPath: directory.appendingPathComponent("out.ridx").path,
                geometry: RANSExpertLayerGeometry(
                    expertStride: UInt64(Self.stride), expertsPerLayer: Self.experts,
                    blocks: Self.blocks))
        }
    }

    /// A reference decoder for the block payload: u16 length table, then one
    /// stream per `streams`, each state-then-bytes and interleaved.
    private static func decodeBlock(
        _ payload: Data, block: ExpertRANSIndex.Block,
        model: ExpertRANSIndex.Model, streams: Int
    ) throws -> [UInt8] {
        var frequencies = [UInt32](repeating: 0, count: 256)
        var cumulative = [UInt32](repeating: 0, count: 257)
        for symbol in 0..<256 {
            frequencies[symbol] = UInt32(model.frequencies[symbol])
            cumulative[symbol + 1] = cumulative[symbol] + frequencies[symbol]
        }
        var lengths: [Int] = []
        let base = Int(block.payloadOffset)
        for stream in 0..<streams {
            lengths.append(
                Int(payload[base + stream * 2]) | Int(payload[base + stream * 2 + 1]) << 8)
        }
        var offsets: [Int] = []
        var cursor = base + streams * 2
        for length in lengths {
            offsets.append(cursor)
            cursor += length
        }
        var out = [UInt8](repeating: 0, count: Int(block.symbolCount))
        for stream in 0..<streams {
            var at = offsets[stream]
            var state =
                UInt32(payload[at]) | UInt32(payload[at + 1]) << 8
                | UInt32(payload[at + 2]) << 16 | UInt32(payload[at + 3]) << 24
            at += 8
            var position = stream
            while position < Int(block.symbolCount) {
                while state < (1 << 23) {
                    state = (state << 8) | UInt32(payload[at])
                    at += 1
                }
                let slot = Int(state & 16383)
                var symbol = 0
                while symbol < 255, Int(cumulative[symbol + 1]) <= slot { symbol += 1 }
                out[position] = UInt8(symbol)
                state = frequencies[symbol] * (state >> 14) + UInt32(slot) - cumulative[symbol]
                position += streams
            }
        }
        return out
    }
}
