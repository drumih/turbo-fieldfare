import Foundation
import Testing

@testable import TinyTitan

/// The CPU Q6_K decoder against llama.cpp, on the real embedding table
/// (model-gated).
///
/// Reference: llama.cpp `7c2653a`, `llama-debug -ngl 0`, the `hc_init` tensor
/// for "The capital of France is" - the embedding rows broadcast to the four
/// hyper streams, dequantised by ggml to F32. Both sides decode the same Q6_K
/// blocks to F32, so the first values must agree exactly and the row's absolute
/// sum to F32 summation order.
@Suite struct GGUFEmbedRowProbeTests {
    private static let modelPath = ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]

    private static let reference: [(id: Int, head: [Float], absoluteSum: Float)] = [
        (760, [0.001239776611328125, -0.00030994415283203125, -0.0021696090698242188], 6.048349),
        (6511, [-0.005069375038146973, 0.006759166717529297, 0.0013518333435058594], 11.299334),
        (314, [0.0009423494338989258, 0.001130819320678711, -0.001319289207458496], 3.705597),
        (9338, [0.010471343994140625, 0.0052356719970703125, -0.0052356719970703125], 11.934510),
        (369, [0.0005767345428466797, 0.004469692707061768, -0.0011534690856933594], 4.141854),
    ]

    @Test(.enabled(if: modelPath != nil))
    func q6KRowsMatchLlamaCpp() throws {
        let path = try #require(Self.modelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let location = try #require(model.tensor(named: "token_embd.weight"))
        #expect(location.tensor.type == .q6K)
        let hidden = Int(location.tensor.dimensions[0])
        let rowBytes = hidden / 256 * 210
        let shard = model.shards[location.shardIndex]
        let handle = try FileHandle(forReadingFrom: shard.url)
        defer { try? handle.close() }
        for reference in Self.reference {
            try handle.seek(
                toOffset: shard.dataOffset + location.tensor.offset
                    + UInt64(reference.id * rowBytes))
            let data = try #require(try handle.read(upToCount: rowBytes))
            let row = try GGUFDequant.decode(type: .q6K, bytes: [UInt8](data))
            #expect(row.count == hidden)
            #expect(Array(row.prefix(3)) == reference.head, "id \(reference.id)")
            let sum = row.reduce(Float(0)) { $0 + abs($1) }
            #expect(abs(sum - reference.absoluteSum) < 1e-3, "id \(reference.id): \(sum)")
        }
    }
}
