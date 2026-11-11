import Foundation
import Metal
import Testing

@testable import TinyTitan

/// Two fresh decoders over the same prompt must agree bit for bit: the
/// precondition every parity and bit-identity check here rests on.
@Suite struct GGUFDecoderDeterminismTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func twoFreshDecodersAgreeBitForBit() throws {
        let path = try #require(ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"])
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let prompt: [Int32] = [785, 6722, 315, 9625, 374, 12095, 13, 576]
        func run() throws -> [[Float]] {
            let decoder = try GGUFDecoder(
                context: context, arch: .qwen38FlashNext, model: model, maxSeq: 16)
            return try prompt.indices.map { try decoder.decode(sequence: prompt, position: $0) }
        }
        let first = try run()
        let second = try run()
        for position in prompt.indices {
            let worst = zip(first[position], second[position]).map { abs($0 - $1) }.max() ?? 0
            print("position \(position): max |diff| \(worst)")
            #expect(first[position] == second[position], "position \(position) differs")
        }
    }
}
