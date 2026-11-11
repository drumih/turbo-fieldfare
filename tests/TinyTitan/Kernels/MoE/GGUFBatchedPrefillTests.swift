import Foundation
import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan

/// Layer-major prefill against the token-at-a-time loop. Every token runs the
/// same kernels in the same order either way, so the last token's logits must
/// be bit-identical; the grouping rule is pinned without a model.
@Suite struct GGUFBatchedPrefillTests {
    @Test func groupsKeepOrderAndFitTheCapacity() {
        let selections = [[1, 2], [2, 3], [4, 5], [5, 6], [1, 2]]
        let groups = GGUFDecoder.expertGroups(selections, capacity: 4)
        #expect(groups == [[0, 1], [2, 3], [4]])
        for group in groups {
            #expect(Set(group.flatMap { selections[$0] }).count <= 4)
        }
        #expect(GGUFDecoder.expertGroups(selections, capacity: 100) == [[0, 1, 2, 3, 4]])
        #expect(GGUFDecoder.expertGroups([], capacity: 4).isEmpty)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func layerMajorPrefillIsBitIdenticalToTokenByToken() throws {
        let path = try #require(ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"])
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let decoder = try GGUFDecoder(
            context: context, arch: .qwen38FlashNext, model: model, maxSeq: 64)
        // Long enough for the PLE n-gram window, several full-attention
        // positions and more than one expert group at the default cache size.
        let prompt: [Int32] = [
            785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 9856, 374, 19846, 13,
            576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315,
        ]
        let reference = try decoder.prefillTokenByToken(tokens: prompt)
        decoder.resetSequence()
        let again = try decoder.prefillTokenByToken(tokens: Array(prompt))
        #expect(again.logits == reference.logits, "reset does not restore the start state")
        decoder.resetSequence()
        let batched = try decoder.prefill(tokens: prompt)
        let worst = zip(batched.logits, reference.logits).map { abs($0 - $1) }.max() ?? 0
        let argmax = { (values: [Float]) in values.indices.max { values[$0] < values[$1] } }
        print(
            "batched vs token-by-token: max |diff| \(worst), argmax "
                + "\(String(describing: argmax(batched.logits))) vs "
                + "\(String(describing: argmax(reference.logits)))")
        #expect(batched.logits.count == reference.logits.count)
        #expect(batched.logits == reference.logits, "layer-major logits differ")
        print(
            "GGUF prefill: token-by-token \(String(format: "%.2f", reference.seconds)) s, "
                + "layer-major \(String(format: "%.2f", batched.seconds)) s "
                + "for \(prompt.count) tokens")
    }
}
