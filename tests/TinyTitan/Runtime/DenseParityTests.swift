import Testing
import Foundation
import TinyTitanCore
@testable import Qwen35Dense4B4Bit
@testable import TinyTitan

/// Parity of the 7.0 dense variant against the 6.x runtime CPU engine on the
/// same install (plan task 8). Model-gated: set TINYTITAN_DENSE_PARITY_MODEL to
/// a dense .ssdai install (e.g. models/qwen3.5_4B_4Bit). The 2B/4B/9B are the
/// deliberate subjects here (the dense family is what this variant serves).
struct DenseParityTests {
    private static func installURL() -> URL? {
        guard let path = ProcessInfo.processInfo.environment["TINYTITAN_DENSE_PARITY_MODEL"]
        else { return nil }
        return URL(fileURLWithPath: path)
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_DENSE_PARITY_MODEL"] != nil,
            "requires a dense .ssdai install; set TINYTITAN_DENSE_PARITY_MODEL"))
    func greedyParityAgainst6x() throws {
        let directory = try #require(Self.installURL())
        // A fixed token-id prompt: parity is about the forward pass, so both
        // engines receive the identical ids (no tokenizer in the loop).
        let prompt = [1, 100, 200, 300, 400]
        let maximumTokens = 24

        // 6.x reference: the runtime CPU engine.
        let referenceSnapshot = try TinyTitan.AffineSnapshot(ssdai: directory)
        let reference = try TinyTitan.CPUQwen35(snapshot: referenceSnapshot)
        let expected = try reference.generate(prompt: prompt, maximumTokens: maximumTokens)

        // 7.0 variant.
        let context = EngineContext(metal: nil, physicalRAMBytes: 24 * 1024 * 1024 * 1024)
        let handle = LoadedModelHandle(tensors: [:], directoryURL: directory)
        let variant = try Qwen35Dense4B4BitVariant(context: context, model: handle)
        var state = SessionState()
        try variant.prefill(prompt, state: &state)
        var produced: [Int] = []
        for _ in 0..<maximumTokens {
            produced.append(try variant.decode(&state))
        }

        #expect(
            produced == expected,
            "7.0 variant diverged from 6.x: \(produced) vs \(expected)")
    }
}
