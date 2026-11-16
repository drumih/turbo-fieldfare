import Testing
import Foundation
import TinyTitanCore
@testable import Qwen35Dense2B4Bit
@testable import Qwen35Dense4B4Bit
@testable import Qwen35Dense4B8Bit
@testable import Qwen35Dense9B4Bit
@testable import Qwen35Dense9B8Bit
@testable import TinyTitan

/// Parity of the 7.0 dense variants against the 6.x runtime CPU engine on the
/// same install (plan task 8). Model-gated: set TINYTITAN_DENSE_PARITY_DIR to the
/// models directory holding the dense installs. The dense family is the
/// deliberate subject here (these variants serve it).
struct DenseParityTests {
    private static func modelsDir() -> URL? {
        guard let path = ProcessInfo.processInfo.environment["TINYTITAN_DENSE_PARITY_DIR"]
        else { return nil }
        return URL(fileURLWithPath: path)
    }

    /// Run the variant and the 6.x runtime engine on the same install and assert
    /// identical greedy tokens. Parity is about the forward pass, so both receive
    /// the identical token ids (no tokenizer in the loop).
    private static func assertParity<V: EngineVariant>(
        _ type: V.Type, install: URL
    ) throws {
        let prompt = [1, 100, 200, 300, 400]
        let maximumTokens = 16

        let referenceSnapshot = try TinyTitan.AffineSnapshot(ssdai: install)
        let reference = try TinyTitan.CPUQwen35(snapshot: referenceSnapshot)
        let expected = try reference.generate(prompt: prompt, maximumTokens: maximumTokens)

        let context = EngineContext(metal: nil, physicalRAMBytes: 24 * 1024 * 1024 * 1024)
        let handle = LoadedModelHandle(tensors: [:], directoryURL: install)
        let variant = try V(context: context, model: handle)
        var state = SessionState()
        try variant.prefill(prompt, state: &state)
        var produced: [Int] = []
        for _ in 0..<maximumTokens { produced.append(try variant.decode(&state)) }

        #expect(
            produced == expected,
            "\(V.id) diverged from 6.x: \(produced) vs \(expected)")
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_DENSE_PARITY_DIR"] != nil,
            "requires dense installs; set TINYTITAN_DENSE_PARITY_DIR"))
    func dense2B4BitParity() throws {
        let dir = try #require(Self.modelsDir())
        try Self.assertParity(
            Qwen35Dense2B4BitVariant.self,
            install: dir.appendingPathComponent("qwen3.5_2B_4Bit"))
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_DENSE_PARITY_DIR"] != nil,
            "requires dense installs; set TINYTITAN_DENSE_PARITY_DIR"))
    func dense4B4BitParity() throws {
        let dir = try #require(Self.modelsDir())
        try Self.assertParity(
            Qwen35Dense4B4BitVariant.self,
            install: dir.appendingPathComponent("qwen3.5_4B_4Bit"))
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_DENSE_PARITY_DIR"] != nil,
            "requires dense installs; set TINYTITAN_DENSE_PARITY_DIR"))
    func dense4B8BitParity() throws {
        let dir = try #require(Self.modelsDir())
        try Self.assertParity(
            Qwen35Dense4B8BitVariant.self,
            install: dir.appendingPathComponent("qwen3.5_4B_8Bit"))
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_DENSE_PARITY_DIR"] != nil,
            "requires dense installs; set TINYTITAN_DENSE_PARITY_DIR"))
    func dense9B4BitParity() throws {
        let dir = try #require(Self.modelsDir())
        try Self.assertParity(
            Qwen35Dense9B4BitVariant.self,
            install: dir.appendingPathComponent("qwen3.5_9B_4Bit"))
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_DENSE_PARITY_DIR"] != nil,
            "requires dense installs; set TINYTITAN_DENSE_PARITY_DIR"))
    func dense9B8BitParity() throws {
        let dir = try #require(Self.modelsDir())
        try Self.assertParity(
            Qwen35Dense9B8BitVariant.self,
            install: dir.appendingPathComponent("qwen3.5_9B_8Bit"))
    }
}
