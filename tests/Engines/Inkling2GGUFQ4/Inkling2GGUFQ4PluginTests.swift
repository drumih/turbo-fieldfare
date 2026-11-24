import Foundation
import Testing
import TinyTitanCore
import TinyTitanPluginAPI

@testable import Inkling2GGUFQ4

/// The Inkling 2 Q4 plugin's declarations (TPP/1), model-free.
@Suite struct Inkling2GGUFQ4PluginTests {
    let install = InstallProbe(
        directory: URL(fileURLWithPath: "/nonexistent/inkling"), manifest: nil,
        declaredFamily: "inkling2")

    @Test func kvBitsStaySixteenWithTheMeasurement() throws {
        let kv = Inkling2GGUFQ4Plugin.parameters(for: install).kvCachePrecision
        #expect(kv.accepts(.fp16))
        let refusal = try #require(
            kv.refusal(
                for: .int8, flag: "--kv-bits", plugin: Inkling2GGUFQ4Plugin.displayName,
                spelled: { "\($0.rawValue)" }))
        #expect(refusal.hasPrefix("--kv-bits 8 is not supported by Inkling 2 Mini Q4"))
        #expect(refusal.contains("95/96"))
        let defaulted = kv.resolveDefault(.int8, flag: "--kv-bits", plugin: "Inkling")
        #expect(defaulted.value == .fp16)
        #expect(defaulted.notice != nil)
    }

    @Test func declaresItsReasoningLevels() {
        let control = Inkling2GGUFQ4Plugin.reasoningControl(for: install)
        #expect(control?.supportedLevels == [.off, .low, .medium, .high, .xhigh])
        #expect(control?.levelWhenOn == .high)
    }

    @Test func aDirectoryWithoutTheInstallIsNotClaimed() {
        #expect(Inkling2GGUFQ4Plugin.probe(install) == nil)
    }

    /// A GGUF header with three metadata keys: a string, an array of strings
    /// and an integer, in the on-disk layout (little endian, v3).
    @Test func readsHeaderMetadataWithoutTensors() throws {
        var bytes = Data()
        func u32(_ v: UInt32) {
            withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) }
        }
        func u64(_ v: UInt64) {
            withUnsafeBytes(of: v.littleEndian) { bytes.append(contentsOf: $0) }
        }
        func string(_ s: String) {
            u64(UInt64(s.utf8.count))
            bytes.append(contentsOf: Array(s.utf8))
        }
        u32(0x4655_4747)
        u32(3)
        u64(0)
        u64(3)
        string("general.architecture")
        u32(8)
        string("inkling2")
        string("tokenizer.ggml.tokens")
        u32(9)
        u32(8)
        u64(2)
        string("a")
        string("b")
        string("inkling2.context_length")
        u32(4)
        u32(1_048_576)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gguf-metadata-\(UUID().uuidString).bin")
        try bytes.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let model = try GGUFModelReader.readHeader(at: url)
        #expect(model.metadata["general.architecture"]?.string == "inkling2")
        #expect(model.metadata["tokenizer.ggml.tokens"]?.strings == ["a", "b"])
        #expect(model.architectureValue("context_length")?.integer == 1_048_576)
    }
}
