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

    @Test func kvBitsAcceptAllPrecisions() {
        let kv = Inkling2GGUFQ4Plugin.parameters(for: install).kvCachePrecision
        #expect(kv.accepts(.fp16))
        #expect(kv.accepts(.int8))
        #expect(kv.accepts(.int4))
        #expect(
            kv.refusal(
                for: .int8, flag: "--kv-bits", plugin: Inkling2GGUFQ4Plugin.displayName,
                spelled: { "\($0.rawValue)" }) == nil)
        #expect(
            kv.refusal(
                for: .int4, flag: "--kv-bits", plugin: Inkling2GGUFQ4Plugin.displayName,
                spelled: { "\($0.rawValue)" }) == nil)
        let defaulted = kv.resolveDefault(.int8, flag: "--kv-bits", plugin: "Inkling")
        #expect(defaulted.value == .int8)
        #expect(defaulted.notice == nil)
    }

    @Test func kvRowBytesScaleWithPrecision() {
        #expect(InklingQ4LoadPlan.kvRowBytes(kvWidth: 512, precision: .fp16) == 1024)
        #expect(InklingQ4LoadPlan.kvRowBytes(kvWidth: 512, precision: .int8) == 576)
        #expect(InklingQ4LoadPlan.kvRowBytes(kvWidth: 512, precision: .int4) == 320)
    }

    @Test func declaresItsReasoningLevels() {
        let control = Inkling2GGUFQ4Plugin.reasoningControl(for: install)
        #expect(control?.supportedLevels == [.off, .low, .medium, .high, .xhigh])
        #expect(control?.levelWhenOn == .high)
    }

    @Test func supportsConcurrentSequences() {
        let params = Inkling2GGUFQ4Plugin.parameters(for: install)
        #expect(params.concurrentSequences.accepts(1))
        #expect(params.concurrentSequences.accepts(2))
        #expect(params.concurrentSequences.accepts(4))
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
