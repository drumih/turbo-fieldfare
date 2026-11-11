import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The opt-in parity capture: off unless asked for, one position or all, and
/// files a comparison script reads with no header (float32 little-endian).
@Suite struct GGUFStageDumpTests {
    @Test func isOffWithoutTheDirectory() {
        #expect(GGUFStageDump.fromEnvironment([:]) == nil)
        #expect(GGUFStageDump.fromEnvironment(["TINYTITAN_GGUF_DUMP_DIR": ""]) == nil)
    }

    @Test func readsTheDirectoryAndOptionalPosition() {
        let every = GGUFStageDump.fromEnvironment(["TINYTITAN_GGUF_DUMP_DIR": "/x"])
        #expect(every?.directory.path == "/x")
        #expect(every?.position == nil)
        let one = GGUFStageDump.fromEnvironment([
            "TINYTITAN_GGUF_DUMP_DIR": "/x", "TINYTITAN_GGUF_DUMP_POSITION": "4",
        ])
        #expect(one?.position == 4)
    }

    @Test func capturesOnlyTheRequestedPosition() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "gguf-stage-dump-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let dump = GGUFStageDump(directory: directory, position: 4)
        #expect(try dump.capture(position: 3) == nil)
        let capture = try #require(try dump.capture(position: 4))
        #expect(capture.folder.lastPathComponent == "p00004")
    }

    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil))
    func writesHalvesWidenedToFloat32() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let values: [Float16] = [1, -2.5, 0.125, 65_504]
        let bits = values.map(\.bitPattern)
        let buffer = try #require(
            device.makeBuffer(bytes: bits, length: bits.count * 2, options: .storageModeShared))
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "gguf-stage-dump-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let capture = try #require(
            try GGUFStageDump(directory: directory, position: nil).capture(position: 0))
        try capture.write("l00.attn_out", half: buffer, count: values.count)

        let file = capture.folder.appendingPathComponent("l00.attn_out.f32")
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        let data = try #require(try handle.read(upToCount: 64))
        let floats = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        #expect(floats == values.map(Float.init))
    }
}
