import Foundation
import Metal
import Testing

@testable import TinyTitan

/// The whole resident set as one bf16 arena: every non-expert, non-PLE tensor
/// from the real checkpoint. This is what a GGUF `Model` would present to the
/// runner, so its size and load time decide whether that design fits in RAM.
@Suite struct GGUFResidentArenaFullTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func buildsTheWholeResidentSet() throws {
        let path = try #require(Self.realModelPath)
        let source = try GGUFModelSource.open(directoryURL: URL(fileURLWithPath: path))
        let context = try MetalContext()
        let start = DispatchTime.now().uptimeNanoseconds
        let arena = try GGUFResidentArena.build(
            names: source.residentNames, from: source.model, context: context)
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9

        #expect(arena.entries.count == source.residentNames.count)
        #expect(arena.byteCount % GGUFResidentArena.alignment == 0)
        // The index must be usable by name, at aligned offsets.
        for name in source.residentNames.prefix(64) {
            let entry = try #require(arena.entries[name])
            #expect(entry.offset % GGUFResidentArena.alignment == 0)
            #expect(entry.byteCount > 0)
        }
        print(
            "resident arena: \(arena.entries.count) tensors, "
                + "\(String(format: "%.2f", Double(arena.byteCount) / 1e9)) GB, "
                + "built in \(String(format: "%.1f", seconds)) s")
    }
}
