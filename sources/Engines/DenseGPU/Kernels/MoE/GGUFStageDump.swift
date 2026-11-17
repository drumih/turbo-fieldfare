import Foundation
import Metal

/// Opt-in capture of the as-is GGUF decoder's intermediate activations, for
/// parity work against a reference implementation (llama.cpp's `llama-debug`
/// with a tensor dump).
///
/// Off unless `TINYTITAN_GGUF_DUMP_DIR` names a directory. Each captured
/// position writes `<dir>/p<position>/<stage>.f32` - little-endian float32, the
/// fp16 activation widened - so a comparison script reads them with no header.
/// `TINYTITAN_GGUF_DUMP_POSITION` restricts the capture to one position; without
/// it every position is captured.
///
/// A captured position commits and waits at every stage boundary so the buffers
/// it reads hold that stage's output and not a later one's: the decoder's normal
/// path batches a layer into two command buffers, and a host read between
/// encode and commit sees whatever the buffer held before. Capture is therefore
/// slow and changes no numerics - the same kernels run in the same order.
package struct GGUFStageDump: Sendable {
    package let directory: URL
    package let position: Int?

    package init(directory: URL, position: Int?) {
        self.directory = directory
        self.position = position
    }

    /// The capture the environment asks for, or nil when it asks for none.
    package static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> GGUFStageDump? {
        guard let path = environment["TINYTITAN_GGUF_DUMP_DIR"], !path.isEmpty else {
            return nil
        }
        let position = environment["TINYTITAN_GGUF_DUMP_POSITION"].flatMap { Int($0) }
        return GGUFStageDump(directory: URL(fileURLWithPath: path), position: position)
    }

    /// The writer for `position`, or nil when this position is not captured.
    package func capture(position: Int) throws -> Capture? {
        if let wanted = self.position, wanted != position { return nil }
        let folder = directory.appendingPathComponent(
            String(format: "p%05d", position), isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return Capture(folder: folder)
    }

    package struct Capture: Sendable {
        package let folder: URL

        /// Writes `count` fp16 values from the start of `buffer` as float32.
        /// The caller has waited on every command buffer that writes it.
        package func write(_ stage: String, half buffer: MTLBuffer, count: Int) throws {
            let pointer = buffer.contents().assumingMemoryBound(to: UInt16.self)
            let values = (0..<count).map { Float(Float16(bitPattern: pointer[$0])) }
            try write(stage, values: values)
        }

        package func write(_ stage: String, values: [Float]) throws {
            let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
            try data.write(to: folder.appendingPathComponent(stage + ".f32"), options: .atomic)
        }
    }
}
