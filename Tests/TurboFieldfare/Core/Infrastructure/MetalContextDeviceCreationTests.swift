import Foundation
import Testing

@Suite struct MetalContextDeviceCreationTests {
    private static var sourcesDirectory: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources")
    }

    @Test func onlyMetalContextCreatesTheSystemDevice() throws {
        let sources = Self.sourcesDirectory
        try #require(FileManager.default.fileExists(atPath: sources.path),
                     "cannot locate Sources/ from \(#filePath)")

        var offenders: [String] = []
        let walker = FileManager.default.enumerator(
            at: sources,
            includingPropertiesForKeys: nil)

        while let url = walker?.nextObject() as? URL {
            guard url.pathExtension == "swift",
                  url.lastPathComponent != "MetalContext.swift",
                  let text = try? String(contentsOf: url, encoding: .utf8),
                  text.contains("MTLCreateSystemDefaultDevice(") else { continue }

            for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false)
                .enumerated() where line.contains("MTLCreateSystemDefaultDevice(") {
                offenders.append("\(url.lastPathComponent):\(index + 1)")
            }
        }

        #expect(offenders.isEmpty,
                "Production device creation bypasses MetalContext: \(offenders.joined(separator: ", "))")
    }
}
