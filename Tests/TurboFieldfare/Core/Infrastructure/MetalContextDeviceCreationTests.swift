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

    private static let sanctionedRelativePath =
        "TurboFieldfare/Infrastructure/Metal/MetalContext.swift"

    @Test func onlyMetalContextCreatesTheSystemDevice() throws {
        let sources = Self.sourcesDirectory
        try #require(FileManager.default.fileExists(atPath: sources.path),
                     "cannot locate Sources/ from \(#filePath)")

        var enumerationFailures: [String] = []
        var readFailures: [String] = []
        var offenders: [String] = []
        let walker = try #require(FileManager.default.enumerator(
            at: sources,
            includingPropertiesForKeys: nil,
            errorHandler: { url, error in
                enumerationFailures.append("\(url.path): \(error)")
                return false
            }))
        let sourcePrefix = sources.standardizedFileURL.path + "/"

        while let url = walker.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            let path = url.standardizedFileURL.path
            guard path.hasPrefix(sourcePrefix) else {
                readFailures.append("cannot make \(path) relative to \(sources.path)")
                continue
            }
            let relativePath = String(path.dropFirst(sourcePrefix.count))
            guard relativePath != Self.sanctionedRelativePath else { continue }
            let text: String
            do {
                text = try String(contentsOf: url, encoding: .utf8)
            } catch {
                readFailures.append("\(relativePath): \(error)")
                continue
            }
            guard text.contains("MTLCreateSystemDefaultDevice(") else { continue }

            for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false)
                .enumerated() where line.contains("MTLCreateSystemDefaultDevice(") {
                offenders.append("\(relativePath):\(index + 1)")
            }
        }

        #expect(enumerationFailures.isEmpty,
                "source enumeration failed: \(enumerationFailures.joined(separator: "; "))")
        #expect(readFailures.isEmpty,
                "source reads failed: \(readFailures.joined(separator: "; "))")
        #expect(offenders.isEmpty,
                "Production device creation bypasses MetalContext: \(offenders.joined(separator: ", "))")
    }
}
