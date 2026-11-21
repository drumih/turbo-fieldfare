import Foundation
import Testing

@testable import TinyTitan

@Suite struct Inkling2GGUFManifestTests {
    /// A directory or `.gguf` path of an Unsloth Inkling-2-Mini GGUF
    /// (`UD-Q2_K_XL` or `UD-Q4_K_XL`). The suite never downloads one.
    private static var modelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_GGUF"]
    }

    @Test func unrolledPositionsMapToStoredLayers() {
        // Inkling-2-Mini: 24 stored layers, core 6..<18 run three times = 48.
        let mapped = (0..<48).map {
            Inkling2GGUFManifest.storedLayer(forUnrolled: $0, loopStart: 6, loopEnd: 18, passes: 3)
        }
        #expect(Array(mapped[0..<6]) == Array(0..<6))
        #expect(Array(mapped[6..<18]) == Array(6..<18))
        #expect(Array(mapped[18..<30]) == Array(6..<18))
        #expect(Array(mapped[30..<42]) == Array(6..<18))
        #expect(Array(mapped[42..<48]) == Array(18..<24))
        // One pass is the identity.
        #expect(
            (0..<24).map {
                Inkling2GGUFManifest.storedLayer(
                    forUnrolled: $0, loopStart: 6, loopEnd: 18, passes: 1)
            } == Array(0..<24))
    }

    @Test(.enabled(if: modelPath != nil))
    func readsInkling2Manifest() throws {
        let path = try #require(Self.modelPath)
        let manifest = try Inkling2GGUFManifest(url: URL(fileURLWithPath: path))
        #expect(manifest.unrolledLayers == 48)
        #expect(manifest.storedLayers == 24)
        #expect(manifest.hidden == 2_304)
        #expect(manifest.headDimension == 128)
        #expect(manifest.expertCount == 128 && manifest.expertsPerToken == 8)
        #expect(manifest.loopStart == 6 && manifest.loopEnd == 18 && manifest.loopPasses == 3)
        #expect(manifest.vocabulary == 200_058 && manifest.paddedVocabulary == 201_024)
        // Layers 5, 11, 17 and 23 are the full-attention layers.
        #expect(manifest.layers.filter { !$0.isSliding }.map(\.index) == [5, 11, 17, 23])
        #expect(manifest.layers.allSatisfy { $0.heads == 24 && $0.kvHeads == 8 })
        #expect(manifest.layers.allSatisfy { $0.expertFeedForward == 1_152 })
        // The global NVFP4 scale survives requantization on layers 0..<20.
        #expect(manifest.layers.map(\.hasExpertScales) == (0..<24).map { $0 < 20 })
        let expertTypes = Set(manifest.layers.flatMap { [$0.gateUpType, $0.downType] })
        FileHandle.standardError.write(
            Data("Inkling2 GGUF expert types: \(expertTypes.map(\.name).sorted())\n".utf8))
        #expect(!expertTypes.isEmpty)
    }
}
