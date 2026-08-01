import Foundation
import Testing

@testable import TurboFieldfare

@Suite struct ManifestArchListTests {
    @Test func supportedListContainsExactlyGemma4() {
        #expect(ArchConfig.supported.count == 1)
        #expect(ArchConfig.supported.first?.numLayers == ArchConfig.gemma4_26B_A4B.numLayers)
        #expect(ArchConfig.supported.first?.hiddenSize == ArchConfig.gemma4_26B_A4B.hiddenSize)
    }

    @Test func matchesTheGemmaArchitecture() {
        let manifestArch = ManifestArch(from: ArchConfig.gemma4_26B_A4B)
        #expect(ManifestReader.matchArch(manifestArch, against: ArchConfig.supported) != nil)
    }

    @Test func returnsNilForAnUnknownArchitecture() {
        var manifestArch = ManifestArch(from: ArchConfig.gemma4_26B_A4B)
        manifestArch.hiddenSize = 2048
        #expect(ManifestReader.matchArch(manifestArch, against: ArchConfig.supported) == nil)
    }
}
