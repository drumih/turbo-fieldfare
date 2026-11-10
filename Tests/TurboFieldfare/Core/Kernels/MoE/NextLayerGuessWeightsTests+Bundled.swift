import Foundation
import Metal
import Testing
@testable import TurboFieldfare

extension NextLayerGuessWeightsTests {
    private static let gemma4Snapshot =
        "sha256:bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13"
    private static let gemma4GuessSHA256 =
        "b18242445534d452b7efb9c7d1c358884ef56637ad6003c187e5bced50dc3dcd"

    /// The runtime ships the fitted guess for gemma4, intact. Any other model
    /// gets none, and the early read falls back to the router.
    @Test func shipsFittedGuessForGemma4Only() throws {
        let device = try MetalContext().device
        func lookup(_ hash: String?) throws -> NextLayerGuessWeights? {
            try NextLayerGuessWeights.bundled(sourceSnapshotHash: hash, numLayers: 30,
                                              numExperts: 128, hiddenSize: 2816,
                                              device: device)
        }
        let gemma4 = try #require(try lookup(Self.gemma4Snapshot))
        #expect(gemma4.sha256 == Self.gemma4GuessSHA256)
        #expect(try lookup("sha256:" + String(repeating: "0", count: 64)) == nil)
        #expect(try lookup(nil) == nil)
    }
}
