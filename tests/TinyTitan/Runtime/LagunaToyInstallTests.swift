import Foundation
import Testing

@testable import TinyTitan
@testable import TinyTitanCore

/// The Laguna loader path end to end on a CI-safe synthetic install: the
/// checked geometry, the resident schema with the per-layer head widths, the
/// dense and sparse tensor roles, the packed experts, the probe's first two
/// tokens, and the runner's refusal.
@Suite("Laguna toy install")
struct LagunaToyInstallTests {
    @Test func loadsTheResidentSchemaAndTheSparseLayout() throws {
        let directory = try LagunaToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let toy = LagunaToySynthetic.toy
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: toy,
            streamingMode: .pread(slotCount: 4), integrityPolicy: .fullSha256)
        #expect(model.config.modelFamily == .laguna)
        #expect(model.config.perLayerHeadGeometry.headsPerLayer == [2, 4])
        #expect(try model.embedding().shape.0 == UInt32(toy.vocabSize))
        // The per-layer head counts, not a single scalar: 2 then 4.
        #expect(try model.qProj(layer: 0).shape.0 == UInt32(2 * toy.fullHeadDim))
        #expect(try model.qProj(layer: 1).shape.0 == UInt32(4 * toy.fullHeadDim))
        #expect(try model.lagunaAttentionGate(layer: 1).shape.0 == 4)
        #expect(try model.qNorm(layer: 1).shape.0 == UInt32(toy.fullHeadDim))
        #expect(try model.lagunaRouterBias(layer: 1).shape.0 == UInt32(toy.numExperts))
        #expect(try model.lagunaSharedGate(layer: 1).shape.0 == UInt32(toy.moeIntermediateSize))
        _ = try model.routedExpertResidency(layer: 1)

        var message = ""
        do {
            _ = try RealForwardRunner(
                model: model, context: ctx, maxContext: 128)
        } catch {
            message = String(describing: error)
        }
        #expect(message.contains("does not run on this runner"), "\(message)")
    }

    @Test func theProbeRunsBothLayerKinds() async throws {
        let directory = try LagunaToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let toy = LagunaToySynthetic.toy
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: toy,
            streamingMode: .pread(slotCount: 4), integrityPolicy: .fullSha256)
        let graph = try LagunaGraphProbe(
            model: model, context: ctx, maxSequence: 4)
        let first = try await graph.forward(tokenID: 7, position: 0)
        #expect(first.count == toy.vocabSize)
        let firstFinite = first.allSatisfy { $0.isFinite }
        #expect(firstFinite)
        // The second token reads the first position's cache through both a
        // full layer and a sliding-window layer.
        let second = try await graph.forward(tokenID: 3, position: 1)
        #expect(second.count == toy.vocabSize)
        let secondFinite = second.allSatisfy { $0.isFinite }
        #expect(secondFinite)
        // The fixture's weights are constant bytes, so the logits cannot
        // depend on the token; what this run proves is that both layer kinds
        // and the second position's cache read execute without trapping.
        #expect(second.count == first.count)
    }
}
