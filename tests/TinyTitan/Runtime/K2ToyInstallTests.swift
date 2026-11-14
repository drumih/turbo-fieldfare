import Foundation
import Testing
import TinyTitanCore

@testable import TinyTitan

/// The K2 loader path end to end on a CI-safe synthetic install: arch
/// resolution, resident schema, manifest/layout validation, the mapped
/// resident buffer, and the runner's graph refusal.
@Suite("K2 toy install")
struct K2ToyInstallTests {
    @Test func loadsThroughTheSchemaAndStopsAtTheRunner() throws {
        let directory = try K2ToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        #expect(config.family == .k2Horizon)
        #expect(config.numLayers == 3)
        #expect(config.denseLayerIndices == [0, 1, 2])
        #expect(config.normGroups == 2)
        #expect(config.attentionGate == .softplus)

        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        #expect(model.config.family == .k2Horizon)
        #expect(try model.embedding().shape.0 == UInt32(config.vocabSize))
        #expect(try model.qProj(layer: 0).shape.1 == UInt32(config.hiddenSize))
        #expect(try model.inputNorm(layer: 2).shape.0 == UInt32(config.hiddenSize))

        var message = ""
        do {
            _ = try RealForwardRunner(
                model: model, context: ctx, maxContext: 4_096)
        } catch {
            message = String(describing: error)
        }
        #expect(message.contains("TinyTitanBench k2gpu"), "\(message)")
        #expect(message.contains("does not run on this runner"), "\(message)")
    }

    @Test func sparseToyValidatesTheMoEAndMoVAProteins() throws {
        let directory = try K2ToySynthetic.write(sparse: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try ArchConfig.resolved(
            forFamily: .k2Horizon, directoryURL: directory)
        #expect(config.numLayers == 4)
        #expect(config.denseLayerIndices == [0, 1, 2])
        #expect(config.mova.numExperts == 4)
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: directory, device: ctx.device, expecting: config,
            integrityPolicy: .fullSha256)
        #expect(model.config.numLayers == 4)
        #expect(try model.inputNorm(layer: 3).shape.0 == UInt32(config.hiddenSize))
        // The sparse layer's routed blob is laid out and readable.
        _ = try model.routedExpertResidency(layer: 3)
    }
}
