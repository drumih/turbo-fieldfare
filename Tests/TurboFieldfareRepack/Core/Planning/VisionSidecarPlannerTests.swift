import Foundation
import Testing

@testable import TurboFieldfareRepackCore

@Suite
struct VisionSidecarPlannerTests {
    @Test func plansOnlyVisionEntriesInResidentIndexFormat() throws {
        let snapshotDirectory = temporaryDirectory("planner")
        let outputDirectory = temporaryDirectory("output")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDirectory)
            try? FileManager.default.removeItem(atPath: outputDirectory)
        }
        let snapshot = try SyntheticSnapshot.build(at: snapshotDirectory)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDirectory)
        let header = try loadHeader(path: snapshot.shardPath)

        let plan = try VisionSidecarPlanner.plan(
            meta: metadata,
            shardHeaders: [header],
            outputDirectory: outputDirectory)

        #expect(plan.entryCount == 3)
        #expect(plan.sourceTensorCount == 7)
        #expect(plan.resident.entries.map(\.name) == [
            "embed_vision.embedding_projection.weight",
            "vision_tower.encoder.layers.0.input_layernorm.weight",
            "vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
        ])
        #expect(plan.resident.entries.allSatisfy {
            VisionSidecarPlanner.isVisionTensor($0.name)
        })
        #expect(plan.resident.entries.first?.sourceScales != nil)
        #expect(plan.resident.entries.first?.sourceBiases != nil)

        let ranges = try RangeCopyPlanner.planVisionSidecar(
            visionPlan: plan,
            rangeChunkBytes: 512)
        #expect(ranges.expectedOutputs == [
            RemoteExpectedOutput(
                relativePath: "weights.bin",
                size: plan.resident.totalSize)
        ])
        #expect(ranges.scalarCopies.count == plan.sourceTensorCount)
        #expect(ranges.scalarCopies.allSatisfy {
            $0.destinationPath == plan.resident.path
        })
    }

    private func loadHeader(path: String) throws -> Safetensors.Header {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let headerSize = data.prefix(8).enumerated().reduce(UInt64(0)) {
            $0 | (UInt64($1.element) << UInt64($1.offset * 8))
        }
        let header = Data(data[8..<(8 + Int(headerSize))])
        return try Safetensors.parseHeaderBytes(
            path: (path as NSString).lastPathComponent,
            fileSize: UInt64(data.count),
            headerBytes: header)
    }

    private func temporaryDirectory(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString).appendingPathComponent(
            "turbofieldfare-vision-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }
}
