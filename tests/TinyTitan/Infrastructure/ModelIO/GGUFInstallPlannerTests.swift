import Foundation
import TinyTitanFormat
import Testing

@testable import TinyTitan

@Suite struct GGUFInstallPlannerTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func plansEveryLayersExpertsWithinTheReadersBound() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let plan = try GGUFInstallPlanner.plan(from: model)

        #expect(plan.layout.numLayers == 48)
        #expect(plan.layout.expertsPerLayer == 512)
        #expect(plan.layout.layers.count == 48)
        #expect(plan.layout.layers.allSatisfy { $0.experts.count == 512 })
        #expect(plan.expertStride == 2_768_896)
        #expect(plan.expertBytes == 68_048_388_096)

        // The reader caps layout.json at 128 MiB; a plan that exceeded it could
        // never be loaded, and this is where that is caught.
        #expect(plan.layoutJSONBytes > 0)
        #expect(UInt64(plan.layoutJSONBytes) < SSDAIFormatV1.packedExpertsLayoutMaxBytes)
        print(
            "GGUF install plan: layout.json \(plan.layoutJSONBytes) bytes, "
                + "expert payload \(plan.expertBytes) bytes")

        // Every expert's records are the same shape at a different offset, and
        // the last expert ends exactly at the stride boundary.
        let layer = try #require(plan.layout.layers.last)
        let first = try #require(layer.experts.first)
        let last = try #require(layer.experts.last)
        #expect(last.offset == 511 * plan.expertStride)
        #expect(first.tensors == last.tensors)
        let highest = first.tensors.values.map { $0.offset + $0.size }.max() ?? 0
        #expect(highest <= plan.expertStride)
    }
}
