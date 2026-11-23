import Foundation
import Testing
import TinyTitanFormat

@testable import TinyTitanRepackCore

/// The Inkling 2 GGUF install plan (TT-121 phase 1), checked against the real
/// Q4_K_XL file's header: no bytes are copied, so it runs in a second, and it
/// pins the arithmetic that decides where every byte of the install comes from.
@Suite struct Inkling2GGUFPlannerTests {
    private static var gguf: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_GGUF"]
    }

    static func source() throws -> Inkling2GGUFPlanner.Source {
        let path = try #require(gguf)
        let handle = try #require(FileHandle(forReadingAtPath: path))
        defer { try? handle.close() }
        let header = try GGUFHeaderV3(
            bytes: [UInt8](try handle.read(upToCount: 64 << 20) ?? Data()))
        let size = try #require(
            (try FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?
                .uint64Value)
        return .init(
            fileName: (path as NSString).lastPathComponent, header: header, fileSize: size)
    }

    /// Every tensor's computed size ends before the next tensor starts, short of
    /// it by less than the file's alignment: the size arithmetic agrees with
    /// how the file is actually laid out, tensor by tensor.
    @Test(.enabled(if: gguf != nil)) func computedSizesTileTheFile() throws {
        let source = try Self.source()
        let tensors = source.header.tensors.sorted { $0.absoluteOffset < $1.absoluteOffset }
        for (index, tensor) in tensors.enumerated() {
            let kept = try Inkling2GGUFPlanner.kept(ggmlType: tensor.ggmlType, tensor: tensor.name)
            let size = try Inkling2GGUFPlanner.bytes(
                of: tensor.dimensions, kept: kept, tensor: tensor.name)
            let end =
                index + 1 < tensors.count ? tensors[index + 1].absoluteOffset : source.fileSize
            #expect(tensor.absoluteOffset + size <= end, "\(tensor.name) overruns")
            #expect(
                end - (tensor.absoluteOffset + size) < source.header.alignment, "\(tensor.name) gap"
            )
        }
    }

    @Test(.enabled(if: gguf != nil)) func thePlanMatchesTheModel() throws {
        let plan = try Inkling2GGUFPlanner.plan(
            source: try Self.source(), outputDir: "/tmp/inkling-plan")
        #expect(plan.arch.numLayers == 24)
        #expect(plan.arch.numExperts == 128)
        #expect(plan.arch.topKExperts == 8)
        // The checkpoint's own `local_layer_ids`: every layer but 5, 11, 17, 23.
        let local = try #require(plan.arch.variantGeometry?.lists["localLayers"])
        #expect(local == (0..<24).filter { ![5, 11, 17, 23].contains($0) })
        #expect(plan.arch.variantGeometry?.scalars["loopPasses"] == 3)
        #expect(plan.resident.entries.count == 734 - 88)
        // NVFP4 layers: 2,985,984 + 1,492,992 + two 4-byte scales, page-rounded.
        let nvfp4Stride = Inkling2GGUFPlanner.roundUp(2_985_984 + 1_492_992 + 64 + 4, to: 16_384)
        #expect(plan.layers[0].expertStride == nvfp4Stride)
        #expect(plan.layers[20].expertStride > nvfp4Stride)
        #expect(plan.manifestExpertStride == plan.layers[20].expertStride)
        #expect(plan.layers[0].subTensors.first?.bitsForWeights == 4)
        #expect(plan.layers[20].subTensors.first?.bitsForWeights == 8)
        #expect(plan.passthroughFiles.first?.size == source0DataStart())
    }

    @Test(.enabled(if: gguf != nil)) func theCopyPlanAndLayoutAreValid() throws {
        let plan = try Inkling2GGUFPlanner.plan(
            source: try Self.source(), outputDir: "/tmp/inkling-plan")
        let copies = try RangeCopyPlanner.plan(repackPlan: plan, rangeChunkBytes: 64 << 20)
        // Every tensor byte is copied once, plus the header sidecar.
        let tensorBytes =
            plan.resident.entries.reduce(UInt64(0)) { $0 + $1.sizeBytes }
            + plan.layers.reduce(UInt64(0)) { total, layer in
                total + layer.subTensors.reduce(0) { $0 + $1.sizeInExpertBlob }
                    * UInt64(layer.expertsPerLayer)
            }
        let copied = copies.scalarCopies.reduce(UInt64(0)) { $0 + $1.size }
        #expect(copied == tensorBytes + (plan.passthroughFiles.first?.size ?? 0))
        let layout = try SSDAIJSON.encodeLayout(plan: plan, expertStride: plan.manifestExpertStride)
        _ = try SSDAIPackedExpertsLayoutCodec.decode(layout)
    }

    private func source0DataStart() -> UInt64 {
        (try? Self.source().header.dataStart) ?? 0
    }
}
