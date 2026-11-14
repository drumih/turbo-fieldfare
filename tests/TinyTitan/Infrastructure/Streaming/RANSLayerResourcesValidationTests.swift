import Foundation
import Metal
import Testing
import TinyTitanFormat
import TinyTitanCore
@testable import TinyTitan
@testable import TinyTitanRepackCore

/// The rANS layer-opening gates on untrusted payload bytes.
///
/// The stream plans are built from the payload itself — a u16 length table per
/// block — at layer open. A length-table read that fails, or stream lengths
/// that walk past the expert's validated run, must refuse the layer instead of
/// producing an empty plan (which silently skips the expert's decode and leaves
/// stale weights in the slot, and understates the staging stride the wave is
/// read into).
@Suite("rANS layer resource validation")
struct RANSLayerResourcesValidationTests {
    private static let stride = 4096
    private static let symbolCount: UInt32 = 4096
    private static var streams: Int {
        ExpertRANSIndex.streams(symbolCount: Int(symbolCount))
    }

    /// A payload file for one expert and one block: u16 lengths for `streams`
    /// streams, then each stream's 4-byte state header and bytes.
    private static func payloadFile(lengths: [Int]) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rans-plan-\(UUID().uuidString).bin")
        var data = Data()
        for length in lengths {
            var value = UInt16(length).littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        for length in lengths {
            data.append(contentsOf: [UInt8](repeating: 0x40, count: length))
        }
        try data.write(to: url)
        return url
    }

    private static func layout(runLength: UInt32) throws -> ExpertRANSIndex.Layout {
        let frequencies = [UInt16](repeating: 16_384 / 256, count: 256)
        let block = ExpertRANSIndex.Block(
            payloadOffset: 0, payloadLength: runLength,
            rawOffset: 0, symbolCount: symbolCount, stride: 1, role: 0)
        return ExpertRANSIndex.Layout(
            scaleBits: ExpertRANSIndex.implementedScaleBits,
            models: [ExpertRANSIndex.Model(frequencies: frequencies)],
            blocks: [[block]])
    }

    @Test("Stream lengths that exceed the expert's run refuse the layer")
    func refusesLengthsPastTheRun() throws {
        // The run holds the length table plus one stream of 16 bytes; the
        // lengths claim two streams of 16, so the second walks past the run.
        let runLength = UInt32(Self.streams * 2 + 16)
        let url = try Self.payloadFile(lengths: [16, 16])
        defer { try? FileManager.default.removeItem(at: url) }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw StreamingMetalUnavailableError()
        }
        #expect(throws: ModelError.self) {
            _ = try RANSLayerResources(
                layout: try Self.layout(runLength: runLength),
                phases: [0],
                device: device,
                payloadPath: url.path)
        }
    }

    @Test func exactStreamLengthsIncludeTheirHeaders() throws {
        let url = try Self.payloadFile(lengths: [16, 16])
        defer { try? FileManager.default.removeItem(at: url) }
        let resources = try RANSLayerResources(
            layout: try Self.layout(runLength: 36), phases: [0],
            device: try MetalContext().device, payloadPath: url.path)
        #expect(resources.plans[0].map(\.payloadOffset) == [4, 20])
        #expect(resources.payloadStride == 36)
        #expect(resources.payloadRange(expert: 0)?.count == 36)
    }

    @Test func aStreamShorterThanItsHeaderIsRefused() throws {
        let url = try Self.payloadFile(lengths: [7, 25])
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: ModelError.self) {
            _ = try RANSLayerResources(
                layout: try Self.layout(runLength: 36), phases: [0],
                device: try MetalContext().device, payloadPath: url.path)
        }
    }

    @Test func aStreamCannotConsumeTheNextBlocksBytes() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rans-block-boundary-\(UUID().uuidString).bin")
        var data = Data([18, 0])
        data.append(Data(repeating: 0x40, count: 8))
        data.append(contentsOf: [32, 0])
        data.append(Data(repeating: 0x40, count: 32))
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let model = ExpertRANSIndex.Model(
            frequencies: [UInt16](repeating: 64, count: 256))
        let layout = ExpertRANSIndex.Layout(
            scaleBits: 14, models: [model, model],
            blocks: [
                [
                    .init(
                        payloadOffset: 0, payloadLength: 10, rawOffset: 0,
                        symbolCount: 64, stride: 1, role: 0),
                    .init(
                        payloadOffset: 10, payloadLength: 34, rawOffset: 64,
                        symbolCount: 64, stride: 1, role: 1),
                ]
            ])
        #expect(throws: ModelError.self) {
            _ = try RANSLayerResources(
                layout: layout, phases: [0, 1],
                device: try MetalContext().device, payloadPath: url.path)
        }
    }

    private static func encodedLayer(
        in directory: URL, bytes: Data, device: MTLDevice, twoPhase: Bool = false
    ) throws -> (streamer: PreadExpertStreamer, payload: URL) {
        let raw = directory.appendingPathComponent("raw.bin")
        let payload = directory.appendingPathComponent("payload.bin")
        let index = directory.appendingPathComponent("payload.ridx")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try bytes.write(to: raw)
        let experts = bytes.count / Self.stride
        let blocks: [RANSBlockSpec] =
            twoPhase
            ? [
                .init(rawOffset: 0, symbolCount: UInt32(Self.stride / 2), stride: 1),
                .init(
                    rawOffset: UInt32(Self.stride / 2), symbolCount: UInt32(Self.stride / 2),
                    stride: 1),
            ]
            : [.init(rawOffset: 0, symbolCount: UInt32(Self.stride), stride: 1)]
        let result = try RANSExpertLayerWriter.write(
            sourcePath: raw.path, destinationPath: payload.path, indexPath: index.path,
            geometry: .init(
                expertStride: UInt64(Self.stride), expertsPerLayer: experts, blocks: blocks))
        let layout = try ExpertRANSIndex.decode(
            Data(contentsOf: index), expectedExperts: experts,
            expertStride: UInt64(Self.stride), payloadSize: result.compressedBytes)
        let streamer = try PreadExpertStreamer(
            layout: .init(
                path: payload.path, streamOffset: 0, streamSize: result.compressedBytes,
                expertsPerLayer: experts, expertStride: UInt64(Self.stride),
                rans: layout, ransPhases: twoPhase ? [0, 1] : [0]),
            device: device, slotCount: experts)
        return (streamer, payload)
    }

    @Test func firstStagePublishesOnlyAfterEveryWavesFrontIsReady() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rans-waves-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let bytes = Data((0..<(Self.stride * 10)).map { UInt8(1 + ($0 / 7) % 5) })
        let (streamer, _) = try Self.encodedLayer(
            in: directory, bytes: bytes, device: try MetalContext().device, twoPhase: true)
        let plan = try streamer.planExpertsCached(experts: Array(0..<10))
        let resources = try #require(streamer.ransResources)
        try #require(plan.misses.count > resources.maximumWave)
        for slot in plan.assignedSlots {
            memset(streamer.slotPointers[slot], 0, Self.stride)
        }
        var publications = 0
        let results = try streamer.executeExpertCachePlan(plan) {
            publications += 1
            for (expert, slot) in plan.assignedSlots.enumerated() {
                let front = Data(bytes: streamer.slotPointers[slot], count: Self.stride / 2)
                let start = expert * Self.stride
                #expect(front == bytes.subdata(in: start..<(start + Self.stride / 2)))
            }
        }
        #expect(publications == 1)
        for (expert, result) in results.enumerated() {
            let restored = Data(
                bytes: result.buffer.contents().advanced(by: Int(result.offset)), count: Self.stride
            )
            #expect(
                restored == bytes.subdata(in: expert * Self.stride..<(expert + 1) * Self.stride))
        }
    }

    @Test func aLaterWaveFailureDoesNotPublishAnIncompletePlan() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rans-wave-failure-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let (streamer, payload) = try Self.encodedLayer(
            in: directory, bytes: Data(repeating: 0, count: Self.stride * 10),
            device: try MetalContext().device, twoPhase: true)
        let resources = try #require(streamer.ransResources)
        let range = try #require(resources.payloadRange(expert: resources.maximumWave))
        let stream = try #require(resources.plans[resources.maximumWave].first { $0.phase == 0 })
        let offset = Int(range.offset) + Int(stream.payloadOffset)
        var corrupt = try Data(contentsOf: payload)
        corrupt.replaceSubrange(offset..<(offset + 4), with: [0, 0, 0, 0])
        try corrupt.write(to: payload)
        let plan = try streamer.planExpertsCached(experts: Array(0..<10))
        var published = false
        #expect(throws: ModelError.self) {
            _ = try streamer.executeExpertCachePlan(plan) { published = true }
        }
        #expect(!published)
        #expect(try streamer.planExpertsCached(experts: Array(0..<10)).hits == 0)
    }

    @Test func writerPayloadRoundTripsThroughTheGPU() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rans-gpu-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let bytes = Data((0..<(Self.stride * 2)).map { UInt8(($0 / 7) % 5) })
        let (streamer, _) = try Self.encodedLayer(
            in: directory, bytes: bytes, device: try MetalContext().device)
        let results = try streamer.loadExpertsCached(experts: [0, 1])
        #expect(results.count == 2)
        for (expert, result) in results.enumerated() {
            let restored = Data(
                bytes: result.buffer.contents().advanced(by: Int(result.offset)), count: Self.stride
            )
            #expect(
                restored == bytes.subdata(in: expert * Self.stride..<(expert + 1) * Self.stride))
        }
    }

    @Test func exhaustedStreamFailsBeforePublishingAndTheNextDecodeCanSucceed() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rans-gpu-corrupt-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let device = try MetalContext().device
        let bytes = Data(repeating: 0, count: Self.stride)
        let (streamer, payload) = try Self.encodedLayer(in: directory, bytes: bytes, device: device)
        let resources = try #require(streamer.ransResources)
        let original = try Data(contentsOf: payload)
        var corrupt = original
        corrupt.replaceSubrange(4..<8, with: [0, 0, 0, 0])
        try corrupt.write(to: payload)
        let output = try #require(
            device.makeBuffer(length: Self.stride, options: .storageModeShared))
        var published = false
        #expect(throws: ModelError.self) {
            try streamer.decodeRANSInto(
                experts: [0], outputs: [output], offsets: [0], resources: resources,
                firstStage: { published = true })
        }
        #expect(!published)
        try original.write(to: payload)
        try streamer.decodeRANSInto(
            experts: [0], outputs: [output], offsets: [0], resources: resources,
            firstStage: { published = true })
        #expect(published)
        #expect(Data(bytes: output.contents(), count: Self.stride) == bytes)
    }

    private static func modelLayer(
        in directory: URL
    ) throws -> (
        packed: TinyTitan.PackedExpertsLayout, payload: URL, index: URL,
        rans: ExpertRANSIndex.Layout
    ) {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let stride = Int(SSDAIFormatV1.alignmentBytes)
        let tensors: [String: SSDAISubTensorV1] = [
            "gate": .init(offset: 0, size: 16, dtype: "U32", shape: [4], bits: 4),
            "down": .init(offset: 16, size: 16, dtype: "U32", shape: [4], bits: 4),
            "gate_scales": .init(offset: 32, size: 16, dtype: "BF16", shape: [8], bits: nil),
        ]
        let layer = SSDAILayerV1(
            layer: 0, file: "layer_00.bin",
            experts: (0..<2).map {
                .init(
                    expert: $0, physicalRank: $0, offset: UInt64($0 * stride),
                    size: UInt64(stride), tensors: tensors)
            })
        let wire = SSDAIPackedExpertsLayoutV1(
            expertStride: UInt64(stride), numLayers: 1, expertsPerLayer: 2,
            compression: .rans, layers: [layer])
        let packed = try PackedExpertsLayoutReader.decode(
            data: JSONEncoder().encode(wire), manifest: nil)
        let raw = directory.appendingPathComponent("raw.bin")
        let payload = directory.appendingPathComponent(layer.file)
        let index = directory.appendingPathComponent("layer_00.ridx")
        try Data((0..<(stride * 2)).map { UInt8($0 % 5) }).write(to: raw)
        let result = try RANSExpertLayerWriter.write(
            sourcePath: raw.path, destinationPath: payload.path, indexPath: index.path,
            geometry: .init(
                expertStride: UInt64(stride), expertsPerLayer: 2,
                blocks: RANSExpertLayerGeometry.blocks(for: layer)))
        let rans = try ExpertRANSIndex.decode(
            Data(contentsOf: index), expectedExperts: 2,
            expertStride: UInt64(stride), payloadSize: result.compressedBytes)
        return (packed, payload, index, rans)
    }

    @Test func modelLoaderAcceptsWriterGeometryIncludingBF16Planes() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rans-model-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try Self.modelLayer(in: directory)
        let (rans, phases) = try Model.ransLayoutAndPhases(
            forLayer: 0, layout: fixture.packed, url: fixture.payload)
        #expect(rans == fixture.rans)
        #expect(phases == [1, 0, 0, 0])
    }

    @Test func modelLoaderAcceptsImplicitWidthWithIdenticalGeometry() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rans-implicit-width-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try Self.modelLayer(in: directory)
        let layer = fixture.packed.layers[0]
        let expert = layer.experts[1]
        var tensors = expert.subTensors
        let gate = try #require(tensors["gate"])
        tensors["gate"] = SubTensorEntry(
            offset: gate.offset, size: gate.size, dtype: gate.dtype, shape: gate.shape, bits: nil)
        let implicitExpert = ExpertEntry(
            expert: expert.expert, physicalRank: expert.physicalRank,
            offset: expert.offset, size: expert.size, subTensors: tensors)
        let layout = PackedExpertsLayout(
            expertStride: fixture.packed.expertStride, numLayers: 1, expertsPerLayer: 2,
            compression: .rans,
            layers: [
                LayerLayout(
                    layer: 0, file: layer.file, experts: [layer.experts[0], implicitExpert])
            ])
        let (rans, phases) = try Model.ransLayoutAndPhases(
            forLayer: 0, layout: layout, url: fixture.payload)
        #expect(rans == fixture.rans)
        #expect(phases == [1, 0, 0, 0])
    }

    @Test(arguments: [0, 1], ["offset", "count", "stride", "role"])
    func modelLoaderRejectsSidecarGeometryMismatch(expert: Int, field: String) throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rans-model-mismatch-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try Self.modelLayer(in: directory)
        var blocks = fixture.rans.blocks
        let original = blocks[expert][0]
        blocks[expert][0] = .init(
            payloadOffset: original.payloadOffset, payloadLength: original.payloadLength,
            rawOffset: field == "offset" ? 0 : original.rawOffset,
            symbolCount: field == "count" ? original.symbolCount - 1 : original.symbolCount,
            stride: field == "stride" ? 2 : original.stride,
            role: field == "role" ? 1 : original.role)
        try ExpertRANSIndex.encode(
            .init(scaleBits: fixture.rans.scaleBits, models: fixture.rans.models, blocks: blocks)
        ).write(to: fixture.index)
        #expect(throws: ModelError.self) {
            _ = try Model.ransLayoutAndPhases(
                forLayer: 0, layout: fixture.packed, url: fixture.payload)
        }
    }

    @Test func modelLoaderRejectsConsistentlySwappedLaterExpertGeometry() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("rans-noncanonical-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let fixture = try Self.modelLayer(in: directory)
        let originalLayer = fixture.packed.layers[0]
        let originalExpert = originalLayer.experts[1]
        var tensors = originalExpert.subTensors
        let gate = try #require(tensors["gate"])
        let down = try #require(tensors["down"])
        tensors["gate"] = down
        tensors["down"] = gate
        let changedExpert = ExpertEntry(
            expert: originalExpert.expert, physicalRank: originalExpert.physicalRank,
            offset: originalExpert.offset, size: originalExpert.size, subTensors: tensors)
        let changedLayout = PackedExpertsLayout(
            expertStride: fixture.packed.expertStride, numLayers: 1, expertsPerLayer: 2,
            compression: .rans,
            layers: [
                LayerLayout(
                    layer: 0, file: originalLayer.file,
                    experts: [originalLayer.experts[0], changedExpert])
            ])
        var blocks = fixture.rans.blocks
        for index in 0..<2 {
            let block = blocks[1][index]
            blocks[1][index] = .init(
                payloadOffset: block.payloadOffset, payloadLength: block.payloadLength,
                rawOffset: UInt32(index == 0 ? gate.offset : down.offset),
                symbolCount: block.symbolCount, stride: block.stride, role: block.role)
        }
        try ExpertRANSIndex.encode(
            .init(scaleBits: fixture.rans.scaleBits, models: fixture.rans.models, blocks: blocks)
        ).write(to: fixture.index)
        #expect(throws: ModelError.self) {
            _ = try Model.ransLayoutAndPhases(
                forLayer: 0, layout: changedLayout, url: fixture.payload)
        }
    }

    @Test("A payload shorter than the declared run refuses the layer")
    func refusesTruncatedPayload() throws {
        // The block declares a 64-byte run, but the file holds a single byte,
        // so even the u16 length-table pread fails: the layer must be refused
        // rather than opened with an empty plan for the expert.
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rans-plan-truncated-\(UUID().uuidString).bin")
        try Data([0x00]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw StreamingMetalUnavailableError()
        }
        #expect(throws: ModelError.self) {
            _ = try RANSLayerResources(
                layout: try Self.layout(runLength: 64),
                phases: [0],
                device: device,
                payloadPath: url.path)
        }
    }
}

private struct StreamingMetalUnavailableError: Error {}
