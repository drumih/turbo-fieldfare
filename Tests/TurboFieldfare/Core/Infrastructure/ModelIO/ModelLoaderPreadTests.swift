import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// `Model.load(streamingMode:)` integration for the bounded pread cache.
@Suite struct ModelLoaderPreadTests {

    private static func readBytes(_ view: TensorView) -> [UInt8] {
        let base = view.buffer.contents().advanced(by: Int(view.offset))
        return [UInt8](UnsafeRawBufferPointer(start: base, count: Int(view.length)))
    }

    @Test func loadsUnderPread_routedExpertBytesAndLazyOpen() throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .gemma4Toy(),
                                   streamingMode: .pread(slotCount: 2))

        #expect(model.openLayerFileCount() == 0)
        let view = try model.routedExpert(layer: 1, expert: 4)
        #expect(model.openLayerFileCount() == 1)

        // pread returns the scratch slot: offset is 0, not exp.offset.
        #expect(view.offset == 0)

        // Same tagged-byte contract as ModelLoaderTests.routedExpertBytesRoundTrip.
        let b = Self.readBytes(view)
        #expect(b[0] == 1)       // layer 1
        #expect(b[1] == 4)       // expert 4
        #expect(b[2] == 0xC1)
        #expect(b[3] == 0xC2)
    }

    @Test func routedExpertCacheSlotCountDoesNotOpenLayerFile() throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .gemma4Toy(),
                                   streamingMode: .pread(slotCount: 2))

        #expect(model.openLayerFileCount() == 0)
        #expect(model.routedExpertCacheSlotCount(layer: 1) == 2)
        #expect(model.openLayerFileCount() == 0)
    }

    @Test func beginOpeningRoutedExpertStreamerIsCompatibleWithLazyFetch() throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .gemma4Toy(),
                                   streamingMode: .pread(slotCount: 2))

        model.beginOpeningRoutedExpertStreamer(layer: 1)
        let view = try model.routedExpert(layer: 1, expert: 4)

        #expect(model.openLayerFileCount() == 1)
        let b = Self.readBytes(view)
        #expect(b[0] == 1)
        #expect(b[1] == 4)
    }

}
