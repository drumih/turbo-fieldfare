import Foundation
import Metal
import Testing

@testable import TinyTitan
import TinyTitanCore

/// The indexer's per-layer buffer growth.
///
/// The raw keys and pooled blocks are storage, not state: a grow copies both
/// bit for bit, so a selection made after a grow sees exactly what it would
/// have seen before one. These tests pin the copy, the doubling, and the
/// allocation policy that is the point of growing at all — a short
/// conversation must not wire the full context's 84 MB a layer
/// (measured in `docs/qwen38-decode-12gb-measurements.md` §5o).
@Suite("QSA buffer growth")
struct QSAIndexerGrowthTests {
    private static let config = SparseIndexerConfig(
        numHeads: 4, numKVHeads: 1, headDim: 128,
        budget: 64, compressRatio: 4)

    private static let cellBytes = 128 * MemoryLayout<Float16>.stride

    private static func makeIndexer(capacity: Int = 32768) throws -> QSAIndexer {
        try QSAIndexer(
            context: try MetalContext(), config: config,
            ropeTheta: 10_000_000, capacity: capacity)
    }

    @Test("A layer starts at the initial capacity, not the full context")
    func allocationStartsSmall() throws {
        let indexer = try Self.makeIndexer()
        let buffers = try indexer.layerBuffers(0, hold: 1)
        #expect(
            buffers.raw.length == QSAIndexer.initialCapacityCells * Self.cellBytes)
        #expect(
            buffers.pooled.length
                == (QSAIndexer.initialCapacityCells / 4 + 1) * Self.cellBytes)
    }

    @Test("A small configured capacity still allocates in full on first use")
    func smallCapacityIsWhole() throws {
        let indexer = try Self.makeIndexer(capacity: 512)
        let buffers = try indexer.layerBuffers(0, hold: 1)
        #expect(buffers.raw.length == 512 * Self.cellBytes)
    }

    @Test("Growth doubles to reach the request and stops at the capacity")
    func growthDoublesAndCaps() throws {
        let indexer = try Self.makeIndexer()
        _ = try indexer.layerBuffers(0, hold: 1)
        let grown = try indexer.layerBuffers(0, hold: QSAIndexer.initialCapacityCells + 1)
        #expect(grown.raw.length == 2 * QSAIndexer.initialCapacityCells * Self.cellBytes)
        let full = try indexer.layerBuffers(0, hold: 32768)
        #expect(full.raw.length == 32768 * Self.cellBytes)
        // Holding at or below what is allocated never reallocates.
        let again = try indexer.layerBuffers(0, hold: 100)
        #expect(again.raw.length == full.raw.length)
    }

    @Test("Growing a layer keeps every raw key and pooled block")
    func growthPreservesContents() throws {
        let indexer = try Self.makeIndexer()
        let cells = QSAIndexer.initialCapacityCells
        let first = try indexer.layerBuffers(0, hold: 1)
        let rawValues = cells * 128
        let raw = first.raw.contents().bindMemory(to: Float16.self, capacity: rawValues)
        for i in 0..<rawValues { raw[i] = Float16(i % 1000) }
        let poolCells = cells / 4 + 1
        let poolValues = poolCells * 128
        let pool = first.pooled.contents().bindMemory(
            to: Float16.self, capacity: poolValues)
        for i in 0..<poolValues { pool[i] = Float16((i * 7) % 1000) }

        let grown = try indexer.layerBuffers(0, hold: cells + 1)
        #expect(grown.raw.length > first.raw.length)
        let grownRaw = grown.raw.contents().bindMemory(
            to: Float16.self, capacity: rawValues)
        for i in 0..<rawValues {
            #expect(grownRaw[i] == Float16(i % 1000))
        }
        let grownPool = grown.pooled.contents().bindMemory(
            to: Float16.self, capacity: poolValues)
        for i in 0..<poolValues {
            #expect(grownPool[i] == Float16((i * 7) % 1000))
        }
    }
}
