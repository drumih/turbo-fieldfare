import Foundation
import Metal
import TinyTitanCore

/// Tiled prefill GEMMs over GGUF weights (`gguf_gemm_tiled.metal`):
/// `Y[p][r] = W[r] · X[x(p)]` for a chunk of tokens, dense or grouped by
/// routed expert. Activations and outputs are f32 rows; weights stay in their
/// GGUF blocks and are dequantised per 32-token tile.
final class GGUFGemmTiled {
    typealias Operand = GGUFGemvIds.Operand

    private struct Args {
        var rows: UInt32
        var columns: UInt32
        var tokens: UInt32
        var expertStride: UInt64
        var listStride: UInt32
        var xDivisor: UInt32
        var indirect: UInt32
        var useScale: UInt32
    }

    static let supported: Set<GGUFType> = [.q8_0, .nvfp4, .f32, .bf16]
    private static let tileRows = 64
    private static let tileTokens = 32

    private var pipelines: [GGUFType: MTLComputePipelineState] = [:]
    private let group: MTLComputePipelineState
    private let clear: MTLComputePipelineState
    private let sizes: MTLComputePipelineState

    init(context: MetalContext) throws {
        for (type, name) in [
            (GGUFType.q8_0, "gemm_tiled_q8_0"), (.nvfp4, "gemm_tiled_nvfp4"),
            (.f32, "gemm_tiled_f32"), (.bf16, "gemm_tiled_bf16"),
        ] {
            pipelines[type] = try context.pipeline(name)
        }
        group = try context.pipeline("moe_group_pairs")
        clear = try context.pipeline("clear_counts")
        sizes = try context.pipeline("moe_dispatch_sizes")
    }

    private func pipeline(_ type: GGUFType, columns: Int) throws -> MTLComputePipelineState {
        guard let state = pipelines[type] else {
            throw GGUFDequantError.unsupportedType(type.name)
        }
        guard columns % 64 == 0 else {
            throw MetalError.invalidState("tiled GEMM: \(columns) columns for \(type.name)")
        }
        return state
    }

    /// `y[t] = W · x[t]` for `tokens` rows of `x` (each `columns` floats).
    func encodeDense(
        type: GGUFType, weights: Operand, rows: Int, columns: Int, tokens: Int, x: Operand,
        y: Operand, encoder: MTLComputeCommandEncoder
    ) throws {
        encoder.setComputePipelineState(try pipeline(type, columns: columns))
        var args = Args(
            rows: UInt32(rows), columns: UInt32(columns), tokens: UInt32(tokens),
            expertStride: 0, listStride: 0, xDivisor: 1, indirect: 0, useScale: 0)
        bind(weights: weights, x: x, y: y, lists: x, counts: x, scales: x, args: &args, encoder)
        dispatch(rows: rows, tokenTiles: (tokens + 31) / 32, experts: 1, encoder)
    }

    /// Expert-grouped GEMM: for every pair `p` in expert `e`'s list,
    /// `y[p] = scales[e] * W_e · x[p / xDivisor]`. `maxPairsPerExpert` is each
    /// list's capacity. With `indirectSizes` (written by `encodeDispatchSizes`)
    /// the grid has only the token tiles the largest expert needs; without it,
    /// tiles for the whole capacity are launched and the empty ones exit.
    func encodeExperts(
        type: GGUFType, weights: Operand, rows: Int, columns: Int, experts: Int,
        expertStride: Int, maxPairsPerExpert: Int, xDivisor: Int, x: Operand, y: Operand,
        lists: Operand, counts: Operand, scales: Operand?, indirectSizes: Operand? = nil,
        encoder: MTLComputeCommandEncoder
    ) throws {
        encoder.setComputePipelineState(try pipeline(type, columns: columns))
        var args = Args(
            rows: UInt32(rows), columns: UInt32(columns), tokens: 0,
            expertStride: UInt64(expertStride), listStride: UInt32(maxPairsPerExpert),
            xDivisor: UInt32(xDivisor), indirect: 1, useScale: scales == nil ? 0 : 1)
        bind(
            weights: weights, x: x, y: y, lists: lists, counts: counts, scales: scales ?? counts,
            args: &args, encoder)
        if let indirectSizes {
            encoder.dispatchThreadgroups(
                indirectBuffer: indirectSizes.buffer,
                indirectBufferOffset: indirectSizes.offset,
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            dispatch(
                rows: rows, tokenTiles: (maxPairsPerExpert + 31) / 32, experts: experts, encoder)
        }
    }

    /// Writes one indirect grid (row tiles, token tiles, experts) per entry of
    /// `rowsOfShapes` (up to four) into `sizes`, 12 bytes each, from the
    /// expert pair counts. Run after grouping, before the GEMMs.
    func encodeDispatchSizes(
        counts: Operand, experts: Int, rowsOfShapes: [Int], sizes out: Operand,
        encoder: MTLComputeCommandEncoder
    ) {
        precondition((1...4).contains(rowsOfShapes.count))
        var tiles = SIMD4<UInt32>(repeating: 1)
        for (index, rows) in rowsOfShapes.enumerated() {
            tiles[index] = UInt32((rows + Self.tileRows - 1) / Self.tileRows)
        }
        var n = UInt32(experts)
        var shapes = UInt32(rowsOfShapes.count)
        encoder.setComputePipelineState(sizes)
        encoder.setBuffer(counts.buffer, offset: counts.offset, index: 0)
        encoder.setBuffer(out.buffer, offset: out.offset, index: 1)
        encoder.setBytes(&n, length: 4, index: 2)
        encoder.setBytes(&tiles, length: 16, index: 3)
        encoder.setBytes(&shapes, length: 4, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// Clears `counts` and fills `lists` from `ids` (`pairs` routed expert ids,
    /// token-major), so `encodeExperts` can run. Two dispatches; the caller
    /// orders them with the GEMMs that read the lists.
    func encodeGrouping(
        ids: Operand, pairs: Int, experts: Int, lists: Operand, counts: Operand,
        listStride: Int, barrier: () -> Void, encoder: MTLComputeCommandEncoder
    ) {
        var n = UInt32(experts)
        encoder.setComputePipelineState(clear)
        encoder.setBuffer(counts.buffer, offset: counts.offset, index: 0)
        encoder.setBytes(&n, length: 4, index: 1)
        encoder.dispatchThreads(
            MTLSize(width: experts, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(experts, 256), height: 1, depth: 1))
        barrier()
        var count = UInt32(pairs)
        var stride = UInt32(listStride)
        encoder.setComputePipelineState(group)
        encoder.setBuffer(ids.buffer, offset: ids.offset, index: 0)
        encoder.setBuffer(counts.buffer, offset: counts.offset, index: 1)
        encoder.setBuffer(lists.buffer, offset: lists.offset, index: 2)
        encoder.setBytes(&count, length: 4, index: 3)
        encoder.setBytes(&stride, length: 4, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: pairs, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(pairs, 256), height: 1, depth: 1))
    }

    private func bind(
        weights: Operand, x: Operand, y: Operand, lists: Operand, counts: Operand,
        scales: Operand, args: inout Args, _ encoder: MTLComputeCommandEncoder
    ) {
        encoder.setBuffer(weights.buffer, offset: weights.offset, index: 0)
        encoder.setBuffer(x.buffer, offset: x.offset, index: 1)
        encoder.setBuffer(y.buffer, offset: y.offset, index: 2)
        encoder.setBuffer(lists.buffer, offset: lists.offset, index: 3)
        encoder.setBuffer(counts.buffer, offset: counts.offset, index: 4)
        encoder.setBuffer(scales.buffer, offset: scales.offset, index: 5)
        encoder.setBytes(&args, length: MemoryLayout<Args>.stride, index: 6)
    }

    private func dispatch(
        rows: Int, tokenTiles: Int, experts: Int, _ encoder: MTLComputeCommandEncoder
    ) {
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (rows + Self.tileRows - 1) / Self.tileRows, height: max(tokenTiles, 1),
                depth: experts),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}
