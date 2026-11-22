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

    static let supported: Set<GGUFType> = [
        .q8_0, .nvfp4, .f32, .bf16, .iq3XXS, .q4K, .q5K, .q6K, .iq1S, .iq2XXS, .iq2XS, .iq2S,
        .q2_0, .mxfp4,
    ]
    /// Row tiles: `GEMM_WIDE_ROWS` for the 32-token kernels, 64 for the
    /// 8-token ones (gguf_gemm_tiled.metal).
    static let wideTileRows = 128
    /// Below this many 128-row threadgroups a dense GEMM takes 64-row tiles.
    static let smallGridThreshold = 128
    // (8-token tiles for dense projections at chunks of <= 64 tokens measured
    // no faster at pp64: ~101 tok/s against 102-107 with these.)
    static let narrowTileRows = 64
    /// `GEMM_NARROW_BK`: the 8-token kernels need columns % depth == 0.
    static let narrowTileDepth = 64

    private var pipelines: [GGUFType: MTLComputePipelineState] = [:]
    private var narrowPipelines: [GGUFType: MTLComputePipelineState] = [:]
    /// 64-row 32-token dense kernels, for grids the 128-row tile leaves small.
    private var smallGridPipelines: [GGUFType: MTLComputePipelineState] = [:]
    /// The IQ grids the block decoders read (`GGUFBlockTables`).
    let tables: MTLBuffer
    private let group: MTLComputePipelineState
    private let clear: MTLComputePipelineState

    init(context: MetalContext) throws {
        let wide: [(GGUFType, String)] = [
            (.q8_0, "q8_0"), (.nvfp4, "nvfp4"), (.f32, "f32"), (.bf16, "bf16"),
            (.iq3XXS, "iq3_xxs"), (.q4K, "q4_k"), (.q5K, "q5_k"), (.q6K, "q6_k"),
            (.iq1S, "iq1_s"), (.iq2XXS, "iq2_xxs"), (.iq2XS, "iq2_xs"), (.iq2S, "iq2_s"),
            (.q2_0, "q2_0"), (.mxfp4, "mxfp4"),
        ]
        for (type, name) in wide { pipelines[type] = try context.pipeline("gemm_tiled_\(name)") }
        // Every supported type must have its 32-token kernel: the small-grid
        // and 8-token paths cover only some shapes, so a gap here shows up only
        // on large chunks.
        guard Set(pipelines.keys) == Self.supported else {
            throw MetalError.invalidState(
                "tiled GEMM: no 32-token kernel for "
                    + Self.supported.subtracting(pipelines.keys).map(\.name).sorted()
                    .joined(separator: ", "))
        }
        let narrow: [(GGUFType, String)] = [
            (.q8_0, "q8_0"), (.nvfp4, "nvfp4"), (.iq1S, "iq1_s"), (.iq2XXS, "iq2_xxs"),
            (.iq2XS, "iq2_xs"), (.iq2S, "iq2_s"), (.q2_0, "q2_0"), (.mxfp4, "mxfp4"),
        ]
        for (type, name) in narrow {
            narrowPipelines[type] = try context.pipeline("gemm_tiled_\(name)_n8")
        }
        for (type, name) in [
            (GGUFType.q8_0, "q8_0"), (.iq3XXS, "iq3_xxs"), (.q4K, "q4_k"), (.q5K, "q5_k"),
            (.q6K, "q6_k"),
        ] {
            smallGridPipelines[type] = try context.pipeline("gemm_tiled_\(name)_m64")
        }
        tables = try GGUFBlockTables.buffer(device: context.device)
        group = try context.pipeline("moe_group_pairs")
        clear = try context.pipeline("clear_counts")
    }

    /// Token-tile width of the expert GEMMs: 8 while experts average at most
    /// sixteen pairs (a 32-wide tile would multiply mostly padding), else 32.
    /// Measured on one routed NVFP4 layer (M3 24 GB): 16 pairs/expert 22.9 ms
    /// narrow vs 35.6-38.3 ms wide; 32 pairs/expert 45.2 ms vs 37.4-39.6 ms.
    static func expertTileTokens(pairs: Int, experts: Int) -> Int {
        pairs <= 16 * experts ? 8 : 32
    }

    private func pipeline(
        _ type: GGUFType, columns: Int, tileTokens: Int = 32
    ) throws -> MTLComputePipelineState {
        // Types without a narrow kernel fall back to the 32-token tile.
        guard let state = (tileTokens == 8 ? narrowPipelines[type] : nil) ?? pipelines[type]
        else {
            throw GGUFDequantError.unsupportedType(type.name)
        }
        guard columns % 64 == 0, columns % Int(type.blockSize) == 0 else {
            throw MetalError.invalidState("tiled GEMM: \(columns) columns for \(type.name)")
        }
        return state
    }

    /// `y[t] = W · x[t]` for `tokens` rows of `x` (each `columns` floats).
    func encodeDense(
        type: GGUFType, weights: Operand, rows: Int, columns: Int, tokens: Int, x: Operand,
        y: Operand, encoder: MTLComputeCommandEncoder
    ) throws {
        let tokenTiles = (tokens + 31) / 32
        let wideGroups = (rows + Self.wideTileRows - 1) / Self.wideTileRows * tokenTiles
        let small = wideGroups < Self.smallGridThreshold ? smallGridPipelines[type] : nil
        encoder.setComputePipelineState(try small ?? pipeline(type, columns: columns))
        var args = Args(
            rows: UInt32(rows), columns: UInt32(columns), tokens: UInt32(tokens),
            expertStride: 0, listStride: 0, xDivisor: 1, indirect: 0, useScale: 0)
        bind(weights: weights, x: x, y: y, lists: x, counts: x, scales: x, args: &args, encoder)
        dispatch(
            rows: rows, tileRows: small == nil ? Self.wideTileRows : 64, tokenTiles: tokenTiles,
            experts: 1, encoder)
    }

    /// Expert-grouped GEMM: for every pair `p` in expert `e`'s list,
    /// `y[p] = scales[e] * W_e · x[p / xDivisor]`. Each list holds up to
    /// `listStride` entries (default `maxPairsPerExpert`); the grid covers
    /// `maxPairsPerExpert` pairs per expert — an upper bound on every count,
    /// known on the CPU — and token tiles past `counts[e]` exit at once.
    func encodeExperts(
        type: GGUFType, weights: Operand, rows: Int, columns: Int, experts: Int,
        expertStride: Int, maxPairsPerExpert: Int, xDivisor: Int, x: Operand, y: Operand,
        lists: Operand, counts: Operand, scales: Operand?, listStride: Int? = nil,
        tileTokens: Int = 32, encoder: MTLComputeCommandEncoder
    ) throws {
        let narrow =
            tileTokens == 8 && narrowPipelines[type] != nil
            && columns % Self.narrowTileDepth == 0
        encoder.setComputePipelineState(
            try pipeline(type, columns: columns, tileTokens: narrow ? 8 : 32))
        var args = Args(
            rows: UInt32(rows), columns: UInt32(columns), tokens: 0,
            expertStride: UInt64(expertStride),
            listStride: UInt32(listStride ?? maxPairsPerExpert), xDivisor: UInt32(xDivisor),
            indirect: 1, useScale: scales == nil ? 0 : 1)
        bind(
            weights: weights, x: x, y: y, lists: lists, counts: counts, scales: scales ?? counts,
            args: &args, encoder)
        // The grid covers the worst case (every token on one expert); empty
        // tiles exit. A grid sized from the average with threadgroups looping
        // over further tiles saved ~8% on evenly routed benchmarks but lost
        // 4% at pp512 on the real, skewed routing (a popular expert's tiles
        // serialise in one threadgroup). Doing that for the 8-token grid only
        // lost ~4-5% at pp64 as well (118 -> 113.6 tok/s on Q2_K_XL).
        let tokensPerTile = narrow ? 8 : 32
        dispatch(
            rows: rows, tileRows: narrow ? Self.narrowTileRows : Self.wideTileRows,
            tokenTiles: (maxPairsPerExpert + tokensPerTile - 1) / tokensPerTile,
            experts: experts, encoder)
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
        encoder.setBuffer(tables, offset: 0, index: 7)
    }

    private func dispatch(
        rows: Int, tileRows: Int, tokenTiles: Int, experts: Int,
        _ encoder: MTLComputeCommandEncoder
    ) {
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (rows + tileRows - 1) / tileRows, height: max(tokenTiles, 1),
                depth: experts),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}
