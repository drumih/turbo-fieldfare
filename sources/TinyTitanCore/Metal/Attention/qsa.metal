#include <metal_stdlib>
using namespace metal;

// Qwen Sparse Attention's indexer: the small head set that scores whole
// blocks of pooled keys so the attention can be restricted to the highest
// scoring ones.
//
// The pooled keys are a cache like the KV cache: only the block containing
// the newest token changes on a decode step, so nothing here recomputes the
// history. The kernels are deliberately small and separate -- pool, then
// norm, then rope, then score -- because the norm and the rope are the ones
// already written, and the reference applies them in exactly that order with
// the block's position being `block_index * compress_ratio`, not the token's.

// pooled[b] = mean of the raw indexer keys in block b.
//
// The tail block is the mean of however many members exist, not of a padded
// four: dividing by the wrong count would scale the last block's score
// against every other one.
[[kernel, max_total_threads_per_threadgroup(256)]]
void qsa_pool_block(
    device const half* rawKeys  [[buffer(0)]],  // [n_kv, D]
    device       half* pooled   [[buffer(1)]],  // [n_blocks, D]
    constant     uint& D        [[buffer(2)]],
    constant     uint& blockIdx [[buffer(3)]],
    constant     uint& first    [[buffer(4)]],  // first key of the block
    constant     uint& count    [[buffer(5)]],  // members present
    uint               d        [[thread_position_in_grid]]
) {
    if (d >= D || count == 0u) return;
    float acc = 0.0f;
    for (uint i = 0; i < count; ++i) {
        acc += float(rawKeys[(first + i) * D + d]);
    }
    pooled[blockIdx * D + d] = half(acc / float(count));
}

// The same pooling over a contiguous range of blocks, for a prefill chunk.
//
// A chunk fills many blocks at once, and dispatching one kernel per block
// would be thousands of dispatches a layer. `n_kv` bounds the last block,
// which is the only one that can be ragged.
[[kernel, max_total_threads_per_threadgroup(256)]]
void qsa_pool_blocks(
    device const half* rawKeys    [[buffer(0)]],  // [n_kv, D]
    device       half* pooled     [[buffer(1)]],  // [n_blocks, D]
    constant     uint& D          [[buffer(2)]],
    constant     uint& firstBlock [[buffer(3)]],
    constant     uint& blockCount [[buffer(4)]],
    constant     uint& ratio      [[buffer(5)]],
    constant     uint& nKV        [[buffer(6)]],
    uint               tid        [[thread_position_in_grid]]
) {
    if (tid >= D * blockCount) return;
    const uint local = tid / D;
    const uint d = tid - local * D;
    const uint block = firstBlock + local;
    const uint first = block * ratio;
    if (first >= nKV) return;
    const uint count = min(ratio, nKV - first);
    float acc = 0.0f;
    for (uint i = 0; i < count; ++i) {
        acc += float(rawKeys[(first + i) * D + d]);
    }
    pooled[block * D + d] = half(acc / float(count));
}

// score[b] = sum over heads of max(0, dot(q[h], pooled[b])).
//
// The rectifier is applied per head and BEFORE the sum, so a head that
// dislikes a block contributes nothing rather than cancelling another head's
// preference. Summing first and rectifying after would be a different, and
// much flatter, ranking.
[[kernel, max_total_threads_per_threadgroup(256)]]
void qsa_block_scores(
    device const half*  query   [[buffer(0)]],  // [H, D], normed and roped
    device const half*  pooled  [[buffer(1)]],  // [n_blocks, D]
    device       float* scores  [[buffer(2)]],  // [n_blocks]
    constant     uint&  D       [[buffer(3)]],
    constant     uint&  H       [[buffer(4)]],
    constant     uint&  blocks  [[buffer(5)]],
    uint                b       [[thread_position_in_grid]]
) {
    if (b >= blocks) return;
    device const half* row = pooled + b * D;
    float total = 0.0f;
    for (uint h = 0; h < H; ++h) {
        device const half* q = query + h * D;
        float dot = 0.0f;
        for (uint d = 0; d < D; ++d) {
            dot = fma(float(q[d]), float(row[d]), dot);
        }
        total += max(dot, 0.0f);
    }
    scores[b] = total;
}

// Block scores for a whole prefill chunk: one score per (query, block).
//
// The decode kernel scores one query against every block; a chunk needs the
// cross product. Splitting it by thread rather than by dispatch keeps this to
// one encode per layer instead of one per query, which at a 2,048-token chunk
// is the difference between a kernel launch and two thousand of them.
[[kernel, max_total_threads_per_threadgroup(256)]]
void qsa_block_scores_rows(
    device const half*  query   [[buffer(0)]],  // [T, H, D], normed and roped
    device const half*  pooled  [[buffer(1)]],  // [n_blocks, D]
    device       float* scores  [[buffer(2)]],  // [T, n_blocks]
    constant     uint&  D       [[buffer(3)]],
    constant     uint&  H       [[buffer(4)]],
    constant     uint&  blocks  [[buffer(5)]],
    constant     uint&  T       [[buffer(6)]],
    uint                tid     [[thread_position_in_grid]]
) {
    if (tid >= blocks * T) return;
    const uint t = tid / blocks;
    const uint b = tid - t * blocks;
    device const half* row = pooled + b * D;
    device const half* q0 = query + t * H * D;
    float total = 0.0f;
    for (uint h = 0; h < H; ++h) {
        device const half* q = q0 + h * D;
        float dot = 0.0f;
        for (uint d = 0; d < D; ++d) {
            dot = fma(float(q[d]), float(row[d]), dot);
        }
        total += max(dot, 0.0f);
    }
    scores[tid] = total;
}

// ---------------------------------------------------------------------------
// Decode-time key selection on the GPU. The reference (QSAIndexer.selectKeys)
// ranks the complete blocks by score descending with the lower index first on
// ties, keeps whole blocks in that order until the budget is spent, gives the
// next block whatever cells remain, and always keeps the ragged tail of the
// query's own block. Doing that on the CPU costs a command-buffer round trip
// per full-attention layer per token once the context passes the budget.
//
// Here the ranking is a radix select over a 64-bit key (orderable score,
// then inverted index) for the block at rank B = remaining / ratio: every
// block whose key is larger is kept whole, the block at rank B gets the
// partial, the rest are dropped. One threadgroup; eight 8-bit passes.
// ---------------------------------------------------------------------------
static inline uint qsa_orderable(float s) {
    if (s == 0.0f) s = 0.0f;                 // -0.0 ties +0.0 in the reference
    uint f = as_type<uint>(s);
    return (f & 0x80000000u) ? ~f : (f | 0x80000000u);
}

static inline ulong qsa_select_key(device const float* scores, uint i) {
    return (ulong(qsa_orderable(scores[i])) << 32) | ulong(0xFFFFFFFFu - i);
}

[[kernel, max_total_threads_per_threadgroup(1024)]]
void qsa_select_decode(
    device const float* scores  [[buffer(0)]],   // [>= blocks]
    device uint8_t*     keep    [[buffer(1)]],   // [visible] out
    constant uint&      visible [[buffer(2)]],
    constant uint&      ratio   [[buffer(3)]],
    constant uint&      width   [[buffer(4)]],   // selectionWidth
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    threadgroup atomic_uint hist[256];
    threadgroup ulong prefix_tg;
    threadgroup uint rank_tg;

    const uint complete = (visible / ratio) * ratio;
    const uint tail = visible - complete;
    for (uint c = complete + lid; c < visible; c += lsize) keep[c] = 1u;
    if (width <= tail) {
        for (uint c = lid; c < complete; c += lsize) keep[c] = 0u;
        return;
    }
    const uint remaining = width - tail;
    const uint blocks = complete / ratio;
    const uint B = remaining / ratio;
    const uint partial = remaining - B * ratio;
    if (B >= blocks) {
        for (uint c = lid; c < complete; c += lsize) keep[c] = 1u;
        return;
    }

    // Radix select: find the key of the block at rank B (0-based, descending).
    if (lid == 0) { prefix_tg = 0ul; rank_tg = B; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int pass = 7; pass >= 0; --pass) {
        const uint shift = uint(pass) * 8u;
        for (uint b = lid; b < 256u; b += lsize)
            atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ulong prefix = prefix_tg;
        const ulong mask = (pass == 7) ? 0ul : (~0ul << (shift + 8u));
        for (uint i = lid; i < blocks; i += lsize) {
            const ulong k = qsa_select_key(scores, i);
            if ((k & mask) == prefix) {
                atomic_fetch_add_explicit(&hist[uint((k >> shift) & 0xFFul)], 1u,
                                          memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            uint r = rank_tg;
            uint digit = 0u;
            for (int b = 255; b >= 0; --b) {
                const uint n = atomic_load_explicit(&hist[uint(b)], memory_order_relaxed);
                if (r < n) { digit = uint(b); break; }
                r -= n;
            }
            rank_tg = r;
            prefix_tg = prefix | (ulong(digit) << shift);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const ulong keyB = prefix_tg;
    for (uint i = lid; i < blocks; i += lsize) {
        const ulong k = qsa_select_key(scores, i);
        const bool whole = k > keyB;
        const bool at_rank = k == keyB;
        const uint base = i * ratio;
        for (uint c = 0; c < ratio; ++c) {
            keep[base + c] = whole ? 1u : ((at_rank && c < partial) ? 1u : 0u);
        }
    }
}
