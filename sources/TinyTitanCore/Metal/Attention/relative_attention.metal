#include <metal_stdlib>
using namespace metal;

// One thread per (query, head). Online softmax keeps the score pass single-pass;
// the sink is an extra zero-valued key. Supports causal sliding or global.
kernel void relative_attention(
    device const float* q        [[buffer(0)]],
    device const float* k        [[buffer(1)]],
    device const float* v        [[buffer(2)]],
    device const float* r        [[buffer(3)]],
    device const float* gate     [[buffer(4)]],
    device const float* sink     [[buffer(5)]],
    device const float* rel      [[buffer(6)]],
    device float* out            [[buffer(7)]],
    constant uint& tokens        [[buffer(8)]],
    constant uint& heads         [[buffer(9)]],
    constant uint& kvHeads       [[buffer(10)]],
    constant uint& dim           [[buffer(11)]],
    constant uint& relDim        [[buffer(12)]],
    constant uint& extent        [[buffer(13)]],
    constant uint& sliding       [[buffer(14)]],
    constant uint& pastKeys      [[buffer(15)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = tokens * heads;
    if (gid >= total || dim > 128u) return;
    const uint head = gid % heads;
    const uint query = gid / heads;
    const uint kvHead = head / (heads / kvHeads);
    const uint qBase = query * heads * dim + head * dim;
    const uint rBase = query * heads * relDim + head * relDim;
    const float sinkValue = sink[query * heads + head];
    float m = -INFINITY;
    float l = 0.0f;
    float acc[128];
    for (uint d = 0; d < dim; ++d) acc[d] = 0.0f;
    for (uint key = 0; key <= pastKeys + query; ++key) {
        const uint distance = pastKeys + query - key;
        if (sliding > 0u && distance >= sliding) continue;
        if (distance >= extent) continue;
        float score = 0.0f;
        for (uint d = 0; d < dim; ++d) {
            score += q[qBase + d] * k[key * kvHeads * dim + kvHead * dim + d];
        }
        score /= sqrt(float(dim));
        for (uint d = 0; d < relDim; ++d) {
            score += r[rBase + d] * rel[d * extent + distance];
        }
        const float newM = max(m, score);
        const float alpha = exp(m - newM);
        const float beta = exp(score - newM);
        l = l * alpha + beta;
        for (uint d = 0; d < dim; ++d) {
            acc[d] = acc[d] * alpha
                + beta * v[key * kvHeads * dim + kvHead * dim + d];
        }
        m = newM;
    }
    {
        const float newM = max(m, sinkValue);
        const float alpha = exp(m - newM);
        const float beta = exp(sinkValue - newM);
        l = l * alpha + beta;
        for (uint d = 0; d < dim; ++d) acc[d] = acc[d] * alpha;
    }
    for (uint d = 0; d < dim; ++d) {
        const float g = 1.0f / (1.0f + exp(-gate[qBase + d]));
        out[qBase + d] = (l > 0.0f ? acc[d] / l : 0.0f) * g;
    }
}
