#include <metal_stdlib>
using namespace metal;

// Nemotron 3.5 Bolt's elementwise decode kernels.
//
// These exist so a layer can be encoded into one command buffer: the graph in
// `NemotronMetalGraph` keeps the hidden state, the Mamba conv/SSM state and the
// KV cache on the GPU, and only stops at a MoE router (where 9 of 240-496
// experts are chosen on the CPU). Running the norms, conv and scan on the CPU
// between GEMVs cost ~6 ms of GPU idle per layer -- ~4 s/token on an M3 -- which
// is the reason this file exists rather than reusing the C kernels.

// ---------------------------------------------------------------------------
// RMS norm. One threadgroup per vector; 256 threads reduce x^2.
// ---------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void nemotron_rmsnorm_f16(
    device const half*   x      [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],   // may be null
    device half*         out    [[buffer(2)]],
    constant uint&       n      [[buffer(3)]],
    constant float&      eps    [[buffer(4)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint i = lid; i < n; i += lsize) {
        const float v = float(x[i]);
        sum += v * v;
    }
    partial[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rstd = rsqrt(partial[0] / float(n) + eps);
    for (uint i = lid; i < n; i += lsize) {
        float v = float(x[i]) * rstd;
        if (weight != nullptr) v *= float(weight[i]);
        out[i] = half(v);
    }
}

// ---------------------------------------------------------------------------
// Mamba's gated RMS norm, one threadgroup per group of `group` elements.
// out = RMSNorm(x, group) * SiLU(gate), matching MambaRMSNormGated.
// ---------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void nemotron_rmsnorm_gated_f16(
    device const half*   x      [[buffer(0)]],
    device const half*   gate   [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    device half*         out    [[buffer(3)]],
    constant uint&       group  [[buffer(4)]],
    constant float&      eps    [[buffer(5)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[256];
    const uint base = gid * group;
    float sum = 0.0f;
    for (uint i = lid; i < group; i += lsize) {
        const float v = float(x[base + i]);
        sum += v * v;
    }
    partial[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rstd = rsqrt(partial[0] / float(group) + eps);
    for (uint i = lid; i < group; i += lsize) {
        const uint index = base + i;
        const float z = float(gate[index]);
        const float silu = z / (1.0f + exp(-z));
        out[index] = half(float(x[index]) * rstd * float(weight[index]) * silu);
    }
}

// ---------------------------------------------------------------------------
// Depthwise causal convolution over the last four inputs, then SiLU.
// `state` is channels x 4 floats, oldest first, updated in place.
// ---------------------------------------------------------------------------
[[kernel]]
kernel void nemotron_conv_silu_f16(
    device const half*   input  [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* bias   [[buffer(2)]],
    device float*        state  [[buffer(3)]],
    device half*         out    [[buffer(4)]],
    constant uint&       channels [[buffer(5)]],
    constant uint&       kernelSize [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= channels) return;
    device float* history = state + gid * kernelSize;
    for (uint k = 0; k + 1 < kernelSize; ++k) history[k] = history[k + 1];
    history[kernelSize - 1] = float(input[gid]);
    float acc = 0.0f;
    for (uint k = 0; k < kernelSize; ++k) {
        acc += history[k] * float(weight[gid * kernelSize + k]);
    }
    if (bias != nullptr) acc += float(bias[gid]);
    out[gid] = half(acc / (1.0f + exp(-acc)));
}

// ---------------------------------------------------------------------------
// Mamba-2 selective scan for one token. One thread per (head, head-dim).
// ---------------------------------------------------------------------------
[[kernel]]
kernel void nemotron_ssm_f16(
    device const half*   x       [[buffer(0)]],
    device const half*   b       [[buffer(1)]],
    device const half*   c       [[buffer(2)]],
    device const half*   dt      [[buffer(3)]],
    device const bfloat* aLog    [[buffer(4)]],
    device const bfloat* skipD   [[buffer(5)]],
    device const bfloat* dtBias  [[buffer(6)]],
    device float*        state   [[buffer(7)]],
    device half*         y       [[buffer(8)]],
    constant uint&       heads   [[buffer(9)]],
    constant uint&       headDim [[buffer(10)]],
    constant uint&       stateSize [[buffer(11)]],
    constant uint&       groups  [[buffer(12)]],
    constant float&      timeStepMin [[buffer(13)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = heads * headDim;
    if (gid >= total) return;
    const uint head = gid / headDim;
    const uint dim = gid % headDim;
    const uint headsPerGroup = heads / groups;
    const uint group = head / headsPerGroup;
    const float a = -exp(float(aLog[head]));
    const float raw = float(dt[head]) + float(dtBias[head]);
    float dtValue = raw > 20.0f ? raw : log(1.0f + exp(raw));
    dtValue = max(dtValue, timeStepMin);
    const float decay = exp(dtValue * a);
    const float xv = float(x[gid]);
    const float dB = dtValue * xv;
    device const half* bGroup = b + group * stateSize;
    device const half* cGroup = c + group * stateSize;
    device float* row = state + (head * headDim + dim) * stateSize;
    float acc = 0.0f;
    for (uint n = 0; n < stateSize; ++n) {
        row[n] = row[n] * decay + dB * float(bGroup[n]);
        acc += row[n] * float(cGroup[n]);
    }
    y[gid] = half(acc + float(skipD[head]) * xv);
}

// ---------------------------------------------------------------------------
// Elementwise helpers.
// ---------------------------------------------------------------------------
[[kernel]]
kernel void nemotron_relu2_f16(
    device const half* x [[buffer(0)]],
    device half*       out [[buffer(1)]],
    constant uint&     n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    const float v = float(x[gid]);
    const float r = v > 0.0f ? v : 0.0f;
    out[gid] = half(r * r);
}

[[kernel]]
kernel void nemotron_add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half*       out [[buffer(2)]],
    constant uint&     n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = half(float(a[gid]) + float(b[gid]));
}

/// `accum[i] = sum_m weight[m] * down[m * width + i]`, the routed-expert reduce.
[[kernel]]
kernel void nemotron_moe_combine_f16(
    device const half* down    [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device half*       accum   [[buffer(2)]],
    constant uint&     width   [[buffer(3)]],
    constant uint&     count   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= width) return;
    float sum = 0.0f;
    for (uint m = 0; m < count; ++m) {
        sum += float(weights[m]) * float(down[m * width + gid]);
    }
    accum[gid] = half(sum);
}

// ---------------------------------------------------------------------------
// One row of an affine group-64 embedding table.
// ---------------------------------------------------------------------------
[[kernel]]
kernel void nemotron_embedding_f16(
    device const uint8_t* weights [[buffer(0)]],
    device const bfloat*  scales  [[buffer(1)]],
    device const bfloat*  biases  [[buffer(2)]],
    device half*          out     [[buffer(3)]],
    constant uint&        columns [[buffer(4)]],
    constant uint&        bits    [[buffer(5)]],
    constant uint&        row     [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= columns) return;
    const uint groups = columns / 64;
    const uint group = gid / 64;
    const uint k = gid % 64;
    float q;
    if (bits == 4) {
        device const uint8_t* rowWeights = weights + row * (columns / 2);
        const uint8_t byte = rowWeights[(group * 64 + k) >> 1];
        q = float((group * 64 + k) & 1 ? (byte >> 4) : (byte & 0x0F));
    } else {
        device const uint8_t* rowWeights = weights + row * columns;
        q = float(rowWeights[group * 64 + k]);
    }
    out[gid] = half(q * float(scales[row * groups + group])
                    + float(biases[row * groups + group]));
}

// ---------------------------------------------------------------------------
// No-RoPE GQA decode attention. One thread per query head; `scores` is scratch
// of heads * maxSequence floats. No positional encoding by construction.
// ---------------------------------------------------------------------------
[[kernel]]
kernel void nemotron_attention_f16(
    device const half* q       [[buffer(0)]],   // heads * headDim
    device const half* kCache  [[buffer(1)]],   // maxSeq * kvHeads * headDim
    device const half* vCache  [[buffer(2)]],
    device float*      scores  [[buffer(3)]],   // heads * maxSeq
    device half*       out     [[buffer(4)]],
    constant uint&     heads   [[buffer(5)]],
    constant uint&     kvHeads [[buffer(6)]],
    constant uint&     headDim [[buffer(7)]],
    constant uint&     maxSeq  [[buffer(8)]],
    constant uint&     position [[buffer(9)]],
    constant float&    scale   [[buffer(10)]],
    uint head [[thread_position_in_grid]]
) {
    if (head >= heads) return;
    const uint groups = heads / kvHeads;
    const uint kvHead = head / groups;
    device const half* qh = q + head * headDim;
    device float* row = scores + head * maxSeq;
    float maximum = -INFINITY;
    for (uint token = 0; token <= position; ++token) {
        device const half* kb = kCache + (token * kvHeads + kvHead) * headDim;
        float dot = 0.0f;
        for (uint d = 0; d < headDim; ++d) dot += float(qh[d]) * float(kb[d]);
        dot *= scale;
        row[token] = dot;
        maximum = max(maximum, dot);
    }
    float total = 0.0f;
    for (uint token = 0; token <= position; ++token) {
        const float e = exp(row[token] - maximum);
        row[token] = e;
        total += e;
    }
    const float inverse = 1.0f / total;
    device half* oh = out + head * headDim;
    for (uint d = 0; d < headDim; ++d) oh[d] = half(0.0f);
    for (uint token = 0; token <= position; ++token) {
        const float probability = row[token] * inverse;
        device const half* vb = vCache + (token * kvHeads + kvHead) * headDim;
        for (uint d = 0; d < headDim; ++d) {
            oh[d] = half(float(oh[d]) + probability * float(vb[d]));
        }
    }
}

// ---------------------------------------------------------------------------
// BF16 router GEMV: one thread per output row.
// ---------------------------------------------------------------------------
[[kernel]]
kernel void nemotron_bf16_gemv(
    device const bfloat* weights [[buffer(0)]],
    device const half*   x       [[buffer(1)]],
    device float*        out     [[buffer(2)]],
    constant uint&       rows    [[buffer(3)]],
    constant uint&       columns [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    device const bfloat* row = weights + gid * columns;
    float acc = 0.0f;
    for (uint i = 0; i < columns; ++i) acc += float(row[i]) * float(x[i]);
    out[gid] = acc;
}

// ---------------------------------------------------------------------------
// Append one token's K and V to the cache. One thread per KV element.
// ---------------------------------------------------------------------------
[[kernel]]
kernel void nemotron_kv_append_f16(
    device const half* k       [[buffer(0)]],
    device const half* v       [[buffer(1)]],
    device half*       kCache  [[buffer(2)]],
    device half*       vCache  [[buffer(3)]],
    constant uint&     kvHeads [[buffer(4)]],
    constant uint&     headDim [[buffer(5)]],
    constant uint&     maxSeq  [[buffer(6)]],
    constant uint&     position [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint width = kvHeads * headDim;
    if (gid >= width) return;
    kCache[position * width + gid] = k[gid];
    vCache[position * width + gid] = v[gid];
}
