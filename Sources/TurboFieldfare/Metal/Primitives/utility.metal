#include <metal_stdlib>
using namespace metal;

// Kept in the shared library so both INT4 and INT8 shared-expert paths use
// the same Gemma activation without compiling a private shader module.
[[kernel, max_total_threads_per_threadgroup(256)]]
void gelu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half(gelu_pytorch_tanh(g) * u);
}
