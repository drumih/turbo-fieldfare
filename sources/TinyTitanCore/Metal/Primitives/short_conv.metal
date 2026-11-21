#include <metal_stdlib>
using namespace metal;

// Depthwise causal short convolution with the Inkling residual and
// 1/sqrt(kernel+1) scale. One thread per (token, channel).
kernel void short_conv4(
    device const float* x        [[buffer(0)]],
    device const float* weights  [[buffer(1)]],
    device const float* stateIn  [[buffer(2)]],
    device float* out            [[buffer(3)]],
    device float* stateOut       [[buffer(4)]],
    constant uint& tokens        [[buffer(5)]],
    constant uint& channels      [[buffer(6)]],
    constant uint& kernelSize    [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = tokens * channels;
    if (gid >= total || kernelSize == 0u) return;
    const uint token = gid / channels;
    const uint channel = gid % channels;
    float sum = 0.0f;
    for (uint lag = 0; lag < kernelSize; ++lag) {
        const int position = int(token) + int(kernelSize) - 1 - int(lag);
        float value = 0.0f;
        if (position >= int(kernelSize) - 1) {
            value = x[(position - (int(kernelSize) - 1)) * channels + channel];
        } else {
            value = stateIn[position * channels + channel];
        }
        sum += value * weights[channel * kernelSize + (kernelSize - 1u - lag)];
    }
    out[gid] = (sum + x[gid]) / sqrt(float(kernelSize + 1u));
    if (token >= tokens - (kernelSize - 1)) {
        // Last kernelSize-1 inputs become the next state.
        const uint history = tokens - (kernelSize - 1);
        const uint index = (token - history) * channels + channel;
        stateOut[index] = x[gid];
    }
}
