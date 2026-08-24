#include <metal_stdlib>
using namespace metal;

// Separable antialiased bicubic resize, in the same fixed-point arithmetic as
// `TorchBicubicResize`.
//
// The weights are built on the CPU — they depend only on the axis ratio, so
// there are `dst` of them per axis, not per pixel — and applied here. Every
// step mirrors the CPU implementation exactly: Int16 weights, an Int32
// accumulator seeded with the rounding term, an arithmetic shift by the axis
// precision, and a clamp to 0...255. That is what makes the GPU output
// byte-identical rather than merely close, which in turn is what lets the CPU
// path stay the oracle in tests.
//
// Two passes, horizontal then vertical, because a separable filter costs
// `src*dst_w + dst_w*dst_h` taps rather than the product.

struct VisionResizeParams {
    uint sourceWidth;
    uint sourceHeight;
    uint destinationWidth;
    uint destinationHeight;
    uint sourceRowBytes;
    uint destinationRowBytes;
    uint tapStride;    // weights per output index, zero-padded
    int  precision;    // fixed-point shift
};

inline uchar apply_taps(int accumulator, int precision) {
    int value = accumulator >> precision;
    return (uchar)clamp(value, 0, 255);
}

kernel void vision_resize_horizontal(
    device const uchar *source [[buffer(0)]],
    device uchar *destination [[buffer(1)]],
    device const int *starts [[buffer(2)]],
    device const short *weights [[buffer(3)]],
    constant VisionResizeParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.destinationWidth || gid.y >= params.sourceHeight) { return; }

    const int rounding = 1 << (params.precision - 1);
    const uint start = (uint)starts[gid.x];
    const device short *tap = weights + gid.x * params.tapStride;
    const device uchar *inputRow = source + gid.y * params.sourceRowBytes;
    device uchar *outputRow = destination + gid.y * params.destinationRowBytes;

    int accumulator[4] = { rounding, rounding, rounding, rounding };
    for (uint index = 0; index < params.tapStride; ++index) {
        const int weight = (int)tap[index];
        if (weight == 0) { continue; }
        const device uchar *pixel = inputRow + (start + index) * 4;
        accumulator[0] += (int)pixel[0] * weight;
        accumulator[1] += (int)pixel[1] * weight;
        accumulator[2] += (int)pixel[2] * weight;
        accumulator[3] += (int)pixel[3] * weight;
    }
    device uchar *out = outputRow + gid.x * 4;
    out[0] = apply_taps(accumulator[0], params.precision);
    out[1] = apply_taps(accumulator[1], params.precision);
    out[2] = apply_taps(accumulator[2], params.precision);
    out[3] = apply_taps(accumulator[3], params.precision);
}

kernel void vision_resize_vertical(
    device const uchar *source [[buffer(0)]],
    device uchar *destination [[buffer(1)]],
    device const int *starts [[buffer(2)]],
    device const short *weights [[buffer(3)]],
    constant VisionResizeParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    // One thread per output byte, as the CPU pass walks `sourceWidth * 4`.
    const uint columnBytes = params.sourceWidth * 4;
    if (gid.x >= columnBytes || gid.y >= params.destinationHeight) { return; }

    const int rounding = 1 << (params.precision - 1);
    const uint start = (uint)starts[gid.y];
    const device short *tap = weights + gid.y * params.tapStride;

    int accumulator = rounding;
    for (uint index = 0; index < params.tapStride; ++index) {
        const int weight = (int)tap[index];
        if (weight == 0) { continue; }
        accumulator += (int)source[(start + index) * params.sourceRowBytes + gid.x]
            * weight;
    }
    destination[gid.y * params.destinationRowBytes + gid.x] =
        apply_taps(accumulator, params.precision);
}
