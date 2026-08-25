Here is the kernel so far:

```metal
kernel void reduce(device float *out [[buffer(0)]],
                   device const float *in [[buffer(1)]],
                   uint tid [[thread_position_in_threadgroup]]) {
    out[tid] = in[tid] * 2.0f;
