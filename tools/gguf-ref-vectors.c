// Emits the vectors tests/TinyTitan/Infrastructure/ModelIO/GGUFDequantVectors.txt
// is built from: deterministic block bytes, decoded by ggml's own
// dequantize_row_* so the Swift decoder is checked against the reference and
// not against itself.
//
// Build against a llama.cpp checkout whose ggml library is configured and built:
//
//   cmake -S <llama.cpp> -B <llama.cpp>/build -DLLAMA_CURL=OFF -DGGML_METAL=OFF
//   cmake --build <llama.cpp>/build --target ggml -j 8
//   clang -O1 -I <llama.cpp>/ggml/include -I <llama.cpp>/ggml/src \
//     -o /tmp/gguf-ref-vectors tools/gguf-ref-vectors.c \
//     <llama.cpp>/build/ggml/src/CMakeFiles/ggml-base.dir/ggml-quants.c.o \
//     <llama.cpp>/build/ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o -Wl,-dead_strip
//   /tmp/gguf-ref-vectors > tests/TinyTitan/Infrastructure/ModelIO/GGUFDequantVectors.txt
//
// Re-run only when the ggml block formats themselves change, and update the
// commit named in the fixture header when you do.
#include "ggml.h"
#include "ggml-quants.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static uint8_t pat(int i) { return (uint8_t)((i * 37 + 11) & 0xFF); }

static void emit(const char *name, const void *bytes, size_t n, const float *y, int count) {
    const uint8_t *b = (const uint8_t *)bytes;
    printf("%s|", name);
    for (size_t i = 0; i < n; i++) printf("%02x", b[i]);
    printf("|");
    for (int i = 0; i < count; i++) {
        uint32_t u; memcpy(&u, &y[i], 4); printf("%08x", u);
    }
    printf("\n");
}

static void set_d(void *block, size_t off, float value) {
    ggml_fp16_t d = ggml_fp32_to_fp16(value);
    memcpy((uint8_t *)block + off, &d, 2);
}

int main(void) {
    float y[512];

    { // q6_K: 2 blocks, d at 208
        uint8_t buf[420];
        for (int i = 0; i < 420; i++) buf[i] = pat(i);
        set_d(buf + 0, 208, 0.002f);
        set_d(buf + 210, 208, -0.0035f);
        dequantize_row_q6_K((const block_q6_K *)buf, y, 512);
        emit("q6_K", buf, sizeof buf, y, 512);
    }
    { // q8_0: 2 blocks, d at 0
        uint8_t buf[68];
        for (int i = 0; i < 68; i++) buf[i] = pat(i);
        set_d(buf + 0, 0, 0.011f);
        set_d(buf + 34, 0, -0.007f);
        dequantize_row_q8_0((const block_q8_0 *)buf, y, 64);
        emit("q8_0", buf, sizeof buf, y, 64);
    }
    { // q2_0: 2 blocks, d at 0
        uint8_t buf[36];
        for (int i = 0; i < 36; i++) buf[i] = pat(i);
        set_d(buf + 0, 0, 0.05f);
        set_d(buf + 18, 0, 0.125f);
        dequantize_row_q2_0((const block_q2_0 *)buf, y, 128);
        emit("q2_0", buf, sizeof buf, y, 128);
    }
    { // iq4_nl: 2 blocks, d at 0
        uint8_t buf[36];
        for (int i = 0; i < 36; i++) buf[i] = pat(i);
        set_d(buf + 0, 0, 0.03f);
        set_d(buf + 18, 0, -0.02f);
        dequantize_row_iq4_nl((const block_iq4_nl *)buf, y, 64);
        emit("iq4_nl", buf, sizeof buf, y, 64);
    }
    { // iq4_xs: 2 blocks, d at 0, scales_h at 2, scales_l at 4, qs at 8
        uint8_t buf[272];
        for (int i = 0; i < 272; i++) buf[i] = pat(i);
        set_d(buf + 0, 0, 0.004f);
        set_d(buf + 136, 0, -0.006f);
        dequantize_row_iq4_xs((const block_iq4_xs *)buf, y, 512);
        emit("iq4_xs", buf, sizeof buf, y, 512);
    }
    return 0;
}
