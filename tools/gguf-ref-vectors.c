// Emits the vectors tests/TinyTitan/Infrastructure/ModelIO/GGUFDequantVectors.txt
// is built from: deterministic block bytes, decoded by ggml's own
// dequantize_row_* so the Swift decoder is checked against the reference and
// not against itself.
//
// Build against a llama.cpp checkout whose ggml library is configured and built.
// Either link the shared library of a default (Metal) build:
//
//   clang -O1 -I <llama.cpp>/ggml/include -I <llama.cpp>/ggml/src \
//     -o gguf-ref-vectors tools/gguf-ref-vectors.c \
//     -L <llama.cpp>/build/bin -lggml-base -Wl,-rpath,<llama.cpp>/build/bin
//
// or the object files of a CPU-only build:
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

    { // q2_K: 2 blocks, d at 0, dmin at 2
        uint8_t buf[168];
        for (int i = 0; i < 168; i++) buf[i] = pat(i);
        set_d(buf + 0, 80, 0.002f);
        set_d(buf + 0, 82, 0.001f);
        set_d(buf + 84, 80, -0.003f);
        set_d(buf + 84, 82, 0.0005f);
        dequantize_row_q2_K((const block_q2_K *)buf, y, 512);
        emit("q2_K", buf, sizeof buf, y, 512);
    }
    { // q4_K: 2 blocks, d at 0, dmin at 2
        uint8_t buf[288];
        for (int i = 0; i < 288; i++) buf[i] = pat(i);
        set_d(buf + 0, 0, 0.004f);
        set_d(buf + 0, 2, 0.0015f);
        set_d(buf + 144, 0, -0.006f);
        set_d(buf + 144, 2, 0.0007f);
        dequantize_row_q4_K((const block_q4_K *)buf, y, 512);
        emit("q4_K", buf, sizeof buf, y, 512);
    }
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
    { // iq2_s: 2 blocks, d at 0, qs/signs at 2, qh at 66, scales at 74
        uint8_t buf[164];
        for (int i = 0; i < 164; i++) buf[i] = pat(i);
        set_d(buf + 0, 0, 0.02f);
        set_d(buf + 82, 0, -0.015f);
        dequantize_row_iq2_s((const block_iq2_s *)buf, y, 512);
        emit("iq2_s", buf, sizeof buf, y, 512);
    }
    { // iq3_xxs: 2 blocks, d at 0, qs at 2
        uint8_t buf[196];
        for (int i = 0; i < 196; i++) buf[i] = pat(i);
        set_d(buf + 0, 0, 0.012f);
        set_d(buf + 98, 0, -0.009f);
        dequantize_row_iq3_xxs((const block_iq3_xxs *)buf, y, 512);
        emit("iq3_xxs", buf, sizeof buf, y, 512);
    }
    { // iq4_xs: 2 blocks, d at 0, scales_h at 2, scales_l at 4, qs at 8
        uint8_t buf[272];
        for (int i = 0; i < 272; i++) buf[i] = pat(i);
        set_d(buf + 0, 0, 0.004f);
        set_d(buf + 136, 0, -0.006f);
        dequantize_row_iq4_xs((const block_iq4_xs *)buf, y, 512);
        emit("iq4_xs", buf, sizeof buf, y, 512);
    }
    { // nvfp4: 2 blocks of 64, d[4] (UE4M3) at 0, qs[32] at 4; the
      // pattern's scale bytes cover subnormal, normal and the 0x7F zero code
        uint8_t buf[72];
        for (int i = 0; i < 72; i++) buf[i] = pat(i);
        buf[0] = 0x38; buf[36] = 0x7F; buf[37] = 0x05;
        dequantize_row_nvfp4((const block_nvfp4 *)buf, y, 128);
        emit("nvfp4", buf, sizeof buf, y, 128);
    }
    { // mxfp4: 2 blocks of 32, e (E8M0) at 0, qs[16] at 1
        uint8_t buf[34];
        for (int i = 0; i < 34; i++) buf[i] = pat(i);
        buf[0] = 127; buf[17] = 121;
        dequantize_row_mxfp4((const block_mxfp4 *)buf, y, 64);
        emit("mxfp4", buf, sizeof buf, y, 64);
    }
    { // q5_K: 2 blocks, d at 0, dmin at 2, scales[4..15], qh[16..47], qs[48..175]
        uint8_t buf[352];
        for (int i = 0; i < 352; i++) buf[i] = pat(i);
        set_d(buf + 0, 0, 0.004f);
        set_d(buf + 0, 2, 0.0015f);
        set_d(buf + 176, 0, -0.005f);
        set_d(buf + 176, 2, 0.0009f);
        dequantize_row_q5_K((const block_q5_K *)buf, y, 512);
        emit("q5_K", buf, sizeof buf, y, 512);
    }
    { // iq2_xxs: 2 blocks, d at 0, qs (uint16 x 32) at 2
        uint8_t buf[132];
        for (int i = 0; i < 132; i++) buf[i] = pat(i);
        set_d(buf + 0, 0, 0.02f);
        set_d(buf + 66, 0, -0.015f);
        dequantize_row_iq2_xxs((const block_iq2_xxs *)buf, y, 512);
        emit("iq2_xxs", buf, sizeof buf, y, 512);
    }
    { // iq2_xs: 2 blocks, d at 0, qs at 2, scales at 66
        uint8_t buf[148];
        for (int i = 0; i < 148; i++) buf[i] = pat(i);
        set_d(buf + 0, 0, 0.02f);
        set_d(buf + 74, 0, -0.012f);
        dequantize_row_iq2_xs((const block_iq2_xs *)buf, y, 512);
        emit("iq2_xs", buf, sizeof buf, y, 512);
    }
    { // iq1_s: 2 blocks, d at 0, qs at 2, qh (uint16 x 8) at 34
        uint8_t buf[100];
        for (int i = 0; i < 100; i++) buf[i] = pat(i);
        set_d(buf + 0, 0, 0.03f);
        set_d(buf + 50, 0, -0.02f);
        dequantize_row_iq1_s((const block_iq1_s *)buf, y, 512);
        emit("iq1_s", buf, sizeof buf, y, 512);
    }
    return 0;
}
