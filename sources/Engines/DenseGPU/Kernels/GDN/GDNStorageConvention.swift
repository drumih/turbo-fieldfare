/// How a checkpoint stores the gated delta net tensors the delta kernels read.
///
/// The engine's own installs keep the Hugging Face layout. A GGUF written by
/// llama.cpp's converter (`conversion/qwen.py`, `_LinearAttentionVReorderBase`,
/// and the `ssm_a` rename) differs in two ways that change the arithmetic, not
/// just the bytes, so the kernels have to be told which one they are reading.
package enum GDNStorageConvention: Sendable, Equatable {
    /// Hugging Face: `A_log` raw, value heads grouped by key head.
    case engine
    /// llama.cpp GGUF: `ssm_a` is `-exp(A_log)`, value heads tiled.
    case llamaCppGGUF

    /// The decay multiplier is already `-exp(A_log)`: the kernel computes
    /// `exp(a * softplus(.))` instead of `exp(-exp(a) * softplus(.))`.
    /// Measured on the UD-IQ3_XXS checkpoint: `blk.0.ssm_a` holds values such
    /// as -9.05 and -77.0, which are not log-space.
    var aLogPreexponentiated: Bool { self == .llamaCppGGUF }

    /// Value head h reads key head `h % Hk` (tiled) rather than
    /// `h / (Hv / Hk)` (grouped). The converter permutes the value rows of
    /// `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt` and
    /// `ssm_conv1d`, and the columns of `ssm_out`, consistently - so only the
    /// key-head pairing inside the kernel has to change.
    var valueHeadsTiled: Bool { self == .llamaCppGGUF }
}
