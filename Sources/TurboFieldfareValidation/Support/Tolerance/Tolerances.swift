/// Tolerance dictionary modeled on MLX (`test_fast.py:80`). Each constant is
/// a *relative* error bar against an FP32 reference, sized to the accumulation
/// depth and dtype of the kernel under test.
public enum Tolerance {
    /// FP32 identity / pass-through. Bit-exact-ish.
    public static let identity: Float = 1e-5

    /// FP16 single-reduction kernels (RMSNorm, one GEMV, short-axis softmax).
    /// Comfortably above the FP16-accumulation-in-FP32 ULP floor.
    public static let fp16Reduction: Float = 5e-3

    /// FP16 chained reductions: multi-stage MoE, attention's softmax + matmul
    /// composition, every block-level fusion. Absorbs FP16 round-tripping at
    /// each intermediate.
    public static let fp16ChainedReduction: Float = 1e-2

    /// Quantization-aware comparisons. Callers usually override with a
    /// mathematically derived bound (e.g. `|w - w_hat| <= |scales|`).
    public static let quantInt4: Float = 1.5e-3
    public static let quantInt8: Float = 1e-3
}
