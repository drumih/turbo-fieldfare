/// Deliberate-pathology shape sweeps modeled on MLX `test_qmm:213`.
///
/// The kernels' threadgroup sizes and SIMD widths are typically multiples
/// of 32; many bugs only surface when the dispatched extent is just past or
/// just shy of those boundaries. New parameterized tests pull shapes from
/// these vectors to ensure we test the edges, not just the nice numbers.
public enum OffByMultiples {
    /// Multiples of the INT4 group size 64. Useful when the kernel
    /// preconditions `N % 64 == 0`.
    public static let multiplesOfGroup: [Int] = [
        64, 128, 192, 256, 320, 512, 704, 1024, 2816
    ]
}
