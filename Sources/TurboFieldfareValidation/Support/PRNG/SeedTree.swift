/// One-root-many-streams pattern, modeled on `mx.random.key` + `mx.random.split`.
/// A test seeds a `SeedTree` once at the top, then derives independent
/// `SplitMix64` streams for each input it generates.
public struct SeedTree {
    public let root: UInt64

    public init(_ root: UInt64) {
        self.root = root
    }

    /// Derive a child PRNG by mixing the root with an FNV-1a hash of the label.
    public func key(_ label: String) -> SplitMix64 {
        var h: UInt64 = 0xCBF29CE484222325
        for byte in label.utf8 {
            h ^= UInt64(byte)
            h &*= 0x100000001B3
        }
        return SplitMix64(seed: root ^ h)
    }
}
