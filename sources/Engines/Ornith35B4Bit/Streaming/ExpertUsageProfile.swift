import Foundation

/// A portable record of which experts a model has actually routed to.
///
/// This is Project Maya's `expert-profile.bin` idea, rebuilt for this engine:
/// the counts a decode leaves behind are written beside the install and read
/// back at the next start, where they seed the streamer's aging-LFU score. The
/// effect is that a restart begins with the experts a conversation used last
/// time already priced above the ones it never touched, instead of every slot
/// starting at zero and the first tokens re-learning the same working set.
///
/// The format is deliberately plain: a fixed header and `layers * experts`
/// little-endian counts. Counts are decayed on load (halved, then capped) so a
/// profile follows a conversation that changes topic instead of freezing the
/// first session's routing forever.
public struct ExpertUsageProfile: Sendable, Equatable {
    static let magic: UInt32 = 0x5054_5445  // "ETTP", little-endian
    static let version: UInt32 = 1
    static let headerBytes = 16
    /// A seeded count never exceeds this. Prefill and decode both add to the
    /// score, so an uncapped historical count would outrank every current use.
    public static let seedCap = 512
    /// Saved counts saturate here. The score is only an ordering.
    public static let storedCap: UInt32 = 4096

    public let layers: Int
    public let expertsPerLayer: Int
    /// Row-major `layers * expertsPerLayer` counts.
    public private(set) var counts: [UInt32]

    public init(layers: Int, expertsPerLayer: Int) {
        self.layers = max(0, layers)
        self.expertsPerLayer = max(0, expertsPerLayer)
        self.counts = [UInt32](
            repeating: 0, count: self.layers * self.expertsPerLayer)
    }

    private init(layers: Int, expertsPerLayer: Int, counts: [UInt32]) {
        self.layers = layers
        self.expertsPerLayer = expertsPerLayer
        self.counts = counts
    }

    public func count(layer: Int, expert: Int) -> UInt32 {
        guard layer >= 0, layer < layers,
            expert >= 0, expert < expertsPerLayer
        else { return 0 }
        return counts[layer * expertsPerLayer + expert]
    }

    public mutating func setCount(layer: Int, expert: Int, to value: UInt32) {
        guard layer >= 0, layer < layers,
            expert >= 0, expert < expertsPerLayer
        else { return }
        counts[layer * expertsPerLayer + expert] = min(value, Self.storedCap)
    }

    public mutating func setCounts(layer: Int, _ values: [Int]) {
        guard layer >= 0, layer < layers else { return }
        for expert in 0..<min(values.count, expertsPerLayer) {
            setCount(
                layer: layer, expert: expert,
                to: UInt32(clamping: max(0, values[expert])))
        }
    }

    /// The LFU prior for one layer: half the stored count, capped.
    ///
    /// Halving on every load is the aging rule. A workload that repeats keeps
    /// its experts at the cap; a workload that moves on lets the old counts
    /// fall behind each session instead of pinning them for good.
    public func seedCounts(layer: Int) -> [Int]? {
        guard layer >= 0, layer < layers else { return nil }
        let base = layer * expertsPerLayer
        return (0..<expertsPerLayer).map { expert in
            min(Int(counts[base + expert] / 2), Self.seedCap)
        }
    }

    /// True when every count is zero, i.e. the profile carries no information.
    public var isEmpty: Bool {
        counts.allSatisfy { $0 == 0 }
    }

    // MARK: - Wire format

    public func encoded() -> Data {
        var data = Data(capacity: Self.headerBytes + counts.count * 4)
        append(Self.magic, to: &data)
        append(Self.version, to: &data)
        append(UInt32(layers), to: &data)
        append(UInt32(expertsPerLayer), to: &data)
        for value in counts { append(value, to: &data) }
        return data
    }

    public static func decode(
        _ data: Data, layers: Int, expertsPerLayer: Int
    ) -> ExpertUsageProfile? {
        guard data.count >= headerBytes else { return nil }
        let words = data.withUnsafeBytes { raw -> [UInt32] in
            var result: [UInt32] = []
            result.reserveCapacity(data.count / 4)
            var offset = 0
            while offset + 4 <= data.count {
                result.append(
                    raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
                offset += 4
            }
            return result
        }
        guard words.count >= 4, words[0] == magic, words[1] == version else {
            return nil
        }
        let storedLayers = Int(words[2])
        let storedExperts = Int(words[3])
        guard storedLayers == layers, storedExperts == expertsPerLayer,
            words.count >= 4 + layers * expertsPerLayer
        else { return nil }
        let body = Array(words[4..<(4 + layers * expertsPerLayer)])
        return ExpertUsageProfile(
            layers: layers, expertsPerLayer: expertsPerLayer, counts: body)
    }

    public static func load(
        url: URL, layers: Int, expertsPerLayer: Int
    ) -> ExpertUsageProfile? {
        guard layers >= 0, expertsPerLayer >= 0 else { return nil }
        // The file is sized by the geometry, so the read is too: a foreign or
        // corrupt file cannot make this allocate more than one profile.
        let expected = headerBytes + layers * expertsPerLayer * 4
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: expected),
            data.count == expected
        else { return nil }
        return decode(data, layers: layers, expertsPerLayer: expertsPerLayer)
    }

    public func write(to url: URL) throws {
        try encoded().write(to: url, options: .atomic)
    }

    private func append(_ value: UInt32, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }
}
