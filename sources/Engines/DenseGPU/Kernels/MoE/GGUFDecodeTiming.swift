import Foundation

/// Per-token phase timings of the as-is GGUF decoder, reported to stderr when
/// `TINYTITAN_GGUF_TRACE=1`. A caller's stdout belongs to the caller, so the
/// report never goes there.
struct GGUFDecodeTiming {
    static let enabled = ProcessInfo.processInfo.environment["TINYTITAN_GGUF_TRACE"] == "1"

    private(set) var phaseNanos: [String: UInt64] = [:]
    private(set) var attentionWaits: [(layer: Int, nanos: UInt64)] = []
    /// GPU time of the per-layer command buffers: since the decoder carries
    /// one buffer across layers, each holds the previous layer's routed
    /// experts and this layer's mixer, router and shared expert.
    var gpuAttentionSeconds: Double = 0
    /// GPU time of the last buffer: the last layer's routed experts and the
    /// output head.
    var gpuRoutedSeconds: Double = 0

    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    mutating func add(_ phase: String, since start: UInt64) {
        phaseNanos[phase, default: 0] &+= Self.now() &- start
    }

    mutating func attentionWait(layer: Int, since start: UInt64) {
        attentionWaits.append((layer, Self.now() &- start))
    }

    /// The report: GPU totals, the first four and the last per-layer attention
    /// waits (where the cold start shows), then every phase by cost.
    func report(to stream: FileHandle = .standardError) {
        func ms(_ nanos: UInt64, _ digits: Int) -> String {
            String(format: "%.\(digits)f", Double(nanos) / 1e6)
        }
        var lines =
            "  GPU: layers \(String(format: "%.0f", gpuAttentionSeconds * 1000)) ms "
            + "(each the previous layer's routed experts + this layer's mixer), "
            + "tail \(String(format: "%.0f", gpuRoutedSeconds * 1000)) ms\n"
        var shown = Array(attentionWaits.prefix(4))
        if let last = attentionWaits.last, attentionWaits.count > 4 { shown.append(last) }
        for wait in shown {
            lines += "  wait(attn) layer \(wait.layer): \(ms(wait.nanos, 0)) ms\n"
        }
        for (name, nanos) in phaseNanos.sorted(by: { $0.value > $1.value }) {
            lines += "  \(name): \(ms(nanos, 1)) ms\n"
        }
        stream.write(Data(lines.utf8))
    }
}
