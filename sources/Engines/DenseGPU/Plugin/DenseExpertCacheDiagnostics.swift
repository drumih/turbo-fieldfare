// Provenance: copied from sources/TinyTitanLib/ExpertCacheDiagnostics.swift (TT-132) so the
// dense Qwen 3.5 plugin logs the same load lines without importing the library.
import Foundation

/// The load-time `ram profile` and `ram target` lines.
///
/// They are the operator's only record of what the engine allocated, so every
/// number describes the slot count actually in effect. The derivation is named
/// as `derived_slots=` only when an override replaced it: a line that printed
/// the derived rung while a different one was allocated read as evidence of a
/// configuration that never ran.
enum DenseExpertCacheDiagnostics {
    private static func gib(_ bytes: Double) -> String {
        String(format: "%.2fG", bytes / 1_073_741_824)
    }

    private static func slotClause(appliedSlots: Int, derivedSlots: Int) -> String {
        appliedSlots == derivedSlots
            ? "slots=\(appliedSlots)"
            : "slots=\(appliedSlots) derived_slots=\(derivedSlots)"
    }

    /// The no-flag case: the budget came from the model's profile, so the line
    /// says it is a cache budget rather than a ceiling on the whole process.
    package static func profileLine(
        appliedSlots: Int,
        derivedSlots: Int,
        cacheBytesPerSlot: Double,
        residentFloorBytes: Int
    ) -> String {
        let cache = Double(appliedSlots) * cacheBytesPerSlot
        let floor = Double(residentFloorBytes)
        return """
            TinyTitan ram profile cache=\(gib(cache)) \
            \(slotClause(appliedSlots: appliedSlots, derivedSlots: derivedSlots)) \
            resident_floor=\(gib(floor)) estimate=\(gib(floor + cache)) \
            (cache budget, not a process target)
            """
    }

    /// The `--ram-budget` case (the launcher spells it `--ram`): the number
    /// names the whole process, so a footprint above it gets a second line.
    /// That line says the cache is at its smallest only when the derivation set
    /// it; an override that pins more slots than the target affords has a
    /// different cause, and an operator who is told the cache is already
    /// minimal will not look at their own flag.
    package static func targetLines(
        targetBytes: Int,
        appliedSlots: Int,
        derivedSlots: Int,
        cacheBytesPerSlot: Double,
        residentFloorBytes: Int
    ) -> [String] {
        let cache = Double(appliedSlots) * cacheBytesPerSlot
        let floor = Double(residentFloorBytes)
        let estimate = floor + cache
        let line = """
            TinyTitan ram target=\(gib(Double(targetBytes))) cache=\(gib(cache)) \
            \(slotClause(appliedSlots: appliedSlots, derivedSlots: derivedSlots)) \
            resident_floor=\(gib(floor)) estimate=\(gib(estimate))
            """
        guard Double(targetBytes) < estimate else { return [line] }
        let warning =
            appliedSlots == derivedSlots
            ? """
            TinyTitan ram warning: \(gib(Double(targetBytes))) is below this install's \
            \(gib(estimate)) floor (\(gib(floor)) resident + the \(appliedSlots)-slot \
            minimum cache); the cache is already at its smallest.
            """
            : """
            TinyTitan ram warning: \(gib(Double(targetBytes))) is below this install's \
            \(gib(estimate)) footprint (\(gib(floor)) resident + the \(appliedSlots)-slot \
            cache named by --expert-cache-slots); the target alone allows \(derivedSlots) \
            slots, so the gap is the override, not the install.
            """
        return [line, warning]
    }

    /// The `[decode expert io]` line, in the shape
    /// `benchmark/expert_cache_slots.py` parses: expert I/O attributable to
    /// decode alone.
    ///
    /// The runner's stage line reports the whole request, prefill included, so
    /// this is the only decode-only attribution. The runner takes the baseline
    /// at the prefill/decode handover, which is why the line appears only when
    /// `TINYTITAN_DECODE_IO_TRACE=1`; `tokens` is 0 for a request that
    /// generated nothing, and then there is no per-token figure to state.
    package static func decodeIOLine(
        hits: UInt64,
        misses: UInt64,
        bytes: UInt64,
        tokens: Int
    ) -> String {
        let total = hits + misses
        let rate = total == 0 ? 0 : 100 * Double(hits) / Double(total)
        var line = "[decode expert io] hits \(hits) misses \(misses)"
        line += String(format: " (%.1f%% hit)", rate)
        line += String(format: " %.2f GiB", Double(bytes) / 1_073_741_824)
        if tokens > 0 {
            line += String(
                format: " = %.1f MiB/token",
                Double(bytes) / 1_048_576 / Double(tokens))
        }
        return line
    }
}
