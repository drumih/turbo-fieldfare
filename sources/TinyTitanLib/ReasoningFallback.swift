import TinyTitanCore

// Fitting one reasoning level to a model that exposes different ones.
//
// Moved out of `ReasoningFallback` in `TinyTitanServerCore`'s
// `ModelRouter.swift` (2026-10-02, phase A1 of `docs/plan-embedded-library.md`)
// as a pure declaration move. The request validator in this target applies the
// same mapping the server-wide `--reasoning` level does, so a request and the
// flag cannot disagree about what a model does with a level it lacks.
//
// Only `effectiveLevel` moves: it is arithmetic over the engine's
// `ReasoningLevel` and needs nothing from the server. The sibling
// `choice(for:requested:)` reads the server's catalog, so it stays there.

/// Fits one server-wide reasoning level to models that expose different ones.
///
/// The level is chosen once for the server and the models under it differ:
/// Qwen 3.6 has an on/off switch, Qwen3.8-Flash-Next has effort levels and no
/// bare "on". Refusing to load a model because the level does not map exactly
/// would make `--reasoning` useless with a mixed catalog, so each model gets
/// the closest thing its template defines.
package enum ReasoningFallback {
    /// `whenOn` is what the model's template does when thinking is switched
    /// on with no effort named; the server's catalog supplies it.
    package static func effectiveLevel(
        _ requested: ReasoningLevel,
        supported: [ReasoningLevel],
        whenOn: ReasoningLevel? = nil
    ) -> ReasoningLevel {
        if supported.contains(requested) || requested == .off { return requested }
        let efforts = supported.filter { $0 != .off && $0 != .on }
        // An on/off model: any effort means "think".
        guard !efforts.isEmpty else { return supported.contains(.on) ? .on : .off }
        // "On" for an effort model: the template's own default, extra high
        // for Qwen3.8. That is what --thinking on has always loaded on a
        // single-model server, and the same flag must not think less because
        // the server was started with a catalog. The middle effort is left
        // only for a caller that cannot say what the template does.
        if requested == .on {
            if let whenOn, efforts.contains(whenOn) { return whenOn }
            return efforts[(efforts.count - 1) / 2]
        }
        // An effort the model lacks: the nearest one it has, ties to the
        // cheaper, since a client that wanted more can ask for it by name.
        let order = ReasoningLevel.allCases
        let rank = { (level: ReasoningLevel) in order.firstIndex(of: level) ?? 0 }
        let target = rank(requested)
        return efforts.min { lhs, rhs in
            let left = abs(rank(lhs) - target)
            let right = abs(rank(rhs) - target)
            return left == right ? rank(lhs) < rank(rhs) : left < right
        } ?? .on
    }
}
