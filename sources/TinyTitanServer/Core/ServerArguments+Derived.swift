import TinyTitanCore
import Foundation
import TinyTitanLib

// The values derived from the parsed flags. Split out of ServerArguments.swift
// (2026-11-24) under the 500-line-per-file rule as pure code motion.
extension ServerArguments {
    /// The server-wide level, however it was spelled: `--reasoning` wins,
    /// otherwise `--thinking` and `--reasoning-effort` say the same thing.
    public var requestedReasoningLevel: TinyTitanCore.ReasoningLevel {
        if let reasoningLevel { return reasoningLevel }
        guard thinkingMode.isEnabled else { return .off }
        return reasoningEffort.flatMap { TinyTitanCore.ReasoningLevel(rawValue: $0.rawValue) }
            ?? .on
    }

    /// Unloading implies deferring the first load — loading at boot only to
    /// drop it moments later is incoherent. Derived rather than folded into
    /// `lazyLoad` so the struct stays a faithful record of what was typed.
    public var managesResidency: Bool { lazyLoad || idleUnloadSeconds > 0 }

    /// Idle unloading discards the in-memory prefix cache with the session.
    /// With a disk cache configured the entries rehydrate on reload; without
    /// one, every unload costs a full cold prefill on the next request.
    public var unloadDiscardsWarmCache: Bool {
        idleUnloadSeconds > 0
            && promptCacheMode != .off
            && promptCacheDiskDirectory == nil
    }
}
