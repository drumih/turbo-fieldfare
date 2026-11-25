import Foundation
import TinyTitanCore

// The server's `--help` text.
//
// Split out of `ServerArguments.swift` (2026-09-28) under the 500-line-per-
// file rule (Task 8 of the cleanup runbook) as pure code motion. It
// interpolates the help spellings below, which are derived from the validators
// that enforce them.
extension ServerArguments {

    public static let usage = """
        usage: TinyTitanServer --model <completed .ssdai directory> [options]
               TinyTitanServer --models-dir <dir> --model <id or dir> [options]
               TinyTitanServer --catalog --models-dir <dir>

          --model <dir>          Required model directory. With --models-dir, the
                                 model loaded first: a catalog id or a directory.
          --models-dir <dir>     Serve every model under dir -- GPU installs and
                                 CPU snapshots alike -- keeping one resident. A
                                 request naming another catalog model waits for
                                 in-flight generations, unloads the resident model
                                 and loads the named one. /v1/models lists them.
          --catalog              With --models-dir: print the catalog as JSON and
                                 exit without loading anything.
          --reasoning <level>    Server-wide reasoning level: off, on, minimal, low,
                                 medium, high, xhigh or max, applied to whichever
                                 model is loaded. A model without that level gets
                                 the closest it has: an effort on an on/off model
                                 is on, on for an effort model is its template's
                                 default effort (extra high for Qwen3.8), off is
                                 always off. Replaces --thinking and
                                 --reasoning-effort, which keep working.
          --mtp-model <dir>      Optional native Qwen/Ornith MTP sidecar directory.
          --mtp-memory-mib <MiB> Strict incremental MTP budget, 256...512
                                 (default 384).
          --port <1...65535>     Loopback port (default 8080).
          --model-id <id>        API model identifier (default derived from the
                                 installed model manifest).
          --max-context <tokens> One of the native rungs:
                                 \(ServerArguments.supportedContextRungsHelp).
                                 Default 262144. With YaRN:
                                 \(ServerArguments.supportedYaRNContextRungsHelp)
                                 (default 1048576).
          --rope-scaling <mode>  Context scaling: none or yarn (default none).
          --queue-limit <count>  Maximum queued requests (default 4).
          --max-concurrent-sequences <count>
                                 Generations served at once: a power of two from 1
                                 to 256 (default 1). Requests beyond this plus
                                 --queue-limit are shed with 429. Above 1 each
                                 sequence holds its own KV cache (so memory use
                                 rises) and answers take longer, because one GPU is
                                 shared; the prompt cache is off above 1. The width
                                 actually built is clamped to what memory allows,
                                 and the log says so when it is.
          --prompt-cache-mode <off|single-prefix|multi-prefix>
                                 Prompt KV reuse mode (default multi-prefix).
          --prompt-cache-entries <count>
                                 Maximum retained prefixes, 1...64 (default 4).
          --prompt-cache-memory-mib <MiB>
                                 RAM snapshot budget, 0...4096 (default 256,
                                 or 2048 on a machine with 48 GiB or more).
          --prompt-cache-disk <dir>
                                 Optional persistent SSD cache directory.
          --prompt-cache-disk-mib <MiB>
                                 SSD snapshot budget, 0...65536 (default 8192).
          --calibration-profile <path>
                                 Apply a measured tuning profile where no
                                 explicit flag wins (explicit flags always
                                 win). Knobs with no runtime switch are logged,
                                 not applied.
          --prefill-chunk <tokens>
                                 Prefill chunk size, one of
                                 \(ServerArguments.prefillChunkTokensHelp)
                                 (default 4096 for supported 35B-A3B text
                                 models).
          --kv-bits <4|8|16>     KV-cache storage precision (default 8).
          --host-kv-window <tokens>
                                 Stream full-attention KV from a host arena
                                 through this many VRAM rows per layer-slot.
                                 Off by default; requires a sparse (QSA)
                                 model for contexts past the window.
          --thinking <off|on>    Ornith/Qwen reasoning mode (default off, or
                                 TINYTITAN_THINKING_MODE). The model does not expose
                                 low/medium/high effort levels.
          --reasoning-effort <low|medium|high|xhigh>
                                 Reasoning-effort level (default unset, or
                                 TINYTITAN_REASONING_EFFORT). Requires --thinking on
                                 and a model family whose chat template defines
                                 effort levels (Qwen3.8-Flash-Next); Ornith 1.5
                                 and Qwen 3.6 reject it.
          --expert-cache-slots <count>
                                 Routed-expert cache slots per layer:
                                 \(ServerArguments.expertCacheSlotsHelp).
                                 The default is derived from the model
                                 profile's tuned budget, not fixed.
                                 Environment override:
                                 TINYTITAN_EXPERT_CACHE_SLOTS.
          --ram-budget <size>    Resident-memory target for the whole server, e.g.
                                 4G, 8G, 16G. Minimum 4G. The routed-expert cache
                                 gets what is left after the resident weights and
                                 the runtime (about 3.7G on a Qwen3.8 4-bit
                                 install), and the slot count is the largest
                                 supported rung that fits, so the process stays
                                 under the number given. Below 4G the target cannot
                                 be honoured at all -- the weights plus the 8-slot
                                 minimum cache are already about 4.7G -- so it is
                                 refused rather than silently overshot.
                                 With no flag the install's profile names the *cache*
                                 budget instead -- the measured optimum, which is not
                                 a process target. --expert-cache-slots overrides
                                 both. Expert reads bypass the page cache and have
                                 no fallback, so a smaller cache is markedly slower.
          --lazy-load            Bind the port immediately and defer the model load
                                 to the first inference request (default off).
          --idle-unload-seconds <n>
                                 Release the model weights after n seconds with no
                                 requests, 0...86400 (default 0, disabled). The
                                 next request reloads transparently. Implies
                                 --lazy-load. Pair with --prompt-cache-disk, since
                                 unloading discards the in-memory prefix cache.
          --cpu                  Serve on the CPU from an affine snapshot instead
                                 of the GPU from an install. --model then points at
                                 a snapshot directory. For models small enough that
                                 a GPU is not the point: a 2B runs at about twenty
                                 tokens a second on the performance cores and
                                 leaves the GPU entirely free.
          --no-cpu-resident      With --cpu, leave residency to the page cache
                                 instead of faulting the snapshot in at startup.
                                 Only worth it on a machine too small to hold the
                                 model, where the alternative is swapping.
          --help                 Show this help.
        """

    /// The accepted slot counts, spelled from the validator that enforces them.
    ///
    /// The help text used to list them by hand and had already drifted: it
    /// stopped at 128 while `RuntimeConfiguration.allowedExpertCacheSlots`
    /// accepts 40, 48, 112, 160, 192 and 256 as well. Spelled from the source
    /// of truth so it cannot drift again.
    static var expertCacheSlotsHelp: String {
        RuntimeLimits.allowedExpertCacheSlots
            .map(String.init).joined(separator: ", ")
    }

    /// The accepted native context windows, spelled from the gate that enforces
    /// them. `--max-context` takes one of these rungs, not any token count
    /// between the first and the last, and the help used to write it as a range.
    static var supportedContextRungsHelp: String {
        RuntimeLimits.supportedContextTokens
            .map(String.init).joined(separator: ", ")
    }

    /// The accepted YaRN rungs, spelled from the gate that enforces them. It was
    /// a literal in the refusal until 2026-10-18, which is how the 2M rung
    /// landed with a message still naming two values.
    static var supportedYaRNContextRungsHelp: String {
        RuntimeLimits.supportedYaRNContextTokens
            .map(String.init).joined(separator: ", ")
    }

    /// The accepted prefill chunk sizes, spelled from the gate that enforces
    /// them -- the hand-written list in the help text is how
    /// `expertCacheSlotsHelp` above came to be derived instead of typed.
    static var prefillChunkTokensHelp: String {
        RuntimeLimits.allowedPrefillChunkTokens
            .map(String.init).joined(separator: ", ")
    }

    /// The RAM tier of the prompt cache that a launch without the flag gets.
    ///
    /// Sized by the machine the way the expert-cache budget is. Prompt state
    /// costs ~0.74 MiB per prefix token on qwen38 4-bit -- measured on a live
    /// generation, a 152-token snapshot at `state_bytes=117_442_560` = 112 MiB,
    /// all of it served from this tier -- so the flat 256 MiB default holds
    /// about 350 tokens and pushes anything longer onto the SSD tier: a second
    /// streaming consumer beside the expert reads that already own the disk, on
    /// a Mac that may have 40 GiB spare above it. 2 GiB is the same rule at a
    /// big machine's size: roughly 2,800 prefix tokens resident. It does not
    /// scale further, because past a few GiB the limits that bind are the
    /// store's own 4 GiB per-snapshot capture ceiling and the retained entry
    /// count rather than RAM; an operator who wants more names it, and the
    /// flag's range reaches 4 GiB.
    /// Below 48 GiB the tuned 256 MiB stays, because every machine this
    /// project has measured lives there and it is what those measurements
    /// priced.
    public static func defaultPromptCacheMemoryMiB(
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        physicalMemory >= 48 << 30 ? 2_048 : 256
    }
}
