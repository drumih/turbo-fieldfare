// The `--help` text and the validator-spelled help fragments it interpolates.
//
// Split out of Args.swift (2026-10-27) under the 500-line-per-file rule; the
// declarations are unchanged.
import TinyTitanCore

extension Args {
    /// The accepted slot counts, spelled from the validator that enforces them.
    ///
    /// The help text used to list them by hand and had already drifted: it
    /// stopped at 128 while `RuntimeLimits.allowedExpertCacheSlots`
    /// accepts 40, 48, 112, 160, 192 and 256 as well, so a caller reading
    /// `--help` would not know 160 was legal. Spelled from the source of truth
    /// so it cannot drift again.
    static var expertCacheSlotsHelp: String {
        RuntimeLimits.allowedExpertCacheSlots
            .map(String.init).joined(separator: ", ")
    }

    /// The accepted YaRN context rungs, spelled from the validator that enforces
    /// them — the help said "524288 or 1048576" until the 2M rung landed.
    static var yarnContextRungsHelp: String {
        RuntimeLimits.supportedYaRNContextTokens
            .map(String.init).joined(separator: ", ")
    }

    public static let usage = """
        TinyTitanCLI — local text generation for Qwen-family MoE and dense models

        usage: TinyTitanCLI --model <dir> (--prompt <string> | --messages-file <path>) [options]
               TinyTitanCLI calibrate (--model <dir> [--max-context <tokens>] [--out <path>]
                                       | --check <path>)
               TinyTitanCLI doctor [--json] [--models-dir <dir>]
               TinyTitanCLI catalog [--check] [--markdown] [--registry <path>]
               TinyTitanCLI plugin-check [--install <dir>] [--plugin <id>] [--json]

        command:
          calibrate                  Write or validate a per-machine tuning
                                    profile (the calibrate command's --help
                                    lists its own flags).
          catalog                    Render or check the one model registry
                                    (the catalog command's --help lists its
                                    own flags).
          plugin-check               Verify TPP/1 engine plugins and installs
                                    against the conformance contract.

        required:
          --model <dir>             Path to a .ssdai model directory.
          --prompt <string>         Raw-completion prompt.
          --messages-file <path>    JSON chat messages with role and content fields.

        options:
          --max-new <int>           Generated-token limit (default 1024).
          --max-context <int>       Native context limit, 1...262144 (default 262144).
                                    With YaRN: \(Args.yarnContextRungsHelp) (default 1048576).
          --rope-scaling <mode>     Context scaling: none or yarn (default none).
          --temperature <float>     Sampling temperature (0 = greedy). Default is
                                    the family's: 1.0 for Qwen3.8-Flash-Next,
                                    0.6 elsewhere.
          --top-k <int>             Top-k truncation, 1...256 (default 20; 0 = off).
          --top-p <float>           Nucleus truncation (default 0.95).
          --repetition-penalty <f>  Repetition penalty (default 1.0).
          --presence-penalty <f>    Presence penalty, -2...2 (default: the model's
                                    own row; 1.5 for Qwen3.8 outside thinking mode).
          --seed <uint64>           Deterministic sampling seed (default off).
          --stop <string>           Stop substring (repeatable).
          --rdadvise <mode>         Expert read-ahead advice: off, default,
                                    bounded, or adaptive. The default is
                                    `default`, which leaves advice ON; pass
                                    off to disable it.
          --ram-budget <size>       Resident-memory target for the whole
                                    process, e.g. 12G. The routed-expert cache
                                    gets what is left of it after the resident
                                    weights and the runtime, stepping down the
                                    slot ladder; the load prints the `ram
                                    target` line it resolved. Minimum 4G.
                                    Overrides the profile's tuned cache; an
                                    explicit --expert-cache-slots wins.
          --expert-cache-slots <n>  Routed-expert cache slots per layer:
                                    \(Self.expertCacheSlotsHelp). The default
                                    is derived from the model profile's tuned
                                    budget, not fixed; 64 is only the
                                    fallback when the manifest cannot be read.
                                    More slots raise the hit rate but use more
                                    memory.
          --calibration-profile <path>
                                    Apply a measured profile where no explicit
                                    flag wins (explicit flags always win).
                                    Unsupported profile knobs are logged, not
                                    applied.
          --engine <cpu|gpu>        Which engine serves a dense Qwen 3.5 install
                                    (both implement it). Default: automatic —
                                    the 7.0 variant's GPU engine. Other
                                    families are GPU-only and refuse cpu.
          --prefill-chunk <n|auto>  Prefill chunk tokens. Larger chunks reduce
                                    routed-expert file sweeps but use more GPU
                                    scratch. Allowed: 32, 64, 128, 256, 512,
                                    1024, 2048, 4096; auto covers the prompt with
                                    the smallest allowed chunk.
          --kv-bits <4|8|16>        KV-cache storage precision (default 8).
          --concise                 Inject the per-quantization concise-mode
                                    system prompt (answers without preamble,
                                    filler, or closing codas).
          --thinking <off|on>       Ornith/Qwen reasoning mode (default off).
                                    These models do not define effort levels.
          --reasoning-effort <lvl>  Reasoning-effort level: low, medium, high,
                                    or xhigh. Requires --thinking on and a model
                                    family whose chat template defines effort
                                    levels (Qwen3.8-Flash-Next, K2-Horizon);
                                    Ornith 1.5 and Qwen 3.6 reject it.
          --quiet                   Suppress the timing footer.
          --help                    Show this message.
        """
}
