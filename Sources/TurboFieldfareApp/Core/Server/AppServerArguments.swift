/// Builds the `TurboFieldfareServer` command-line flags from state the app
/// already tracks, so the server is always configured with exactly what the
/// Memory/Runtime sections show — never a separate, driftable copy.
public enum AppServerArguments {
    public static func build(modelPath: String,
                             maxContextTokens: Int,
                             runtimeOptions: AppRuntimeOptions,
                             visionPackPath: String?,
                             port: Int,
                             queueLimit: Int) -> [String] {
        var arguments = [
            "--model", modelPath,
            "--max-context", String(maxContextTokens),
            "--expert-cache-slots", String(runtimeOptions.expertCacheSlots),
            "--expert-cache-policy", runtimeOptions.expertCachePolicy.rawValue,
            "--prefill", runtimeOptions.prefillEnabled ? "on" : "off",
            "--prefill-chunk-tokens", String(runtimeOptions.prefillChunkTokens),
            "--rdadvise", runtimeOptions.rdadvisePolicy.rawValue,
            "--port", String(port),
            "--queue-limit", String(queueLimit),
        ]
        if let visionPackPath {
            arguments += ["--vision-pack", visionPackPath]
        }
        return arguments
    }
}
