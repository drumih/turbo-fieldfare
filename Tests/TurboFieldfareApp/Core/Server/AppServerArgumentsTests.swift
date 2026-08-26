import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppServerArgumentsTests {
    @Test func buildsFlagsFromRuntimeOptionsWithoutVision() {
        let options = AppRuntimeOptions(
            expertCacheSlots: 24,
            expertCachePolicy: .lru,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            rdadvisePolicy: .bounded)

        let arguments = AppServerArguments.build(
            modelPath: "/tmp/gemma4.gturbo",
            maxContextTokens: 32_768,
            runtimeOptions: options,
            visionPackPath: nil,
            port: 9090,
            queueLimit: 8)

        #expect(arguments == [
            "--model", "/tmp/gemma4.gturbo",
            "--max-context", "32768",
            "--expert-cache-slots", "24",
            "--expert-cache-policy", "lru",
            "--prefill", "off",
            "--prefill-chunk-tokens", "64",
            "--rdadvise", "bounded",
            "--port", "9090",
            "--queue-limit", "8",
        ])
    }

    @Test func prefillOnMapsToOn() {
        let arguments = AppServerArguments.build(
            modelPath: "/tmp/gemma4.gturbo",
            maxContextTokens: 16_384,
            runtimeOptions: AppRuntimeOptions(prefillEnabled: true),
            visionPackPath: nil,
            port: 8080,
            queueLimit: 4)

        #expect(arguments.contains("--prefill"))
        let index = arguments.firstIndex(of: "--prefill")!
        #expect(arguments[index + 1] == "on")
    }

    @Test func appendsVisionPackFlagWhenPathProvided() {
        let arguments = AppServerArguments.build(
            modelPath: "/tmp/gemma4.gturbo",
            maxContextTokens: 16_384,
            runtimeOptions: AppRuntimeOptions(),
            visionPackPath: "/tmp/gemma4.vision.gturbo",
            port: 8080,
            queueLimit: 4)

        #expect(arguments.suffix(2) == ["--vision-pack", "/tmp/gemma4.vision.gturbo"])
    }

    @Test func omitsVisionPackFlagWhenPathIsNil() {
        let arguments = AppServerArguments.build(
            modelPath: "/tmp/gemma4.gturbo",
            maxContextTokens: 16_384,
            runtimeOptions: AppRuntimeOptions(),
            visionPackPath: nil,
            port: 8080,
            queueLimit: 4)

        #expect(!arguments.contains("--vision-pack"))
    }
}
