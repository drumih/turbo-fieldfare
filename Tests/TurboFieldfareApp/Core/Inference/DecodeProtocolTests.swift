import Foundation
import Testing
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareAppCore

@Suite struct DecodeProtocolTests {
    @Test(arguments: ["null", "1", "\"true\"", "{}", "[]"])
    func invalidExpertPrefetchIsRejected(_ value: String) throws {
        let encoded = try JSONEncoder().encode(DecodeRuntimeOptions())
        var object = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object["expertPrefetchEnabled"] = try JSONSerialization.jsonObject(
            with: Data(value.utf8), options: .fragmentsAllowed)
        let data = try JSONSerialization.data(withJSONObject: object)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(DecodeRuntimeOptions.self, from: data)
        }
    }

    @Test func expertPrefetchWireDefaultAndLegacyDecodeAreOff() throws {
        #expect(!DecodeRuntimeOptions().expertPrefetchEnabled)
        let legacy = Data("""
        {"expertCacheSlots":16,"expertCachePolicy":"lfu","prefillEnabled":true,
         "prefillChunkTokens":128,"rdadvisePolicy":"off","modelVerification":"full-sha256"}
        """.utf8)
        #expect(try !JSONDecoder().decode(DecodeRuntimeOptions.self, from: legacy).expertPrefetchEnabled)
    }

    @Test func outboundRuntimeMappingPreservesExpertPrefetchChoice() throws {
        for enabled in [false, true] {
            let options = AppRuntimeOptions(expertPrefetchEnabled: enabled)
            let wire = DecodeServiceInferenceClient.decodeRuntimeOptions(options)
            #expect(wire.expertPrefetchEnabled == enabled)
            let decoded = try JSONDecoder().decode(DecodeRuntimeOptions.self, from: JSONEncoder().encode(wire))
            #expect(decoded == wire)
            #expect(decoded.expertPrefetchEnabled == enabled)
        }
    }

    @Test func loadRequestRoundTripPreservesEveryPublicRuntimeOption() throws {
        let options = DecodeRuntimeOptions(
            expertCacheSlots: 32,
            expertCachePolicy: "lru",
            prefillEnabled: false,
            expertPrefetchEnabled: false,
            prefillChunkTokens: 64,
            rdadvisePolicy: "adaptive",
            modelVerification: "trusted-install")
        let request = DecodeLoadRequest(
            modelPath: "/tmp/model.gturbo",
            maxContextTokens: 8192,
            runtimeOptions: options,
            forceLogitsHead: true)

        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(request))
        try pipe.fileHandleForWriting.close()
        let decoded = try DecodeFrameCodec.read(
            DecodeLoadRequest.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.modelPath == request.modelPath)
        #expect(decoded.maxContextTokens == 8192)
        #expect(decoded.runtimeOptions == options)
        #expect(decoded.forceLogitsHead)
    }

    @Test func terminalEventRoundTripPreservesDiagnosticsAndMemory() throws {
        let runner = DecodeRunnerDiagnostics(
            cb1MillisecondsPerToken: 0.6,
            ioMillisecondsPerToken: 12,
            cb2MillisecondsPerToken: 0.4,
            headMillisecondsPerToken: 1.7,
            rdadviseMillisecondsPerToken: 0,
            rdadviseCallsPerToken: 0,
            rdadviseMegabytesPerToken: 0,
            rdadviseSkippedPerToken: 0,
            rdadviseFailures: 0)
        let event = DecodeServiceEvent(
            kind: .finished,
            generationID: UUID(),
            tokenCount: 256,
            promptTokenCount: 1_017,
            computedPrefillTokens: 17,
            prefillSeconds: 10.2,
            timeToFirstTokenSeconds: 0.04,
            decodeSeconds: 7.7,
            tokensPerSecond: 33.2,
            currentMemoryBytes: 2_000_000_000,
            peakMemoryBytes: 2_100_000_000,
            runner: runner)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(event))
        try pipe.fileHandleForWriting.close()
        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.tokenCount == 256)
        #expect(decoded.promptTokenCount == 1_017)
        #expect(decoded.computedPrefillTokens == 17)
        #expect(decoded.currentMemoryBytes == 2_000_000_000)
        #expect(decoded.peakMemoryBytes == 2_100_000_000)
        #expect(decoded.runner == runner)
    }

    @Test func prefillEventRoundTripPreservesProgress() throws {
        let event = DecodeServiceEvent(
            kind: .prefill,
            generationID: UUID(),
            sequence: 7,
            prefillDone: 128,
            prefillTotal: 514)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(
            contentsOf: DecodeFrameCodec.encode(event))
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.kind == .prefill)
        #expect(decoded.sequence == 7)
        #expect(decoded.prefillDone == 128)
        #expect(decoded.prefillTotal == 514)
    }

    @Test func decoderAcceptsAFrameSplitAcrossSingleByteWrites() throws {
        let event = DecodeServiceEvent(
            kind: .snapshot,
            generationID: UUID(),
            sequence: 1,
            textDelta: "caf\u{00E9}",
            tokenCount: 1)
        let frame = try DecodeFrameCodec.encode(event)
        let pipe = Pipe()
        for byte in frame {
            try pipe.fileHandleForWriting.write(contentsOf: Data([byte]))
        }
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: pipe.fileHandleForReading)

        #expect(decoded.sequence == 1)
        #expect(decoded.textDelta == "caf\u{00E9}")
    }

    @Test func oversizedPayloadIsRejectedBeforeEncoding() {
        let request = DecodeGenerationRequest(
            prompt: String(repeating: "x", count: DecodeFrameCodec.maximumPayloadBytes + 1),
            maxNewTokens: 1,
            maxContextTokens: 4_096,
            temperature: 0)

        #expect(throws: DecodeFrameError.self) {
            _ = try DecodeFrameCodec.encode(request)
        }
    }

    @Test func oversizedFrameIsRejectedBeforePayloadRead() throws {
        let pipe = Pipe()
        var count = UInt32(DecodeFrameCodec.maximumPayloadBytes + 1).littleEndian
        try pipe.fileHandleForWriting.write(contentsOf: withUnsafeBytes(of: &count) { Data($0) })
        try pipe.fileHandleForWriting.close()

        #expect(throws: DecodeFrameError.self) {
            _ = try DecodeFrameCodec.read(
                DecodeServiceEvent.self,
                from: pipe.fileHandleForReading)
        }
    }
    @Test func generationRequestRoundTripPreservesImageDescriptors() throws {
        let attachment = DecodeImageAttachment(
            id: UUID(),
            path: "/tmp/staged-image",
            displayName: "image.png",
            encodedBytes: 123,
            sha256: String(repeating: "b", count: 64))
        let request = DecodeGenerationRequest(
            prompt: "describe",
            imageAttachments: [attachment],
            maxNewTokens: 16,
            maxContextTokens: 4096,
            temperature: 0)
        let pipe = Pipe()
        try pipe.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(request))
        try pipe.fileHandleForWriting.close()

        let decoded = try DecodeFrameCodec.read(
            DecodeGenerationRequest.self,
            from: pipe.fileHandleForReading)
        #expect(decoded.imageAttachments == [attachment])
    }

}
