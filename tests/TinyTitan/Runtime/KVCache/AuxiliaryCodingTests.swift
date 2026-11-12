import Foundation
import Testing

@testable import TinyTitan

/// The PLE token context in the auxiliary payload (version 2). Without it, the
/// first `ngramSize - 1` tokens after every restore hashed the previous
/// sequence's n-grams; the wide-verify identity bisect measured that as the
/// largest divergence from scalar decode (docs/plan-wide-verify.md).
@Suite struct AuxiliaryCodingTests {
    @Test func tokenContextRoundTrips() throws {
        var data = Data()
        AuxiliaryCoding.appendTokens([248_044, 0, 13], to: &data)
        let tokens = try data.withUnsafeBytes { bytes in
            var offset = 0
            let read = try AuxiliaryCoding.readTokens(bytes, offset: &offset, maximumCount: 3)
            #expect(offset == bytes.count)
            return read
        }
        #expect(tokens == [248_044, 0, 13])
    }

    @Test func anEmptyContextIsValid() throws {
        var data = Data()
        AuxiliaryCoding.appendTokens([], to: &data)
        let tokens = try data.withUnsafeBytes { bytes in
            var offset = 0
            return try AuxiliaryCoding.readTokens(bytes, offset: &offset, maximumCount: 3)
        }
        #expect(tokens.isEmpty)
    }

    @Test func aContextLongerThanTheNgramIsRefused() {
        var data = Data()
        AuxiliaryCoding.appendTokens([1, 2, 3, 4], to: &data)
        #expect(throws: InferenceAuxiliaryStateError.self) {
            try data.withUnsafeBytes { bytes in
                var offset = 0
                _ = try AuxiliaryCoding.readTokens(bytes, offset: &offset, maximumCount: 3)
            }
        }
    }

    /// A version-1 payload ends where the context would start: the read must
    /// refuse rather than invent a context, so the restore fails and the caller
    /// re-prefills instead of hashing the wrong n-grams.
    @Test func aPayloadWithoutTheContextIsRefused() {
        let data = Data()
        #expect(throws: InferenceAuxiliaryStateError.self) {
            try data.withUnsafeBytes { bytes in
                var offset = 0
                _ = try AuxiliaryCoding.readTokens(bytes, offset: &offset, maximumCount: 3)
            }
        }
    }

    @Test func theFormatVersionMovedPastTheOneWithoutTheContext() {
        #expect(InferenceAuxiliaryState.currentVersion == 2)
    }
}
