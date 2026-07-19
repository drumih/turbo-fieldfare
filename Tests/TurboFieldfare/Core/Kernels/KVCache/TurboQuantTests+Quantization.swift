import Testing

extension TurboQuantTests {
    @Test func packedRoundTripD256() throws {
        try Self.runPackedRoundTrip(headDim: 256)
    }

    @Test func packedRoundTripD512() throws {
        try Self.runPackedRoundTrip(headDim: 512)
    }
}
