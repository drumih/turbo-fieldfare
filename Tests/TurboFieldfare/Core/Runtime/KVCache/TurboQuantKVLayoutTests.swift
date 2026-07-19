import Testing

@testable import TurboFieldfare

@Suite struct TurboQuantKVLayoutTests {
    @Test func swaLayoutUsesPackedK4V4Storage() {
        let layout = TurboQuantKVLayout.layer(mode: .k4v4NormCorrected,
                                              headDim: 256,
                                              numKVHeads: 8,
                                              capacity: 4)

        #expect(layout.key.packedOffsetPerHead == 0)
        #expect(layout.key.scaleOffsetPerHead == 128)
        #expect(layout.key.bytesPerHead == 130)
        #expect(layout.key.bytesPerToken == 1_040)
        #expect(layout.key.bytesPerToken + layout.value.bytesPerToken == 2_080)
        #expect(4 * (layout.key.bytesPerToken + layout.value.bytesPerToken) == 8_320)
    }

    @Test func fullAttentionLayoutUsesPackedK4V4Storage() {
        let layout = TurboQuantKVLayout.layer(mode: .k4v4NormCorrected,
                                              headDim: 512,
                                              numKVHeads: 2,
                                              capacity: 4)

        #expect(layout.key.scaleOffsetPerHead == 256)
        #expect(layout.key.bytesPerHead == 258)
        #expect(layout.key.bytesPerToken == 516)
        #expect(layout.value.bytesPerToken == 516)
        #expect(layout.key.bytesPerToken + layout.value.bytesPerToken == 1_032)
        #expect(4 * (layout.key.bytesPerToken + layout.value.bytesPerToken) == 4_128)
    }

    @Test func offsetsFollowTokenAndHeadStrides() {
        let layout = TurboQuantKVLayout.layer(mode: .k4v4NormCorrected,
                                              headDim: 256,
                                              numKVHeads: 8,
                                              capacity: 8)
        let headBase = 2 * 1_040 + 3 * 130

        #expect(headBase + layout.key.packedOffsetPerHead == headBase)
        #expect(headBase + layout.key.scaleOffsetPerHead == headBase + 128)
    }
}
