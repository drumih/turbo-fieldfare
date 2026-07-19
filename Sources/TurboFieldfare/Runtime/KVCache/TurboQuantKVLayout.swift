import Foundation

enum TurboQuantKVMode: Sendable, Equatable {
    case disabled
    case k4v4NormCorrected
}

struct TurboQuantKVRoleLayout: Sendable, Equatable {
    let packedOffsetPerHead: Int
    let scaleOffsetPerHead: Int
    let bytesPerHead: Int
    let bytesPerToken: Int

}

struct TurboQuantKVLayerLayout: Sendable, Equatable {
    let key: TurboQuantKVRoleLayout
    let value: TurboQuantKVRoleLayout

}

enum TurboQuantKVLayout {
    static let fp16ScaleBytes = 2

    static func layer(mode: TurboQuantKVMode,
                             config: ArchConfig,
                             layer: Int,
                             capacity: Int) -> TurboQuantKVLayerLayout {
        let isFull = config.fullAttentionLayerMask[layer] != 0
        let headDim = isFull ? config.fullHeadDim : config.headDim
        let numKVHeads = isFull ? config.numFullKVHeads : config.numKVHeads
        return self.layer(mode: mode,
                          headDim: headDim,
                          numKVHeads: numKVHeads,
                          capacity: capacity)
    }

    static func layer(mode: TurboQuantKVMode,
                             headDim: Int,
                             numKVHeads: Int,
                             capacity: Int) -> TurboQuantKVLayerLayout {
        precondition(mode != .disabled, "disabled mode has no TurboQuant layout")
        precondition(headDim > 0 && headDim % 8 == 0,
                     "headDim must be positive and byte-packable")
        precondition(numKVHeads > 0)
        precondition(capacity > 0)
        let roleLayout = role(mode: mode,
                              headDim: headDim,
                              numKVHeads: numKVHeads)
        return TurboQuantKVLayerLayout(key: roleLayout,
                                       value: roleLayout)
    }

    static func role(mode: TurboQuantKVMode,
                            headDim: Int,
                            numKVHeads: Int) -> TurboQuantKVRoleLayout {
        precondition(mode == .k4v4NormCorrected)
        let packed = (headDim * 4 + 7) / 8
        let packedOffset = 0
        let scaleOffset = packedOffset + packed
        let perHead = scaleOffset + fp16ScaleBytes
        return TurboQuantKVRoleLayout(packedOffsetPerHead: packedOffset,
                                      scaleOffsetPerHead: scaleOffset,
                                      bytesPerHead: perHead,
                                      bytesPerToken: perHead * numKVHeads)
    }
}
