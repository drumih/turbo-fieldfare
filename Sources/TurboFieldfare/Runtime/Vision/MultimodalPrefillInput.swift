import Metal

public enum MultimodalPrefillInputError: Error, Equatable {
    case tokenCountMismatch
    case invalidImageTokenRange
    case featureShapeMismatch
}

public struct MultimodalImageSpan: Sendable {
    public let tokenRange: Range<Int>
    public let features: VisionFeatures

    public init(tokenRange: Range<Int>, features: VisionFeatures) {
        self.tokenRange = tokenRange
        self.features = features
    }
}

public struct MultimodalPrefillInput: Sendable {
    public let effectiveTokenIDs: [Int32]
    public let embeddingTokenIDs: [Int32]
    public let imageSpans: [MultimodalImageSpan]

    public var imageTokenRange: Range<Int> { imageSpans[0].tokenRange }
    public var imageFeatures: VisionFeatures { imageSpans[0].features }

    /// The same input with `tokens` placed in front of it, spans shifted to
    /// match. Used when a turn has to replay a boundary token the previous run
    /// left outside the KV: without this the multimodal input no longer lines
    /// up with the prefill suffix and the images were dropped instead.
    public func prepending(_ tokens: [Int32]) throws -> MultimodalPrefillInput {
        guard !tokens.isEmpty else { return self }
        let shift = tokens.count
        return try MultimodalPrefillInput(
            effectiveTokenIDs: tokens + effectiveTokenIDs,
            embeddingTokenIDs: tokens + embeddingTokenIDs,
            imageSpans: imageSpans.map {
                MultimodalImageSpan(
                    tokenRange: ($0.tokenRange.lowerBound + shift)
                        ..< ($0.tokenRange.upperBound + shift),
                    features: $0.features)
            })
    }

    /// The tail of a rendered prompt, for resuming on a cached KV prefix. Spans
    /// are rebased to the tail; a span that starts before `dropping` belongs to
    /// the cached prefix, whose features are already in the KV and are not
    /// available here, so it is rejected rather than re-injected.
    public func suffix(dropping cachedTokens: Int) throws -> MultimodalPrefillInput {
        guard cachedTokens >= 0, cachedTokens < effectiveTokenIDs.count else {
            throw MultimodalPrefillInputError.invalidImageTokenRange
        }
        let kept = imageSpans.filter { $0.tokenRange.lowerBound >= cachedTokens }
        guard kept.count == imageSpans.filter({
            $0.tokenRange.upperBound > cachedTokens
        }).count else {
            // A span straddling the boundary cannot be split.
            throw MultimodalPrefillInputError.invalidImageTokenRange
        }
        guard !kept.isEmpty else {
            // Every image is already inside the cached prefix, so the tail is
            // plain text and needs no multimodal input at all.
            throw MultimodalPrefillInputError.invalidImageTokenRange
        }
        return try MultimodalPrefillInput(
            effectiveTokenIDs: Array(effectiveTokenIDs.dropFirst(cachedTokens)),
            embeddingTokenIDs: Array(embeddingTokenIDs.dropFirst(cachedTokens)),
            imageSpans: kept.map {
                MultimodalImageSpan(
                    tokenRange: ($0.tokenRange.lowerBound - cachedTokens)
                        ..< ($0.tokenRange.upperBound - cachedTokens),
                    features: $0.features)
            })
    }

    public init(effectiveTokenIDs: [Int32],
                embeddingTokenIDs: [Int32],
                imageTokenRange: Range<Int>,
                imageFeatures: VisionFeatures) throws {
        try self.init(
            effectiveTokenIDs: effectiveTokenIDs,
            embeddingTokenIDs: embeddingTokenIDs,
            imageSpans: [MultimodalImageSpan(
                tokenRange: imageTokenRange,
                features: imageFeatures)])
    }

    public init(effectiveTokenIDs: [Int32],
                embeddingTokenIDs: [Int32],
                imageSpans: [MultimodalImageSpan]) throws {
        guard effectiveTokenIDs.count == embeddingTokenIDs.count else {
            throw MultimodalPrefillInputError.tokenCountMismatch
        }
        // The span count is self-limiting: spans are ordered, non-overlapping and
        // bounded by the prompt, so the prompt length caps how many can exist.
        guard !imageSpans.isEmpty else {
            throw MultimodalPrefillInputError.invalidImageTokenRange
        }
        var previousUpperBound = 0
        for span in imageSpans {
            let range = span.tokenRange
            guard !range.isEmpty,
                  range.lowerBound >= previousUpperBound,
                  range.upperBound <= embeddingTokenIDs.count,
                  range.count <= VisionConfig().maximumPooledTokens else {
                throw MultimodalPrefillInputError.invalidImageTokenRange
            }
            let featureBytes = range.count
                * VisionConfig().textHiddenSize
                * MemoryLayout<Float16>.stride
            guard span.features.tokenCount == range.count,
                  span.features.hiddenSize == VisionConfig().textHiddenSize,
                  span.features.buffer.length >= featureBytes else {
                throw MultimodalPrefillInputError.featureShapeMismatch
            }
            previousUpperBound = range.upperBound
        }
        self.effectiveTokenIDs = effectiveTokenIDs
        self.embeddingTokenIDs = embeddingTokenIDs
        self.imageSpans = imageSpans
    }
}
