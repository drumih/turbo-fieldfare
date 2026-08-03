import Foundation
import Metal

public struct Gemma4VisionFeatureCacheKey: Hashable, Codable, Sendable {
    public static let processorVersion = 1

    public let imageContentHash: String
    public let sidecarSnapshotHash: String
    public let processorVersion: Int

    public init(imageContentHash: String,
                sidecarSnapshotHash: String,
                processorVersion: Int = Self.processorVersion) {
        self.imageContentHash = imageContentHash.lowercased()
        self.sidecarSnapshotHash = sidecarSnapshotHash.lowercased()
        self.processorVersion = processorVersion
    }

    public var isValid: Bool {
        imageContentHash.count == 64
            && imageContentHash.allSatisfy(\.isHexDigit)
            && sidecarSnapshotHash.hasPrefix("sha256:")
            && sidecarSnapshotHash.count == 71
            && sidecarSnapshotHash.dropFirst(7).allSatisfy(\.isHexDigit)
            && processorVersion == Self.processorVersion
    }
}

/// Optional bounded cache for multi-turn image follow-ups. The encoder itself
/// retains no projected buffers; AppCore may own one cache and explicitly
/// clear it when switching away from an image chat.
public actor Gemma4VisionFeatureCache {
    private struct Entry {
        let features: Gemma4VisionFeatures
        var lastUse: UInt64
    }

    public let maximumBytes: Int
    public let maximumEntries: Int
    private var entries: [Gemma4VisionFeatureCacheKey: Entry] = [:]
    private var clock: UInt64 = 0
    private var residentBytes = 0

    public init(maximumBytes: Int = 8 * 1_024 * 1_024,
                maximumEntries: Int = 4) {
        precondition(maximumBytes >= 0)
        precondition(maximumEntries >= 0)
        self.maximumBytes = maximumBytes
        self.maximumEntries = maximumEntries
    }

    public func features(for key: Gemma4VisionFeatureCacheKey) -> Gemma4VisionFeatures? {
        guard key.isValid, var entry = entries[key] else { return nil }
        clock &+= 1
        entry.lastUse = clock
        entries[key] = entry
        return entry.features
    }

    public func insert(_ features: Gemma4VisionFeatures) {
        let key = Gemma4VisionFeatureCacheKey(
            imageContentHash: features.contentHash,
            sidecarSnapshotHash: features.sidecarSnapshotHash)
        guard key.isValid,
              (1...Gemma4VisionEncoder.maximumSoftTokens).contains(features.tokenCount),
              features.byteCount == features.buffer.length,
              features.byteCount <= maximumBytes,
              maximumEntries > 0 else {
            return
        }
        if let old = entries.removeValue(forKey: key) {
            residentBytes -= old.features.byteCount
        }
        clock &+= 1
        entries[key] = Entry(features: features, lastUse: clock)
        residentBytes += features.byteCount
        evictToLimits()
    }

    public func remove(_ key: Gemma4VisionFeatureCacheKey) {
        if let removed = entries.removeValue(forKey: key) {
            residentBytes -= removed.features.byteCount
        }
    }

    public func removeAll() {
        entries.removeAll(keepingCapacity: false)
        residentBytes = 0
    }

    public var count: Int { entries.count }
    public var byteCount: Int { residentBytes }

    private func evictToLimits() {
        while entries.count > maximumEntries || residentBytes > maximumBytes {
            guard let victim = entries.min(by: { $0.value.lastUse < $1.value.lastUse }) else {
                break
            }
            residentBytes -= victim.value.features.byteCount
            entries.removeValue(forKey: victim.key)
        }
    }
}
