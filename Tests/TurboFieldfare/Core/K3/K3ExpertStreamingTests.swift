import Testing
import Foundation
import Darwin
import Metal
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

/// `K3ExpertStreaming` against synthetic per-layer expert files with known
/// byte patterns: split-pread chunk correctness, prediction hit/miss
/// accounting across two tokens, transient-bank ping-pong, direct-resident
/// layer banks, and the stats counters. No model bundle involved — the
/// designated initializer takes a layer-file provider.
@Suite struct K3ExpertStreamingTests {

    static let pageSize = 16 * 1024
    /// 4 pages: splits 4 ways into clean 1-page chunks.
    static let stride4 = 4 * pageSize
    /// 5 pages: exercises the uneven remainder distribution (2,1,1,1).
    static let stride5 = 5 * pageSize

    /// Every byte of expert `expert`'s blob in layer `layer` identifies its
    /// origin: layer/expert in the high nibble mix, page index in the low
    /// bits, so a misordered split chunk reads as the wrong page tag.
    static func patternByte(layer: Int, expert: Int, offset: Int) -> UInt8 {
        UInt8((layer * 16 + expert + offset / pageSize) % 251)
    }

    static func expectedBlob(layer: Int, expert: Int, stride: Int) -> [UInt8] {
        (0..<stride).map { patternByte(layer: layer, expert: expert, offset: $0) }
    }

    struct SyntheticFiles {
        let directory: URL
        let layers: [Int]
        let expertsPerLayer: Int
        let stride: Int
    }

    /// Write one file per layer, `expertsPerLayer` blobs of `stride` bytes.
    @discardableResult
    static func writeFiles(layers: [Int], expertsPerLayer: Int, stride: Int)
        throws -> SyntheticFiles {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("k3-streaming-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for layer in layers {
            var bytes = [UInt8](repeating: 0, count: expertsPerLayer * stride)
            for expert in 0..<expertsPerLayer {
                let base = expert * stride
                for i in 0..<stride {
                    bytes[base + i] = patternByte(layer: layer, expert: expert, offset: i)
                }
            }
            try Data(bytes).write(
                to: dir.appendingPathComponent(String(format: "layer_%02d.bin", layer)))
        }
        return SyntheticFiles(directory: dir, layers: layers,
                              expertsPerLayer: expertsPerLayer, stride: stride)
    }

    static func provider(_ files: SyntheticFiles)
        -> (Int) throws -> K3ExpertLayerFile {
        { layer in
            let path = files.directory
                .appendingPathComponent(String(format: "layer_%02d.bin", layer)).path
            let fd = open(path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else { throw StreamerError.openFailed(path: path, errno: errno) }
            return K3ExpertLayerFile(
                layer: layer, path: path, fileDescriptor: fd,
                expertsPerLayer: files.expertsPerLayer,
                expertStride: UInt64(files.stride),
                expertOffsets: (0..<files.expertsPerLayer).map {
                    UInt64($0 * files.stride)
                })
        }
    }

    static func makeStreamer(files: SyntheticFiles,
                             policy: K3ExpertPrefetchPolicy,
                             slotsPerBank: Int = 4,
                             ioSplits: Int = 4,
                             ioWorkers: K3ExpertIOWorkers = .adaptive,
                             residentCacheBudgetBytes: UInt64 = 0) throws
        -> K3ExpertStreaming {
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try K3ExpertStreaming(
            device: device,
            expertStride: UInt64(files.stride),
            expertsPerLayer: files.expertsPerLayer,
            moeLayers: files.layers,
            policy: policy,
            slotsPerBank: slotsPerBank,
            ioSplits: ioSplits,
            ioWorkers: ioWorkers,
            residentCacheBudgetBytes: residentCacheBudgetBytes,
            layerFileProvider: provider(files))
    }

    /// Read one served expert's bytes back from the batch buffer.
    static func verifyBatch(_ batch: K3ExpertBatch,
                            layer: Int,
                            experts: [Int],
                            stride: Int) {
        let base = batch.buffer.contents()
        for (index, expert) in experts.enumerated() {
            let offset = Int(batch.slotOffsets[index])
            let actual = UnsafeRawBufferPointer(
                start: base.advanced(by: offset), count: stride)
            let expected = expectedBlob(layer: layer, expert: expert, stride: stride)
            #expect(Array(actual) == expected,
                    "layer \(layer) expert \(expert) bytes mismatch")
        }
    }

    // MARK: - Split-pread correctness

    @Test func splitReadsMatchFileBytes() throws {
        let files = try Self.writeFiles(layers: [1, 2, 3], expertsPerLayer: 4,
                                   stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        for splits in [1, 4] {
            let streamer = try Self.makeStreamer(files: files, policy: .off,
                                            ioSplits: splits)
            // Scrambled request order: slot offsets must follow request order.
            let experts = [3, 1, 0, 2]
            let batch = try streamer.beginLayer(2, actualExperts: experts)
            #expect(batch.bankIndex == 0)
            Self.verifyBatch(batch, layer: 2, experts: experts,
                             stride: Self.stride4)
            streamer.endLayer(batch)
            let stats = streamer.stats()
            #expect(stats.demandMisses == 4)
            #expect(stats.demandHits == 0)
            #expect(stats.demandBytes == UInt64(4 * Self.stride4))
        }
    }

    @Test func unevenSplitReadsMatchFileBytes() throws {
        let files = try Self.writeFiles(layers: [1, 2, 3], expertsPerLayer: 4,
                                   stride: Self.stride5)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let streamer = try Self.makeStreamer(files: files, policy: .off, ioSplits: 4)
        let experts = [0, 2, 3, 1]
        let batch = try streamer.beginLayer(3, actualExperts: experts)
        Self.verifyBatch(batch, layer: 3, experts: experts, stride: Self.stride5)
        streamer.endLayer(batch)
    }

    /// Logical experts alternate between two physical files. The selected
    /// top-k order still resolves exact bytes while the bounded pool receives
    /// jobs carrying two independent descriptors (the runtime shape used by
    /// multi-SSD expert stripes).
    @Test func stripedExpertLocationsReadAcrossDescriptors() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("k3-striped-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let layer = 1
        let expertsPerLayer = 8
        var paths: [URL] = []
        for shard in 0..<2 {
            let experts = stride(from: shard, to: expertsPerLayer, by: 2)
            var bytes: [UInt8] = []
            for expert in experts {
                bytes += Self.expectedBlob(
                    layer: layer, expert: expert, stride: Self.stride4)
            }
            let url = directory.appendingPathComponent("shard_\(shard).bin")
            try Data(bytes).write(to: url)
            paths.append(url)
        }
        let fds = try paths.map { url -> Int32 in
            let fd = open(url.path, O_RDONLY | O_CLOEXEC)
            guard fd >= 0 else {
                throw StreamerError.openFailed(path: url.path, errno: errno)
            }
            return fd
        }
        let locations = (0..<expertsPerLayer).map { expert in
            K3ExpertLayerFile.Location(
                fileDescriptor: fds[expert % 2],
                offset: UInt64((expert / 2) * Self.stride4))
        }
        let file = K3ExpertLayerFile(
            layer: layer,
            path: "striped://test/layer/1",
            expertsPerLayer: expertsPerLayer,
            expertStride: UInt64(Self.stride4),
            expertLocations: locations)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let streamer = try K3ExpertStreaming(
            device: device,
            expertStride: UInt64(Self.stride4),
            expertsPerLayer: expertsPerLayer,
            moeLayers: [layer],
            policy: .off,
            slotsPerBank: 4,
            ioSplits: 1,
            ioWorkers: .fixed(2),
            layerFileProvider: { _ in file })

        let experts = [7, 0, 5, 2]
        let batch = try streamer.beginLayer(layer, actualExperts: experts)
        Self.verifyBatch(batch, layer: layer, experts: experts, stride: Self.stride4)
        streamer.endLayer(batch)
        #expect(streamer.stats().demandMisses == 4)
        #expect(streamer.stats().peakConcurrentReads <= 2)
    }

    @Test func fixedWorkerPoolBoundsConcurrentPreads() throws {
        let files = try Self.writeFiles(layers: [1], expertsPerLayer: 4,
                                        stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let streamer = try Self.makeStreamer(
            files: files, policy: .off, ioSplits: 4, ioWorkers: .fixed(2))
        let experts = [0, 1, 2, 3]
        let batch = try streamer.beginLayer(1, actualExperts: experts)
        Self.verifyBatch(batch, layer: 1, experts: experts, stride: Self.stride4)
        streamer.endLayer(batch)

        let stats = streamer.stats()
        #expect(stats.ioWorkerLimit == 2)
        #expect(stats.peakConcurrentReads > 0)
        #expect(stats.peakConcurrentReads <= 2)
        #expect(stats.ioBatches == 1)
        #expect(stats.ioNanos > 0)
        #expect(stats.ioTuningComplete)
    }

    @Test func adaptiveTunerPinsHighestMeasuredThroughput() {
        var tuner = K3ExpertIOAutotuner(candidates: [1, 2, 4],
                                        samplesPerCandidate: 2)
        let bytes: UInt64 = 1_000
        // Candidates are interleaved by round: 1 worker = 1 byte/ns,
        // 2 workers = 4 bytes/ns, 4 workers = 2 bytes/ns.
        for nanos: UInt64 in [1_000, 250, 500, 1_000, 250, 500] {
            let plan = tuner.plan(chunkCount: 16)
            tuner.record(plan: plan, bytes: bytes, nanos: nanos)
        }
        #expect(tuner.isComplete)
        #expect(tuner.selectedWorkers == 2)
        #expect(tuner.currentWorkers == 2)
        #expect(tuner.observationCount == 6)
        #expect(tuner.plan(chunkCount: 1).workers == 1)
    }

    @Test func productionDefaultsUseWholeExpertReadsAndBoundedCandidates() {
        #expect(K3ExpertStreaming.defaultIOSplits == 1)
        var tuner = K3ExpertIOAutotuner()
        var observed: [Int] = []
        for _ in 0..<6 {
            let plan = tuner.plan(chunkCount: 16)
            observed.append(plan.workers)
            tuner.record(plan: plan, bytes: 1_000, nanos: UInt64(1_000 / plan.workers))
        }
        #expect(Set(observed) == Set([1, 2, 4]))
        #expect(tuner.isComplete)
        #expect(tuner.selectedWorkers == 4)
    }

    @Test func defaultPolicyUsesOnDemandReads() throws {
        let files = try Self.writeFiles(layers: [1], expertsPerLayer: 2,
                                        stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let streamer = try K3ExpertStreaming(
            device: device,
            expertStride: UInt64(files.stride),
            expertsPerLayer: files.expertsPerLayer,
            moeLayers: files.layers,
            layerFileProvider: Self.provider(files))
        #expect(streamer.policy == .off)
    }

    @Test func explicitUncachedModeAppliesToExpertDescriptors() throws {
        let files = try Self.writeFiles(layers: [1], expertsPerLayer: 2,
                                        stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let streamer = try K3ExpertStreaming(
            device: device,
            expertStride: UInt64(files.stride),
            expertsPerLayer: files.expertsPerLayer,
            moeLayers: files.layers,
            policy: .off,
            slotsPerBank: 2,
            ioSplits: 1,
            ioWorkers: .fixed(1),
            ioCachePolicy: .uncached,
            layerFileProvider: Self.provider(files))
        let batch = try streamer.beginLayer(1, actualExperts: [0, 1])
        streamer.endLayer(batch)
        #expect(streamer.stats().ioCacheMode == "uncached")
    }

    // MARK: - Prediction across two tokens

    /// Two tokens over three MoE layers with identical routing. Token A is
    /// all cold misses; by token B every layer is served from prediction
    /// (wrap-around prefetch covers the first layer too).
    @Test func predictionPrefetchAccountingAcrossTwoTokens() throws {
        let files = try Self.writeFiles(layers: [1, 2, 3], expertsPerLayer: 8,
                                   stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let streamer = try Self.makeStreamer(files: files, policy: .predict)
        let routing: [Int: [Int]] = [1: [0, 1, 2, 3], 2: [0, 1, 2, 3],
                                     3: [0, 1, 2, 3]]

        // Token A: everything cold.
        for layer in [1, 2, 3] {
            let batch = try streamer.beginLayer(layer, actualExperts: routing[layer]!)
            Self.verifyBatch(batch, layer: layer, experts: routing[layer]!,
                             stride: Self.stride4)
            streamer.recordRouting(layer0: layer, experts: routing[layer]!)
            streamer.endLayer(batch)
        }
        var stats = streamer.stats()
        #expect(stats.demandMisses == 12)
        #expect(stats.demandHits == 0)
        #expect(stats.prefetchSkippedCold == 2)   // layers 2 and 3 had no prediction yet
        // The wrap-around prefetch at layer 3 covers token B's layer 1.
        #expect(stats.prefetchesIssued == 4)
        #expect(stats.prefetchBytes == UInt64(4 * Self.stride4))

        // Token B: identical routing — all hits.
        for layer in [1, 2, 3] {
            let batch = try streamer.beginLayer(layer, actualExperts: routing[layer]!)
            Self.verifyBatch(batch, layer: layer, experts: routing[layer]!,
                             stride: Self.stride4)
            streamer.recordRouting(layer0: layer, experts: routing[layer]!)
            streamer.endLayer(batch)
        }
        stats = streamer.stats()
        #expect(stats.demandHits == 12)
        #expect(stats.demandMisses == 12)
        #expect(stats.prefetchSkippedBankBusy == 0)
        // 4 (token A wrap) + 12 (token B steady-state) prefetch reads.
        #expect(stats.prefetchesIssued == 16)
        #expect(stats.prefetchBytes == UInt64(16 * Self.stride4))
        #expect(stats.demandBytes == UInt64(12 * Self.stride4))
        #expect(stats.bytesRead == UInt64(28 * Self.stride4))
        #expect(stats.hitRate == 0.5)
    }

    /// Holding the previous batch open (no endLayer) makes the target bank
    /// busy: the prefetch is skipped, the bank's contents stay untouched, and
    /// the next layer degrades to on-demand reads.
    @Test func bankBusySkipsPrefetchAndKeepsContents() throws {
        let files = try Self.writeFiles(layers: [1, 2, 3], expertsPerLayer: 8,
                                   stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let streamer = try Self.makeStreamer(files: files, policy: .predict)
        let routing = [0, 1, 2, 3]

        // Token A, but keep the layer-3 batch (bank 0) open afterwards.
        var held: K3ExpertBatch?
        for layer in [1, 2, 3] {
            let batch = try streamer.beginLayer(layer, actualExperts: routing)
            streamer.recordRouting(layer0: layer, experts: routing)
            if layer == 3 {
                held = batch
            } else {
                streamer.endLayer(batch)
            }
        }

        // Token B, layer 1 (bank 1): token A's layer-3 batch still occupies
        // bank 0, so the prefetch for layer 2 (target: bank 0) sees a live
        // bank and is skipped; bank 0's contents stay untouched.
        let bank0Before = streamer.bankContents(0)
        let batch1 = try streamer.beginLayer(1, actualExperts: routing)
        streamer.recordRouting(layer0: 1, experts: routing)
        let statsMid = streamer.stats()
        #expect(statsMid.prefetchSkippedBankBusy == 4)
        let bank0After = streamer.bankContents(0)
        #expect(bank0After.count == bank0Before.count)
        #expect(zip(bank0After, bank0Before).allSatisfy {
            $0.layer == $1.layer && $0.expert == $1.expert
        }, "busy bank 0 contents were overwritten")

        // Release everything, then layer 2 must demand-read (no prefetch ran,
        // and bank 0 holds layer-3 content, not layer-2's).
        streamer.endLayer(batch1)
        streamer.endLayer(held!)
        let batch2 = try streamer.beginLayer(2, actualExperts: routing)
        Self.verifyBatch(batch2, layer: 2, experts: routing, stride: Self.stride4)
        streamer.endLayer(batch2)
        let stats = streamer.stats()
        // 12 (token A, cold) + 4 (layer 2: no prefetch ran, bank 0 held
        // layer-3 content). Token B layer 1 scored 4 hits via the wrap-around
        // prefetch issued at token A's layer 3.
        #expect(stats.demandMisses == 16)
        #expect(stats.demandHits == 4)
    }

    /// A partially-wrong prediction: bank holds {0,1,2,3}, the actual set is
    /// {2,3,4,5} — two prediction hits, two demand misses evicting the
    /// predicted-but-unused slots.
    @Test func partialPredictionOverlap() throws {
        let files = try Self.writeFiles(layers: [1, 2], expertsPerLayer: 8,
                                   stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let streamer = try Self.makeStreamer(files: files, policy: .predict)

        let first = [0, 1, 2, 3]
        for layer in [1, 2] {
            let batch = try streamer.beginLayer(layer, actualExperts: first)
            streamer.recordRouting(layer0: layer, experts: first)
            streamer.endLayer(batch)
        }
        streamer.resetStats()

        let second = [2, 3, 4, 5]
        let batch = try streamer.beginLayer(1, actualExperts: second)
        Self.verifyBatch(batch, layer: 1, experts: second, stride: Self.stride4)
        streamer.recordRouting(layer0: 1, experts: second)
        streamer.endLayer(batch)
        let stats = streamer.stats()
        #expect(stats.demandHits == 2)
        #expect(stats.demandMisses == 2)
        #expect(stats.demandBytes == UInt64(2 * Self.stride4))
    }

    /// Selective mode waits for two observations, then prefetches no more
    /// than four experts that survived in both routes. The remaining experts
    /// are exact demand reads, so a bad or incomplete prediction is harmless.
    @Test func selectivePredictionCapsStableIntersectionAtFour() throws {
        let files = try Self.writeFiles(layers: [1, 2, 3], expertsPerLayer: 8,
                                        stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let streamer = try Self.makeStreamer(files: files, policy: .selective,
                                             slotsPerBank: 6)
        let stable = [0, 1, 2, 3, 4, 5]

        // Tokens A and B establish two-route history. The layer-3 wrap at
        // the end of token B schedules four stable experts for token C.
        for _ in 0..<2 {
            for layer in [1, 2, 3] {
                let batch = try streamer.beginLayer(layer, actualExperts: stable)
                Self.verifyBatch(batch, layer: layer, experts: stable,
                                 stride: Self.stride4)
                streamer.recordRouting(layer0: layer, experts: stable)
                streamer.endLayer(batch)
            }
        }
        let before = streamer.stats()
        #expect(before.prefetchesIssued == 4)

        for layer in [1, 2, 3] {
            let batch = try streamer.beginLayer(layer, actualExperts: stable)
            Self.verifyBatch(batch, layer: layer, experts: stable,
                             stride: Self.stride4)
            streamer.recordRouting(layer0: layer, experts: stable)
            streamer.endLayer(batch)
        }
        let stats = streamer.stats()
        #expect(stats.demandHits == 12)
        #expect(stats.demandMisses == 42)
        #expect(stats.prefetchesIssued == 16)
        #expect(stats.prefetchBytes == UInt64(16 * Self.stride4))
    }

    /// A complete resident layer bank is itself the Metal compute buffer.
    /// The second visit reuses exact packed bytes in place: no SSD read, hit
    /// copy, or miss-population copy is permitted.
    @Test func directResidentBankReusesExactBytesWithoutCopies() throws {
        let files = try Self.writeFiles(layers: [1], expertsPerLayer: 4,
                                        stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let cacheBytes = UInt64(4 * Self.stride4)
        let streamer = try Self.makeStreamer(
            files: files,
            policy: .off,
            slotsPerBank: 4,
            residentCacheBudgetBytes: cacheBytes)
        let experts = [0, 1, 2, 3]

        let cold = try streamer.beginLayer(1, actualExperts: experts)
        Self.verifyBatch(cold, layer: 1, experts: experts, stride: Self.stride4)
        #expect(streamer.directResidentMetalViewCount() == 1)
        streamer.endLayer(cold)
        #expect(streamer.directResidentMetalViewCount() == 0)
        var stats = streamer.stats()
        #expect(stats.demandMisses == 4)
        #expect(stats.residentCacheEntries == 4)
        #expect(stats.residentCachePopulateBytes == 0)
        #expect(stats.residentCacheCopyBytes == 0)

        // The same layer returns the same fixed compute bank.
        let warm = try streamer.beginLayer(1, actualExperts: experts)
        Self.verifyBatch(warm, layer: 1, experts: experts, stride: Self.stride4)
        #expect(streamer.directResidentMetalViewCount() == 1)
        streamer.endLayer(warm)
        #expect(streamer.directResidentMetalViewCount() == 0)
        stats = streamer.stats()
        #expect(stats.demandMisses == 4)
        #expect(stats.demandHits == 4)
        #expect(stats.residentCacheHits == 4)
        #expect(stats.residentCacheMisses == 4)
        #expect(stats.residentCacheCopyBytes == 0)
        #expect(stats.residentCachePopulateBytes == 0)
        #expect(stats.demandBytes == cacheBytes)

        streamer.resetStats()
        stats = streamer.stats()
        #expect(stats.residentCacheEntries == 4)
        #expect(stats.residentCacheCapacity == 4)
        #expect(stats.residentCacheBytes == cacheBytes)
        #expect(stats.residentCacheHits == 0)
    }

    @Test func directResidentBankReadsOnlyChangedExperts() throws {
        let files = try Self.writeFiles(layers: [1], expertsPerLayer: 6,
                                        stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let streamer = try Self.makeStreamer(
            files: files,
            policy: .off,
            slotsPerBank: 4,
            residentCacheBudgetBytes: UInt64(4 * Self.stride4))

        let first = try streamer.beginLayer(1, actualExperts: [0, 1, 2, 3])
        Self.verifyBatch(first, layer: 1, experts: [0, 1, 2, 3],
                         stride: Self.stride4)
        streamer.endLayer(first)
        let second = try streamer.beginLayer(1, actualExperts: [2, 3, 4, 5])
        Self.verifyBatch(second, layer: 1, experts: [2, 3, 4, 5],
                         stride: Self.stride4)
        streamer.endLayer(second)

        let stats = streamer.stats()
        #expect(stats.demandHits == 2)
        #expect(stats.demandMisses == 6)
        #expect(stats.demandBytes == UInt64(6 * Self.stride4))
        #expect(stats.residentCacheHits == 2)
        #expect(stats.residentCacheMisses == 6)
        #expect(stats.residentCacheCopyBytes == 0)
        #expect(stats.residentCachePopulateBytes == 0)
    }

    /// Resident budgets are allocated only as complete per-layer banks. This
    /// keeps every returned batch in one MTLBuffer, matching the production
    /// kernel contract, while layers outside the budget use transient banks.
    @Test func residentBudgetUsesWholeLayerBanks() throws {
        let files = try Self.writeFiles(layers: [1, 2, 3], expertsPerLayer: 4,
                                        stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let layerBytes = UInt64(4 * Self.stride4)
        let streamer = try Self.makeStreamer(
            files: files,
            policy: .off,
            slotsPerBank: 4,
            residentCacheBudgetBytes: layerBytes * 2 + UInt64(Self.stride4))

        var stats = streamer.stats()
        #expect(stats.residentCacheCapacity == 8)
        #expect(stats.residentCacheBytes == layerBytes * 2)
        // Two fixed resident banks plus one transient bank for the last layer.
        #expect(streamer.slotPoolBytes == 3 * Int(layerBytes))
        #expect(streamer.slotPoolAllocationCount() == 1)

        for _ in 0..<2 {
            for layer in files.layers {
                let batch = try streamer.beginLayer(layer, actualExperts: [0, 1, 2, 3])
                Self.verifyBatch(batch, layer: layer, experts: [0, 1, 2, 3],
                                 stride: Self.stride4)
                streamer.endLayer(batch)
            }
        }
        stats = streamer.stats()
        #expect(stats.residentCacheHits == 8)
        #expect(stats.residentCacheMisses == 8)
        #expect(stats.residentCacheEntries == 8)
        #expect(stats.residentCacheCopyBytes == 0)
        #expect(stats.residentCachePopulateBytes == 0)
        // The sole transient layer also remains in place between tokens, so
        // all three layers avoid their second demand read in this fixture.
        #expect(stats.demandHits == 12)
        #expect(stats.demandMisses == 12)
    }

    /// `off` policy: no prefetch traffic at all; with three layers the bank
    /// parity flips between tokens so every visit demand-reads.
    @Test func offPolicyReadsEverythingOnDemand() throws {
        let files = try Self.writeFiles(layers: [1, 2, 3], expertsPerLayer: 8,
                                   stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let streamer = try Self.makeStreamer(files: files, policy: .off)
        let routing = [0, 1, 2, 3]

        for _ in 0..<2 {
            for layer in [1, 2, 3] {
                let batch = try streamer.beginLayer(layer, actualExperts: routing)
                Self.verifyBatch(batch, layer: layer, experts: routing,
                                 stride: Self.stride4)
                streamer.recordRouting(layer0: layer, experts: routing)
                streamer.endLayer(batch)
            }
        }
        let stats = streamer.stats()
        #expect(stats.demandMisses == 24)
        #expect(stats.demandHits == 0)
        #expect(stats.prefetchesIssued == 0)
        #expect(stats.prefetchBytes == 0)
        #expect(stats.bytesRead == stats.demandBytes)
    }

    /// Slot-pool memory math: an off-policy one-layer fixture needs one bank;
    /// canonical zero-cache K3 still has 92 layers and therefore two.
    @Test func slotPoolMemoryMath() throws {
        let files = try Self.writeFiles(layers: [1], expertsPerLayer: 2,
                                   stride: Self.stride4)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        let streamer = try Self.makeStreamer(files: files, policy: .off,
                                        slotsPerBank: 2)
        #expect(streamer.slotPoolBytes == 2 * Self.stride4)
        #expect(streamer.slotPoolAllocationCount() == 1)
        // Canonical K3: 2 x 16 x 17,547,264 B = 561,512,448 B (~535.5 MiB).
        #expect(2 * 16 * Int(KimiK3FormatProfile.expertStride) == 561_512_448)
        #expect(KimiK3FormatProfile.expertStride % UInt64(Self.pageSize) == 0)
    }
}
