/// The read-only diagnostic aggregate an `EngineRunner` exposes in place of
/// the individual `runner.total*` counters `ServerModelSession` used to read.
///
/// The 33 members are exactly the counters the session reads, pinned by
/// `EngineConformanceTests` (`tests/TinyTitan/Engine/`); a conforming
/// engine's `statistics` must forward each one from its existing counter, and
/// `resetStatistics()` must clear them through the engine's own resets. Two
/// further runner counters (`totalRDAdviseFailures`,
/// `totalRDAdviseSkipped`) are deliberately absent: nothing outside the
/// engine reads them. All values are `UInt64` nanosecond or event totals in
/// the 6.x runner's existing units — this type changes no accounting.
public struct EngineStatistics: Sendable, Equatable {
    public let totalBodyNanos: UInt64
    public let totalCachePlanNanos: UInt64
    public let totalCb1Nanos: UInt64
    public let totalCb2Nanos: UInt64
    public let totalEmbedNanos: UInt64
    public let totalExpertIOHostWaits: UInt64
    public let totalExpertIOHostWaitsAvoided: UInt64
    public let totalExposedIoNanos: UInt64
    public let totalGPUClassifiedHits: UInt64
    public let totalGPUClassifiedMisses: UInt64
    public let totalGPUResidencyAllHitLayers: UInt64
    public let totalGatherNanos: UInt64
    public let totalHeadFusedNanos: UInt64
    public let totalHeadNanos: UInt64
    public let totalHitFixupLayers: UInt64
    public let totalIOCompletionToFixupSubmitNanos: UInt64
    public let totalIOQueueNanos: UInt64
    public let totalIoNanos: UInt64
    public let totalLoopOtherNanos: UInt64
    public let totalLoopProgressNanos: UInt64
    public let totalLoopSampleNanos: UInt64
    public let totalMissIoNanos: UInt64
    public let totalPreambleNanos: UInt64
    public let totalPreamblePinNanos: UInt64
    public let totalPreambleReleaseNanos: UInt64
    public let totalPreambleReserveNanos: UInt64
    public let totalPrefetchAdopted: UInt64
    public let totalPrefetchIssued: UInt64
    public let totalRDAdviseBytes: UInt64
    public let totalRDAdviseCalls: UInt64
    public let totalRDAdviseNanos: UInt64
    public let totalRouterReadbackNanos: UInt64
    public let totalWaitNanos: UInt64

    public init(
        totalBodyNanos: UInt64,
        totalCachePlanNanos: UInt64,
        totalCb1Nanos: UInt64,
        totalCb2Nanos: UInt64,
        totalEmbedNanos: UInt64,
        totalExpertIOHostWaits: UInt64,
        totalExpertIOHostWaitsAvoided: UInt64,
        totalExposedIoNanos: UInt64,
        totalGPUClassifiedHits: UInt64,
        totalGPUClassifiedMisses: UInt64,
        totalGPUResidencyAllHitLayers: UInt64,
        totalGatherNanos: UInt64,
        totalHeadFusedNanos: UInt64,
        totalHeadNanos: UInt64,
        totalHitFixupLayers: UInt64,
        totalIOCompletionToFixupSubmitNanos: UInt64,
        totalIOQueueNanos: UInt64,
        totalIoNanos: UInt64,
        totalLoopOtherNanos: UInt64,
        totalLoopProgressNanos: UInt64,
        totalLoopSampleNanos: UInt64,
        totalMissIoNanos: UInt64,
        totalPreambleNanos: UInt64,
        totalPreamblePinNanos: UInt64,
        totalPreambleReleaseNanos: UInt64,
        totalPreambleReserveNanos: UInt64,
        totalPrefetchAdopted: UInt64,
        totalPrefetchIssued: UInt64,
        totalRDAdviseBytes: UInt64,
        totalRDAdviseCalls: UInt64,
        totalRDAdviseNanos: UInt64,
        totalRouterReadbackNanos: UInt64,
        totalWaitNanos: UInt64
    ) {
        self.totalBodyNanos = totalBodyNanos
        self.totalCachePlanNanos = totalCachePlanNanos
        self.totalCb1Nanos = totalCb1Nanos
        self.totalCb2Nanos = totalCb2Nanos
        self.totalEmbedNanos = totalEmbedNanos
        self.totalExpertIOHostWaits = totalExpertIOHostWaits
        self.totalExpertIOHostWaitsAvoided = totalExpertIOHostWaitsAvoided
        self.totalExposedIoNanos = totalExposedIoNanos
        self.totalGPUClassifiedHits = totalGPUClassifiedHits
        self.totalGPUClassifiedMisses = totalGPUClassifiedMisses
        self.totalGPUResidencyAllHitLayers = totalGPUResidencyAllHitLayers
        self.totalGatherNanos = totalGatherNanos
        self.totalHeadFusedNanos = totalHeadFusedNanos
        self.totalHeadNanos = totalHeadNanos
        self.totalHitFixupLayers = totalHitFixupLayers
        self.totalIOCompletionToFixupSubmitNanos = totalIOCompletionToFixupSubmitNanos
        self.totalIOQueueNanos = totalIOQueueNanos
        self.totalIoNanos = totalIoNanos
        self.totalLoopOtherNanos = totalLoopOtherNanos
        self.totalLoopProgressNanos = totalLoopProgressNanos
        self.totalLoopSampleNanos = totalLoopSampleNanos
        self.totalMissIoNanos = totalMissIoNanos
        self.totalPreambleNanos = totalPreambleNanos
        self.totalPreamblePinNanos = totalPreamblePinNanos
        self.totalPreambleReleaseNanos = totalPreambleReleaseNanos
        self.totalPreambleReserveNanos = totalPreambleReserveNanos
        self.totalPrefetchAdopted = totalPrefetchAdopted
        self.totalPrefetchIssued = totalPrefetchIssued
        self.totalRDAdviseBytes = totalRDAdviseBytes
        self.totalRDAdviseCalls = totalRDAdviseCalls
        self.totalRDAdviseNanos = totalRDAdviseNanos
        self.totalRouterReadbackNanos = totalRouterReadbackNanos
        self.totalWaitNanos = totalWaitNanos
    }

    /// The all-zero aggregate, for tests that assert a reset.
    public static let zero = EngineStatistics(
        totalBodyNanos: 0,
        totalCachePlanNanos: 0,
        totalCb1Nanos: 0,
        totalCb2Nanos: 0,
        totalEmbedNanos: 0,
        totalExpertIOHostWaits: 0,
        totalExpertIOHostWaitsAvoided: 0,
        totalExposedIoNanos: 0,
        totalGPUClassifiedHits: 0,
        totalGPUClassifiedMisses: 0,
        totalGPUResidencyAllHitLayers: 0,
        totalGatherNanos: 0,
        totalHeadFusedNanos: 0,
        totalHeadNanos: 0,
        totalHitFixupLayers: 0,
        totalIOCompletionToFixupSubmitNanos: 0,
        totalIOQueueNanos: 0,
        totalIoNanos: 0,
        totalLoopOtherNanos: 0,
        totalLoopProgressNanos: 0,
        totalLoopSampleNanos: 0,
        totalMissIoNanos: 0,
        totalPreambleNanos: 0,
        totalPreamblePinNanos: 0,
        totalPreambleReleaseNanos: 0,
        totalPreambleReserveNanos: 0,
        totalPrefetchAdopted: 0,
        totalPrefetchIssued: 0,
        totalRDAdviseBytes: 0,
        totalRDAdviseCalls: 0,
        totalRDAdviseNanos: 0,
        totalRouterReadbackNanos: 0,
        totalWaitNanos: 0)
}
