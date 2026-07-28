import Foundation
import Darwin
import Darwin.Mach

/// Counters tracked during a repack run. Emitted to JSON via the
/// `--copy-audit` flag.
public final class RepackAudit {
    public var sourceBytesRead: UInt64 = 0
    public var outputBytesWritten: UInt64 = 0
    public var intentionalCopyBytes: UInt64 = 0
    public var byteCopyTiles: UInt64 = 0
    public var largestScratchBytes: Int = 0
    public var wholeFileHeapBuffers: Bool = false
    public var platformMemmoveObserved: Bool = false
    public var mallocCountObserved: Int = 0
    public var bitWidthOverridesHonored: Int = 0
    public var sourceSnapshotSha256: String = ""
    public var tensorsDroppedMultimodal: [String] = []
    public var wallTimeSeconds: Double = 0
    public var peakRssBytes: UInt64 = 0
    public var outputFiles: [OutputFile] = []
    public var rssSamples: [UInt64] = []
    public var packedExpertLayoutMode: String = "identity"
    public var packedExpertLayoutOrderPath: String?
    public var packedExpertLayoutOrderSha256: String?
    public var packedExpertLayoutStrategy: String?
    public var packedExpertReorderedLayerCount: Int = 0
    public var packedExpertLayoutAuditLogicalIDCount: Int = 0
    public var packedExpertLayoutOffsetValidationPassed: Bool = false
    public var remoteBytesDownloaded: UInt64 = 0
    public var remoteGapBytesDownloaded: UInt64 = 0
    public var remoteRangeRequests: UInt64 = 0
    public var remoteRangeRetries: UInt64 = 0
    public var remoteRangeStreamingSupported: Bool = false
    public var largestRemoteTransferBytes: Int = 0
    public var largestRemotePayloadHeapBytes: Int = 0
    public var remoteRepoID: String?
    public var remoteRequestedRevision: String?
    public var remoteResolvedCommit: String?
    public var remoteRetries: [RemoteRetryRecord] = []
    public var stalePartialsRemoved: [String] = []

    public init() {}

    public struct OutputFile {
        public let relativePath: String
        public let size: UInt64
        public let sha256: String
    }

    public struct RemoteRetryRecord {
        public let label: String
        public let attempt: Int
        public let detail: String
    }

    public func recordTile(bytes: Int) {
        byteCopyTiles &+= 1
        intentionalCopyBytes &+= UInt64(bytes)
    }

    public func recordWrite(bytes: Int) {
        outputBytesWritten &+= UInt64(bytes)
    }

    public func recordRead(bytes: Int) {
        sourceBytesRead &+= UInt64(bytes)
    }

    public func recordRemoteRetry(label: String, attempt: Int, detail: String) {
        remoteRangeRetries &+= 1
        remoteRetries.append(RemoteRetryRecord(label: label, attempt: attempt, detail: detail))
    }

    public func sampleRSS() {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        if kr == KERN_SUCCESS {
            let phys = UInt64(info.phys_footprint)
            if phys > peakRssBytes { peakRssBytes = phys }
            rssSamples.append(phys)
        }
    }

    public func toJSONData(outputDir: String) throws -> Data {
        var filesArr: [[String: Any]] = []
        for f in outputFiles {
            filesArr.append([
                "path": f.relativePath,
                "size": f.size,
                "sha256": f.sha256
            ])
        }
        var retryArr: [[String: Any]] = []
        for retry in remoteRetries {
            retryArr.append([
                "label": retry.label,
                "attempt": retry.attempt,
                "detail": retry.detail
            ])
        }
        var dict: [String: Any] = [
            "output_dir": outputDir,
            "peak_rss_bytes": peakRssBytes,
            "rss_sample_count": rssSamples.count,
            "source_bytes_read": sourceBytesRead,
            "output_bytes_written": outputBytesWritten,
            "intentional_copy_bytes": intentionalCopyBytes,
            "byte_copy_tiles": byteCopyTiles,
            "largest_scratch_bytes": largestScratchBytes,
            "whole_file_heap_buffers": wholeFileHeapBuffers,
            "platform_memmove_observed": platformMemmoveObserved,
            "malloc_count_observed": mallocCountObserved,
            "bit_width_overrides_honored": bitWidthOverridesHonored,
            "source_snapshot_sha256": sourceSnapshotSha256,
            "tensors_dropped_multimodal": tensorsDroppedMultimodal,
            "wall_time_s": wallTimeSeconds,
            "packed_expert_layout_mode": packedExpertLayoutMode,
            "packed_expert_reordered_layer_count": packedExpertReorderedLayerCount,
            "packed_expert_layout_audit_logical_id_count": packedExpertLayoutAuditLogicalIDCount,
            "packed_expert_layout_offset_validation_passed": packedExpertLayoutOffsetValidationPassed,
            "remote_bytes_downloaded": remoteBytesDownloaded,
            "remote_gap_bytes_downloaded": remoteGapBytesDownloaded,
            "remote_range_requests": remoteRangeRequests,
            "remote_range_retries": remoteRangeRetries,
            "remote_range_streaming_supported": remoteRangeStreamingSupported,
            "largest_remote_transfer_bytes": largestRemoteTransferBytes,
            "largest_remote_payload_heap_bytes": largestRemotePayloadHeapBytes,
            "remote_retries": retryArr,
            "stale_partials_removed": stalePartialsRemoved,
            "output_files": filesArr
        ]
        if let packedExpertLayoutOrderPath {
            dict["packed_expert_layout_order_path"] = packedExpertLayoutOrderPath
        }
        if let packedExpertLayoutOrderSha256 {
            dict["packed_expert_layout_order_sha256"] = packedExpertLayoutOrderSha256
        }
        if let packedExpertLayoutStrategy {
            dict["packed_expert_layout_strategy"] = packedExpertLayoutStrategy
        }
        if let remoteRepoID {
            dict["remote_repo_id"] = remoteRepoID
        }
        if let remoteRequestedRevision {
            dict["remote_requested_revision"] = remoteRequestedRevision
        }
        if let remoteResolvedCommit {
            dict["remote_resolved_commit"] = remoteResolvedCommit
        }
        return try JSONSerialization.data(withJSONObject: dict,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
