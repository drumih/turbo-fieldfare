import Foundation
import TinyTitan
import TinyTitanCore

/// The report `tinytitan doctor` renders, as one value so the human and JSON
/// surfaces cannot disagree.
///
/// Every host fact arrives through `DoctorEnvironment`; building a report in a
/// test therefore reads no real machine data.
struct DoctorReport: Codable, Equatable, Sendable {
    static let schemaVersion = 1

    struct Machine: Codable, Equatable, Sendable {
        /// `hw.model`.
        let chip: String
        let gpuName: String?
        let gpuFamily: String?
        let apple10TensorOps: Bool
        let metal4: Bool
        let ane: Bool
        let ramBytes: UInt64
        let osVersion: String

        enum CodingKeys: String, CodingKey {
            case chip
            case gpuName = "gpu_name"
            case gpuFamily = "gpu_family"
            case apple10TensorOps = "apple10_tensor_ops"
            case metal4
            case ane
            case ramBytes = "ram_bytes"
            case osVersion = "os_version"
        }
    }

    struct Disk: Codable, Equatable, Sendable {
        let path: String
        let freeBytes: UInt64?
        let capacityBytes: UInt64?

        enum CodingKeys: String, CodingKey {
            case path
            case freeBytes = "free_bytes"
            case capacityBytes = "capacity_bytes"
        }
    }

    struct TensorOps: Codable, Equatable, Sendable {
        let mode: String
        /// `tensor-ops` or `fallback`.
        let path: String
        let reason: String
        let capabilityPresent: Bool

        enum CodingKeys: String, CodingKey {
            case mode
            case path
            case reason
            case capabilityPresent = "capability_present"
        }
    }

    struct Model: Codable, Equatable, Sendable {
        struct Fit: Codable, Equatable, Sendable {
            let weightsBytes: UInt64
            let kvBytes: UInt64
            let kvPrecision: String
            let contextTokens: Int
            let cacheFloorBytes: UInt64
            /// The cache the model's profile derives, which is what a load
            /// allocates; the floor above is the smallest legal cache.
            let expertCacheBytes: UInt64
            let totalBytes: UInt64
            let ramBytes: UInt64
            let fits: Bool
            /// `totalBytes / ramBytes`, for the human percentage.
            let utilization: Double

            enum CodingKeys: String, CodingKey {
                case weightsBytes = "weights_bytes"
                case kvBytes = "kv_bytes"
                case kvPrecision = "kv_precision"
                case contextTokens = "context_tokens"
                case cacheFloorBytes = "cache_floor_bytes"
                case expertCacheBytes = "expert_cache_bytes"
                case totalBytes = "total_bytes"
                case ramBytes = "ram_bytes"
                case fits
                case utilization
            }
        }

        struct PageCache: Codable, Equatable, Sendable {
            let policy: String
            let reason: String
        }

        let id: String
        let directory: String
        let family: String
        let weightBits: Int
        /// Allocated bytes on disk, as the catalog reports them.
        let sizeBytes: Int64
        let fit: Fit?
        let expertSlots: Int?
        let prefillChunkTokens: Int?
        let pageCache: PageCache?
        let note: String?

        enum CodingKeys: String, CodingKey {
            case id
            case directory
            case family
            case weightBits = "weight_bits"
            case sizeBytes = "size_bytes"
            case fit
            case expertSlots = "expert_slots"
            case prefillChunkTokens = "prefill_chunk_tokens"
            case pageCache = "page_cache"
            case note
        }
    }

    struct Calibration: Codable, Equatable, Sendable {
        let configured: Bool
        let path: String?
        /// `not-configured`, `valid`, or `invalid`.
        let status: String
        let detail: String?
        let measured: Bool?
        let source: String?
        let modelID: String?
        let contextTokens: Int?
        let machineMatches: Bool?

        static let notConfigured = Calibration(
            configured: false, path: nil, status: "not-configured",
            detail: nil, measured: nil, source: nil, modelID: nil,
            contextTokens: nil, machineMatches: nil)

        enum CodingKeys: String, CodingKey {
            case configured
            case path
            case status
            case detail
            case measured
            case source
            case modelID = "model_id"
            case contextTokens = "context_tokens"
            case machineMatches = "machine_matches"
        }
    }

    struct Process: Codable, Equatable, Sendable {
        let pid: Int32
        let name: String
    }

    struct Skipped: Codable, Equatable, Sendable {
        let name: String
        let reason: String
    }

    let schema: Int
    let machine: Machine
    let modelsDir: String
    let modelsDirExists: Bool
    let disk: Disk
    let tensorOps: TensorOps
    let models: [Model]
    let skipped: [Skipped]
    let calibration: Calibration
    let otherModelProcesses: [Process]
    let messages: [String]

    enum CodingKeys: String, CodingKey {
        case schema
        case machine
        case modelsDir = "models_dir"
        case modelsDirExists = "models_dir_exists"
        case disk
        case tensorOps = "tensor_ops"
        case models
        case skipped
        case calibration
        case otherModelProcesses = "other_model_processes"
        case messages
    }
}

/// Every fact the report is built from. Production fills it with live probes;
/// a test fills it with synthetic capabilities, a temporary `models/` tree and
/// fixed disk/process lists.
struct DoctorEnvironment: Sendable {
    let capabilities: HardwareCapabilities
    let modelsDirectory: URL
    let environment: [String: String]
    let maxContextTokens: Int
    let disk: DoctorReport.Disk
    let otherProcesses: [DoctorReport.Process]
    let calibration: DoctorReport.Calibration
}
