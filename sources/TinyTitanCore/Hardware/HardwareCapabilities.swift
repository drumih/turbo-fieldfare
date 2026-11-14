import Darwin
import Foundation
import IOKit
import Metal

/// What this machine can do, as a value rather than a property of the host.
///
/// The runtime used to ask the Metal device directly whether it supported
/// Apple10 tensor operations. That is a question about *hardware*, but it was
/// asked at the point a kernel was built, so a test could only run the answer
/// the host happened to give. Naming the answer once lets the kernel builders
/// take it as a parameter: the live probe is what production passes, and a
/// synthetic value is what a test passes.
///
/// `package`, not `public`: this is an internal seam (K0) and the CLI reads it
/// for `tinytitan doctor`, both targets living in this package. It is not part
/// of the library's supported surface.
package struct HardwareCapabilities: Equatable, Sendable {
    /// The family names the project's M1-M6 range uses. Apple10 is the first
    /// with the MPP tensor operations the 2D prefill kernel needs.
    package enum GPUFamily: String, Comparable, Sendable {
        case apple7
        case apple8
        case apple9
        case apple10
        case apple11

        private var rank: Int {
            switch self {
            case .apple7: 7
            case .apple8: 8
            case .apple9: 9
            case .apple10: 10
            case .apple11: 11
            }
        }

        package static func < (lhs: GPUFamily, rhs: GPUFamily) -> Bool {
            lhs.rank < rhs.rank
        }
    }

    /// `hw.model` (for example `Mac15,3`). Not the host name.
    package let chipModel: String
    /// `MTLDevice.name` (`Apple M3`), nil when no Metal device exists.
    package let gpuName: String?
    /// The highest Apple GPU family the device reports, nil without a device.
    package let gpuFamily: GPUFamily?
    /// `MTLDevice.supportsFamily(.apple10)`.
    package let supportsApple10TensorOps: Bool
    /// `MTLDevice.supportsFamily(.metal4)`.
    package let metal4Available: Bool
    /// Whether the I/O registry exposes an ANE node.
    package let anePresent: Bool
    /// `ProcessInfo.physicalMemory`.
    package let physicalMemoryBytes: UInt64
    /// `major.minor.patch` of the running macOS.
    package let osVersion: String

    package init(
        chipModel: String,
        gpuName: String?,
        gpuFamily: GPUFamily?,
        supportsApple10TensorOps: Bool,
        metal4Available: Bool,
        anePresent: Bool,
        physicalMemoryBytes: UInt64,
        osVersion: String
    ) {
        self.chipModel = chipModel
        self.gpuName = gpuName
        self.gpuFamily = gpuFamily
        self.supportsApple10TensorOps = supportsApple10TensorOps
        self.metal4Available = metal4Available
        self.anePresent = anePresent
        self.physicalMemoryBytes = physicalMemoryBytes
        self.osVersion = osVersion
    }

    /// The live answer for this process.
    ///
    /// `device` is the one the caller is about to build kernels on; omitted, the
    /// system default is probed. A machine with no Metal device is reported as
    /// lacking every GPU capability rather than being refused here, so a
    /// diagnostic can still describe the rest of the machine.
    package static func probe(device: MTLDevice? = nil) -> HardwareCapabilities {
        let resolved = device ?? MTLCreateSystemDefaultDevice()
        let version = ProcessInfo.processInfo.operatingSystemVersion
        return HardwareCapabilities(
            chipModel: chipModel(),
            gpuName: resolved?.name,
            gpuFamily: resolved.flatMap(gpuFamily(of:)),
            supportsApple10TensorOps: resolved?.supportsFamily(.apple10) ?? false,
            metal4Available: resolved?.supportsFamily(.metal4) ?? false,
            anePresent: anePresent(),
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            osVersion: "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)")
    }

    /// `sysctlbyname("hw.model")`, the machine identifier a calibration
    /// profile records. `unknown` when the sysctl is unavailable.
    package static func chipModel() -> String {
        var size = 0
        guard sysctlbyname("hw.model", nil, &size, nil, 0) == 0, size > 0 else {
            return "unknown"
        }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("hw.model", &bytes, &size, nil, 0) == 0 else {
            return "unknown"
        }
        let scalars = bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        guard let value = String(bytes: scalars, encoding: .utf8), !value.isEmpty
        else {
            return "unknown"
        }
        return value
    }

    /// The highest family the device answers for, tested in descending order
    /// so a newer GPU is not reported as the oldest family it also supports.
    private static func gpuFamily(of device: MTLDevice) -> GPUFamily? {
        let ordered: [(MTLGPUFamily, GPUFamily)] = [
            (.apple11, .apple11),
            (.apple10, .apple10),
            (.apple9, .apple9),
            (.apple8, .apple8),
            (.apple7, .apple7),
        ]
        return ordered.first { device.supportsFamily($0.0) }?.1
    }

    /// Whether the I/O registry has an ANE node.
    ///
    /// There is no ANE sysctl (`hw.optional.ane` is unknown on Apple Silicon),
    /// so this walks the `AppleARMIODevice` services and looks for the device
    /// tree's `ane@…` node. The node's address differs by chip, which is why
    /// the name is matched by prefix rather than by a chip-specific path.
    private static func anePresent() -> Bool {
        guard let matching = IOServiceMatching("AppleARMIODevice") else {
            return false
        }
        var iterator: io_iterator_t = 0
        guard
            IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator)
                == KERN_SUCCESS
        else {
            return false
        }
        defer { IOObjectRelease(iterator) }
        while true {
            let entry = IOIteratorNext(iterator)
            guard entry != 0 else { return false }
            defer { IOObjectRelease(entry) }
            var name = [CChar](repeating: 0, count: 128)
            guard IORegistryEntryGetName(entry, &name) == KERN_SUCCESS else {
                continue
            }
            let scalars = name.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
            guard let node = String(bytes: scalars, encoding: .utf8) else {
                continue
            }
            if node == "ane" || node.hasPrefix("ane@") {
                return true
            }
        }
    }
}
