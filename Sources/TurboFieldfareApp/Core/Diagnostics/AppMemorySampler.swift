import Darwin
import Darwin.Mach
import Foundation

public final class AppMemorySampler: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private var residentPeak: UInt64 = 0
    private var occupiedPeak: UInt64 = 0
    private let processFootprint: @Sendable () -> UInt64?

    public init() {
        self.processFootprint = { Self.readProcessFootprint() }
    }

    init(processFootprint: @escaping @Sendable () -> UInt64?) {
        self.processFootprint = processFootprint
    }

    public func resetPeak() {
        lock.lock()
        peak = 0
        residentPeak = 0
        occupiedPeak = 0
        lock.unlock()
    }

    public func sample() -> UInt64? {
        guard let current = processFootprint() else { return nil }
        lock.lock()
        if current > peak { peak = current }
        lock.unlock()
        return current
    }

    /// Bytes this process has resident, including the file-backed pages of the
    /// mapped weights. `phys_footprint` deliberately excludes those — they are
    /// clean and the kernel can drop them — which is why a freshly loaded 26B
    /// model reports about 160 MB. That is true and also the opposite of what
    /// someone reads it as, so both numbers are reported.
    public func residentSample() -> UInt64? {
        guard let current = Self.readResidentSize() else { return nil }
        lock.lock()
        if current > residentPeak { residentPeak = current }
        lock.unlock()
        return current
    }

    /// Peak resident bytes since the last reset. Tracked separately from the
    /// footprint peak because the two measure different things and showing one
    /// beside the other's current value invites a false comparison.
    public var peakResidentBytes: UInt64? {
        lock.lock()
        let value = residentPeak
        lock.unlock()
        return value == 0 ? nil : value
    }

    /// What the process is actually occupying: the bytes charged to it plus the
    /// file-backed pages it has resident.
    ///
    /// Neither half alone answers the question. `phys_footprint` counts
    /// anonymous, wired and compressed memory but excludes clean mapped pages,
    /// so it cannot see the weights. `resident_size` counts mapped pages but
    /// also re-counts the anonymous ones already in the footprint. Adding the
    /// footprint to the *external* (file-backed) resident bytes counts each
    /// page once: about 2 GB of runtime plus about 1 GB of image tower reads as
    /// about 3 GB, which is what is really in RAM.
    public func occupiedSample() -> UInt64? {
        guard let info = Self.readVMInfo() else { return nil }
        let total = UInt64(info.phys_footprint) + UInt64(info.external)
        lock.lock()
        if total > occupiedPeak { occupiedPeak = total }
        lock.unlock()
        return total
    }

    /// File-backed resident bytes on their own: the mapped weights and image
    /// tower, which is the part the footprint can never show.
    public func mappedResidentSample() -> UInt64? {
        Self.readVMInfo().map { UInt64($0.external) }
    }

    public var peakOccupiedBytes: UInt64? {
        lock.lock()
        let value = occupiedPeak
        lock.unlock()
        return value == 0 ? nil : value
    }

    private static func readVMInfo() -> task_vm_info_data_t? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info : nil
    }

    private static func readResidentSize() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return UInt64(info.resident_size)
    }

    private static func readProcessFootprint() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return UInt64(info.phys_footprint)
    }

    public var peakBytes: UInt64? {
        lock.lock()
        let value = peak
        lock.unlock()
        return value == 0 ? nil : value
    }
}
