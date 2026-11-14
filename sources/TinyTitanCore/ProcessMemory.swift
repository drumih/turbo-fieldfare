import Darwin
import Foundation

/// This process's `phys_footprint`, in MiB, for memory diagnostics.
///
/// `phys_footprint` is the kernel's own accounting — dirty pages, compressed
/// pages and IOKit mappings — which is what a memory guard has to watch. RSS
/// alone misses the Neural Engine's E5RT arenas, and those are exactly what a
/// Core ML prefill can leave behind after `releaseModels()` claims to drop it.
public enum ProcessMemory {
    /// - Returns: the footprint in MiB, or -1 when `task_info` refuses.
    public static func physFootprintMiB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return -1 }
        return Double(info.phys_footprint) / 1_048_576.0
    }
}
