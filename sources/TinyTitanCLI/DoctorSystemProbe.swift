import Darwin
import Foundation
import TinyTitanCore

/// The two host probes `doctor` needs beyond `HardwareCapabilities`: free space
/// where the model store lives, and other TinyTitan model processes.
enum DoctorSystemProbe {
    /// Free and total space of the volume holding `directory`, or its nearest
    /// existing ancestor when the store has not been created yet.
    static func disk(for directory: URL) -> DoctorReport.Disk {
        let target = existingAncestor(of: directory)
        let keys: Set<URLResourceKey> = [
            .volumeAvailableCapacityForImportantUsageKey,
            .volumeAvailableCapacityKey,
            .volumeTotalCapacityKey,
        ]
        let values = try? target.resourceValues(forKeys: keys)
        let free =
            values?.volumeAvailableCapacityForImportantUsage.map {
                UInt64(max(0, $0))
            }
            ?? values?.volumeAvailableCapacity.map { UInt64(max(0, $0)) }
        let capacity = values?.volumeTotalCapacity.map { UInt64(max(0, $0)) }
        return DoctorReport.Disk(
            path: target.path, freeBytes: free, capacityBytes: capacity)
    }

    /// Other TinyTitan executables that can hold a loaded model. The CLI running
    /// `doctor` is excluded by pid; a `tinytitan` launcher shell is not a model
    /// process and is not reported.
    static func otherProcesses(
        currentPID: pid_t = getpid()
    ) -> [DoctorReport.Process] {
        let modelProcesses: Set<String> = [
            "TinyTitanServer", "TinyTitanCLI", "TinyTitanBench",
        ]
        var pids = [pid_t](repeating: 0, count: 4_096)
        let count = proc_listallpids(
            &pids, Int32(pids.count * MemoryLayout<pid_t>.stride))
        guard count > 0 else { return [] }
        var found: [DoctorReport.Process] = []
        for index in 0..<Int(count) {
            let pid = pids[index]
            guard pid > 0, pid != currentPID else { continue }
            var path = [CChar](repeating: 0, count: 4_096)
            let length = proc_pidpath(pid, &path, UInt32(path.count))
            guard length > 0 else { continue }
            let scalars = path.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
            guard
                let executable = String(bytes: scalars, encoding: .utf8),
                !executable.isEmpty
            else { continue }
            let name = URL(fileURLWithPath: executable).lastPathComponent
            guard modelProcesses.contains(name) else { continue }
            found.append(DoctorReport.Process(pid: pid, name: name))
        }
        return found.sorted { $0.pid < $1.pid }
    }

    private static func existingAncestor(of directory: URL) -> URL {
        let fileManager = FileManager.default
        var candidate = directory.standardizedFileURL
        while !fileManager.fileExists(atPath: candidate.path) {
            let parent = candidate.deletingLastPathComponent()
            if parent.path == candidate.path { break }
            candidate = parent
        }
        return candidate
    }
}
