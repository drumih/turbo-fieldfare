import Foundation
import TinyTitanKernelsC

/// `y = W · x` for an affine-INT8 matrix, on the CPU.
///
/// This is the decode primitive of the side-engine: the small resident model
/// that keeps memory and checks replies runs entirely here, on cores the
/// main engine leaves idle. Measured on this machine while a 35B generates,
/// TinyTitanServer uses 0.20 of one core out of eight — the GPU is the busy
/// resource, and the CPU is free.
///
/// Two facts shape everything below.
///
/// **Decode is bandwidth-bound, not arithmetic-bound.** One token reads every
/// weight exactly once: about 2.4 GB at 8-bit for a 2B model. Arranging the
/// arithmetic differently cannot beat the memory system, so the kernel reads
/// each row sequentially and the threading splits *rows*, never columns —
/// two threads walking one row would read it twice.
///
/// **Threads are not the same on an M3.** Four performance cores share one
/// matrix unit with four efficiency cores that are much slower at this. Work
/// is dispatched at `.userInitiated`, which the scheduler places on the
/// performance cores, and the default width is the performance-core count
/// rather than `activeProcessorCount` — on the 4+4 M3, asking for eight
/// gives four fast workers and four slow ones, and the fast four then wait.
public enum Int8AffineGEMV {

    /// Performance cores, which is what this work wants. `perflevel0` is the
    /// performance cluster on Apple silicon; on a machine that does not
    /// report one, half the cores is a better guess than all of them.
    public static let preferredThreads: Int = {
        var count: Int32 = 0
        var size = MemoryLayout<Int32>.size
        if sysctlbyname("hw.perflevel0.logicalcpu", &count, &size, nil, 0) == 0, count > 0 {
            return Int(count)
        }
        return max(1, ProcessInfo.processInfo.activeProcessorCount / 2)
    }()

    /// Single-threaded. `weights` is `rows * n` bytes, `scales` and `biases`
    /// are `rows * (n / 64)` BF16 bit patterns each, all row-major.
    /// The contract both entry points share with the C kernel behind them.
    ///
    /// `n` is the input width, and the kernel derives two things from it: the
    /// scale/bias groups per row, `n / 64`, and the packed bytes per row,
    /// `n / 2` at 4 bits or `n` at 8. A width that is not a whole number of
    /// groups makes those two disagree — every row after the first is read from
    /// the wrong byte offset and `x` is truncated to whole groups — so the result
    /// is silently wrong rather than short. The header documents the requirement
    /// and nothing enforced it; every Metal wrapper does check.
    @inline(__always)
    static func requireWholeGroups(_ n: Int, _ what: String) {
        precondition(
            n % Quantization.groupSize == 0,
            "\(what): input width \(n) is not a whole number of "
                + "\(Quantization.groupSize)-element groups")
    }

    @inline(__always)
    public static func apply(
        weights: UnsafePointer<UInt8>,
        scales: UnsafePointer<UInt16>,
        biases: UnsafePointer<UInt16>,
        x: UnsafePointer<Float>,
        rows: Int,
        n: Int,
        out: UnsafeMutablePointer<Float>
    ) {
        requireWholeGroups(n, "Int8AffineGEMV.apply")
        tinytitan_int8_affine_gemv(weights, scales, biases, x, rows, n, out)
    }

    /// Split across performance cores by row range.
    ///
    /// Rows are independent, so each worker advances every pointer together
    /// and writes a disjoint slice of `out`. There is no accumulation across
    /// workers and therefore no reduction order to get wrong: the threaded
    /// result is bit-identical to the single-threaded one, which the tests
    /// assert rather than assume.
    ///
    /// Below `minimumRowsPerThread` the dispatch costs more than it saves, so
    /// small matrices run inline.
    public static let minimumRowsPerThread = 64

    public static func threaded(
        weights: UnsafePointer<UInt8>,
        scales: UnsafePointer<UInt16>,
        biases: UnsafePointer<UInt16>,
        x: UnsafePointer<Float>,
        rows: Int,
        n: Int,
        out: UnsafeMutablePointer<Float>,
        threads: Int = preferredThreads
    ) {
        requireWholeGroups(n, "Int8AffineGEMV.threaded")
        let groups = n / Quantization.groupSize
        let usable = max(1, min(threads, rows / minimumRowsPerThread))
        if usable == 1 {
            tinytitan_int8_affine_gemv(weights, scales, biases, x, rows, n, out)
            return
        }
        let chunk = (rows + usable - 1) / usable
        // `concurrentPerform` returns only after every iteration has finished,
        // so the buffers outlive the closure; Swift cannot see that through the
        // call and unsafe pointers are not `Sendable`, which is what these
        // bindings record. The workers write disjoint row ranges, so no byte of
        // `out` is shared and no read races a write.
        nonisolated(unsafe) let weights = weights
        nonisolated(unsafe) let scales = scales
        nonisolated(unsafe) let biases = biases
        nonisolated(unsafe) let x = x
        nonisolated(unsafe) let out = out
        DispatchQueue.concurrentPerform(iterations: usable) { slice in
            let first = slice * chunk
            guard first < rows else { return }
            let count = min(chunk, rows - first)
            tinytitan_int8_affine_gemv(
                weights + first * n,
                scales + first * groups,
                biases + first * groups,
                x, count, n, out + first)
        }
    }
}
