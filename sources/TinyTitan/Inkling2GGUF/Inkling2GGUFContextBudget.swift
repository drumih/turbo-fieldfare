import Foundation

/// How long a context an Inkling2 GGUF session can afford on this device
/// (TT-120).
///
/// The full-attention visits keep one K and one V row per position for the
/// whole window, allocated up front, while the sliding visits keep a fixed ring.
/// The weights are mapped from the file, and the GPU must hold both within its
/// working set: on a 24 GB M3 that is 17.76 GiB against a 16.42 GiB Q4 file,
/// and a 64K window there failed its first request with Metal's
/// `kIOGPUCommandBufferCallbackErrorOutOfMemory`. So the window is the smallest
/// of what was asked, the model's declared window (`modelMaxContext`), and
/// what the working set leaves after the weights and a reserve for the fixed
/// buffers.
package enum Inkling2GGUFContextBudget {
    /// Working-set bytes kept back for everything that does not scale with the
    /// window: the sliding rings, the hidden-vector pool, chunk scratch and
    /// the logits. Measured on the Q4 file: Metal holds 556 MB at a 4K window
    /// and 940 MB at 16K, so 32 KB a position and about 420 MB fixed; the
    /// reserve is that plus about 90 MB of headroom.
    package static let reserveBytes = 512 << 20

    /// The smallest window worth serving; below it the device is too small.
    package static let minimumContext = 4_096

    /// KV bytes one position costs across the full-attention visits: K and V,
    /// fp16, `kvHeads x headDimension` each.
    static func bytesPerPosition(_ manifest: Inkling2GGUFManifest) -> Int {
        var visits = 0
        for visit in 0..<manifest.unrolledLayers
        where !manifest.layers[manifest.storedLayer(forUnrolled: visit)].isSliding {
            visits += 1
        }
        let kvWidth = manifest.headDimension * manifest.layers[0].kvHeads
        return visits * kvWidth * 2 * 2
    }

    /// The window to allocate: `requested`, capped by the model's window and by what
    /// `workingSet` leaves after `weightBytes` and the reserve, rounded down to
    /// a multiple of 1,024 and never below `minimumContext`.
    package static func affordable(
        requested: Int, model: Int, workingSet: Int, weightBytes: Int, bytesPerPosition: Int
    ) -> Int {
        let free = workingSet - weightBytes - reserveBytes
        let fits = free > 0 ? free / max(bytesPerPosition, 1) : 0
        let window = min(requested, model, fits) / 1_024 * 1_024
        return max(window, minimumContext)
    }
}
