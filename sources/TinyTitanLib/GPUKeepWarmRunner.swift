import Foundation
import Metal
import Synchronization
import TinyTitanCore

/// An opt-in second command queue that issues one micro-dispatch on a fixed
/// period for the duration of a generation.
///
/// §5e measured the identical head kernel at 9.42 ms back-to-back and 12.73 ms
/// with 50 ms between launches and closed the lever on the assertion that
/// nothing in the engine controls GPU clock state. §5bk tested that premise with
/// this shape of instrument and falsified it in one direction: a concurrent
/// stream of dispatches too small to matter made the *measured* launch's device
/// time shorter, by 5.88 % at a 200 us period in both rounds. This is the
/// engine-side arm of the same question, and unlike the bench it is measured
/// against a real token.
///
/// The spin is the bench's shape: eight rows of a self-contained 8-bit affine
/// GEMV, one threadgroup, 256 threads, 20 KiB of weights. Buffers are private to
/// the runner and no model tensor is touched, so generation output is identical
/// with the knob on or off. The stream queue runs at `.utility` QoS because a
/// thread that wakes ~1600 times a second competes for the host cores the expert
/// I/O threads use — §5v priced that competition at -15 % when it was the
/// engine's own worker spinning.
/// unchecked-invariant: `running` is an atomic cross-thread flag.
/// `dispatches` is written only on `serial` and read only after `stop()` has
/// joined it; every Metal object the worker touches is created in `init` and
/// never mutated afterwards, and `serial` is the only encoder of the buffers it owns.
package final class GPUKeepWarmRunner: @unchecked Sendable {
    private static let spinRows: UInt32 = 8
    private static let spinColumns: UInt32 = 2560
    private static let weightBits: Int = 8
    private static let groupSize: Int = 64
    private static let threadsPerThreadgroup = 256

    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let weights: MTLBuffer
    private let scales: MTLBuffer
    private let biases: MTLBuffer
    private let input: MTLBuffer
    private let output: MTLBuffer
    private let intervalUs: Int
    private let serial = DispatchQueue(label: "tinytitan.keepwarm", qos: .utility)
    private let finished = DispatchSemaphore(value: 0)
    private let running = Atomic<Bool>(false)
    private var dispatches = 0
    private var startedAt = 0.0

    /// The interval is validated by the knob, not by the caller, so an operator
    /// who sets a typo gets a refusal naming the range rather than an arm that
    /// silently reads as "off".
    package static func startIfEnabled(
        context: MetalContext,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> GPUKeepWarmRunner? {
        guard let intervalUs = try RuntimeGPUKeepWarm.environmentValue(environment) else {
            return nil
        }
        let runner = try GPUKeepWarmRunner(context: context, intervalUs: intervalUs)
        runner.start()
        return runner
    }

    package init(context: MetalContext, intervalUs: Int) throws {
        try RuntimeGPUKeepWarm.validate(intervalUs)
        self.intervalUs = intervalUs
        guard let queue = context.device.makeCommandQueue() else {
            throw MetalError.noQueue
        }
        self.queue = queue
        self.pipeline = try context.pipeline(
            "affine_quant_gemv_simd",
            constants: [
                MetalFunctionConstant(index: 100, value: .uint32(UInt32(Self.weightBits)))
            ],
            maxTotalThreadsPerThreadgroup: Self.threadsPerThreadgroup)
        let rows = Int(Self.spinRows)
        let columns = Int(Self.spinColumns)
        let groups = columns / Self.groupSize
        self.weights = try Self.makeBuffer(
            context.device, bytes: rows * columns * Self.weightBits / 8, fill: 0x5A)
        self.scales = try Self.makeBuffer(
            context.device, bytes: rows * groups * 2, fill: 0x3C)
        self.biases = try Self.makeBuffer(
            context.device, bytes: rows * groups * 2, fill: 0x00)
        self.input = try Self.makeBuffer(context.device, bytes: columns * 2, fill: 0x00)
        self.output = try Self.makeBuffer(context.device, bytes: rows * 2, fill: 0x00)
    }

    private static func makeBuffer(
        _ device: MTLDevice, bytes: Int, fill: Int32
    ) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed("keepwarm \(bytes) B")
        }
        memset(buffer.contents(), fill, bytes)
        return buffer
    }

    package func start() {
        running.store(true, ordering: .releasing)
        startedAt = Date().timeIntervalSince1970
        serial.async { [weak self] in
            guard let self else { return }
            while self.running.load(ordering: .acquiring) {
                guard let cb = self.queue.makeCommandBuffer() else { break }
                self.encode(cb)
                cb.commit()
                // Waiting bounds how many command buffers are in flight; the spin
                // itself is microseconds, so this thread sleeps for essentially
                // the whole interval.
                cb.waitUntilCompleted()
                self.dispatches += 1
                Thread.sleep(forTimeInterval: Double(self.intervalUs) / 1_000_000)
            }
            self.finished.signal()
        }
    }

    /// Joins the stream thread and writes one stderr line carrying the *achieved*
    /// rate, because §5bk's nominal 200 us period was 616 us on the wire and a
    /// live arm can only be read against what it actually issued.
    package func stop() {
        running.store(false, ordering: .releasing)
        finished.wait()
        let elapsed = Date().timeIntervalSince1970 - startedAt
        let issued = dispatches
        guard issued > 0, elapsed > 0 else { return }
        FileHandle.standardError.write(
            Data(
                String(
                    format:
                        "TinyTitan keepwarm interval_us=%d dispatches=%d elapsed_ms=%.0f achieved_period_us=%.0f",
                    intervalUs, issued, elapsed * 1000, elapsed * 1_000_000 / Double(issued)
                )
                .utf8) + Data([0x0A]))
    }

    private func encode(_ cb: MTLCommandBuffer) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        var rows = Self.spinRows
        var columns = Self.spinColumns
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(weights, offset: 0, index: 0)
        enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(biases, offset: 0, index: 2)
        enc.setBuffer(input, offset: 0, index: 3)
        enc.setBuffer(output, offset: 0, index: 4)
        enc.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: Self.threadsPerThreadgroup, height: 1, depth: 1))
        enc.endEncoding()
    }
}
