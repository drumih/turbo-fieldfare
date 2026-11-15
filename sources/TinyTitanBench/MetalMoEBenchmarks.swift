//
//  MetalMoEBenchmarks.swift
//  TinyTitanBench
//
//  The routed-MoE decode kernel benchmark and the shader-side expert-offset
//  mirror it needs.
//
//  Split out of `MetalBenchmarks.swift` (2026-09-28) under the 500-line-per-file
//  rule (Task 8 of the cleanup runbook) as pure code motion; the
//  `lint:allow-long` marker travelled with its declaration.

import Foundation
import Metal
import TinyTitan
import TinyTitanCore

extension TinyTitanBench {

    /// Shader-side `ExpertOffsets` mirror: 9 packed UInt32 in the same order.
    private struct MoEBenchOffsets {
        var gateW: UInt32
        var gateS: UInt32
        var gateB: UInt32
        var upW: UInt32
        var upS: UInt32
        var upB: UInt32
        var downW: UInt32
        var downS: UInt32
        var downB: UInt32
    }

    /// Routed-MoE decode kernels at the real 4-bit shapes. Weights are the
    /// metric: phase-1 reads gate+up (8 x 2 x F*D/2 bytes), phase-2 reads
    /// down (8 x D*F/2 bytes). The combined "moe" mode dispatches both in
    /// one command buffer, mirroring the decode routedCB.
    /// lint:allow-long TinyTitanBench is a development harness, not a shipped
    /// product: each run* is one linear measurement script whose setup,
    /// dispatch and reporting only make sense read top to bottom.
    static func runMoE(
        kernelName: String,
        iterations: Int,
        context: MetalContext
    ) throws {
        let device = context.device
        // Default shape is Qwen 3.6 35B-A3B. TINYTITAN_BENCH_MOE_SHAPE=qwen38
        // selects Qwen3.8-Flash-Next's routed expert (D 2560, F 640, top-10),
        // the shape the decode profile's moe_* numbers come from.
        let qwen38 = ProcessInfo.processInfo.environment["TINYTITAN_BENCH_MOE_SHAPE"] == "qwen38"
        let D: UInt32 = qwen38 ? 2560 : 2048  // hiddenSize
        let F: UInt32 = qwen38 ? 640 : 512  // moeIntermediateSize
        let topK: UInt32 = qwen38 ? 10 : 8
        let groupCount = Int(D) / 64  // kMoEGroupSize = 64 elements

        // Per-blob 4-bit layout: gate_W, gate_s, gate_b, up_W, up_s, up_b,
        // down_W, down_s, down_b.
        let fD2 = Int(F) * Int(D) / 2
        let groups = Int(F) * groupCount * 2
        let gateWOff = 0
        let gateSOff = fD2
        let gateBOff = gateSOff + groups
        let upWOff = gateBOff + groups
        let upSOff = upWOff + fD2
        let upBOff = upSOff + groups
        let downWOff = upBOff + groups
        let downSOff = downWOff + Int(D) * Int(F) / 2
        let downBOff = downSOff + groups
        let blobBytes = downBOff + groups

        func makeBuffer(_ bytes: Int, _ value: UInt8) throws -> MTLBuffer {
            guard
                let buf = device.makeBuffer(
                    length: bytes,
                    options: .storageModeShared)
            else {
                throw BenchHarnessError.metalObjectUnavailable("buffer of \(bytes) bytes")
            }
            memset(buf.contents(), Int32(value), bytes)
            return buf
        }
        var blobs: [MTLBuffer] = []
        for i in 0..<Int(topK) {
            let blob = try makeBuffer(blobBytes, UInt8(0x11 + i))
            // Scale/bias regions as bf16 0x3C3C (~0.011) rather than the
            // byte pattern (~1e-38), so the activations are non-zero and
            // acts_fnv actually compares kernel variants.
            for off in [gateSOff, gateBOff, upSOff, upBOff, downSOff, downBOff] {
                memset(blob.contents().advanced(by: off), 0x3C, groups)
            }
            blobs.append(blob)
        }
        // Non-uniform activation pattern so staging/indexing bugs surface:
        // the uniform fill used before masked wrong-element reads.
        let x = try makeBuffer(Int(D) * 2, 0)
        let xPtr = x.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<Int(D) {
            xPtr[i] = Float16(Float(i + 1) * 0.001).bitPattern
        }
        let acts = try makeBuffer(Int(topK) * Int(F) * 2, 0)
        let routingW = try makeBuffer(Int(topK) * 2, 0x3F)
        let residual = try makeBuffer(Int(D) * 2, 0x33)
        let y = try makeBuffer(Int(D) * 2, 0)

        // RoutedBlobs arg buffer: 8 device pointers (shared memory, so the
        // host address is the GPU address).
        guard
            let argBuf = device.makeBuffer(
                length: Int(topK) * 8,
                options: .storageModeShared)
        else {
            fatalError("arg buffer alloc failed")
        }
        let argPtr = argBuf.contents().assumingMemoryBound(to: UnsafeMutableRawPointer?.self)
        for i in 0..<Int(topK) {
            argPtr[i] = blobs[i].contents()
        }

        var offsets = MoEBenchOffsets(
            gateW: UInt32(gateWOff), gateS: UInt32(gateSOff), gateB: UInt32(gateBOff),
            upW: UInt32(upWOff), upS: UInt32(upSOff), upB: UInt32(upBOff),
            downW: UInt32(downWOff), downS: UInt32(downSOff), downB: UInt32(downBOff))

        // Variant dispatch: r4/r16 change rows-per-threadgroup, xsh8/16 stage
        // the activation in threadgroup memory. The production kernel is the
        // 16-row threadgroup-staged layout, so all modes dispatch 16 rows.
        let phase1Variant: String?
        let phase1RowsPerTG: Int
        switch kernelName {
        case "moe_phase1_r8":
            phase1Variant = "moe_phase1_gate_up_act_u16load_r8"
            phase1RowsPerTG = 8
        case "moe_phase1_r16":
            phase1Variant = "moe_phase1_gate_up_act_u16load_r16"
            phase1RowsPerTG = 16
        case "moe_phase1_xsh8":
            phase1Variant = "moe_phase1_gate_up_act_u16load"
            phase1RowsPerTG = 16
        case "moe_phase1_xsh16":
            phase1Variant = "moe_phase1_gate_up_act_u16load"
            phase1RowsPerTG = 16
        case "moe_phase1_v2":
            // Two rows per simdgroup, 16 simdgroups: 32 rows per 512 threads.
            phase1Variant = "moe_phase1_gate_up_act_u16load_v2"
            phase1RowsPerTG = 32
        default:
            // The production kernel is the 16-row threadgroup-staged layout.
            phase1Variant = nil
            phase1RowsPerTG = 16
        }
        let phase1Threads = kernelName == "moe_phase1_v2" ? 512 : phase1RowsPerTG * 32
        let phase1Kernel = phase1Variant ?? "moe_phase1_gate_up_act_u16load"

        // TINYTITAN_BENCH_MOE_SPECIALIZE=1 builds the phase-1 pipeline with the
        // runtime's function constants (D, F, top-k, silu, host-gated I/O);
        // the unspecialized kernel reads its shape from buffers and measures
        // ~35 GB/s where the specialized one moves the same blobs at ~60.
        let specialize =
            ProcessInfo.processInfo.environment["TINYTITAN_BENCH_MOE_SPECIALIZE"] == "1"
        let phase1Constants: [MetalFunctionConstant] =
            specialize
            ? [
                MetalFunctionConstant(index: 0, value: .uint32(D)),
                MetalFunctionConstant(index: 1, value: .uint32(F)),
                MetalFunctionConstant(index: 2, value: .uint32(topK)),
                MetalFunctionConstant(index: 3, value: .bool(true)),
                MetalFunctionConstant(index: 4, value: .bool(true)),
                MetalFunctionConstant(index: 6, value: .bool(false)),
            ] : []
        let phase1PSO = try context.pipeline(
            phase1Kernel,
            constants: phase1Constants,
            maxTotalThreadsPerThreadgroup: phase1Threads)
        let phase2PSO = try context.pipeline(
            topK == 8 ? "moe_phase2_down_reduce_k8" : "moe_phase2_down_reduce_kn",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        let subsetPSO = try context.pipeline(
            kernelName == "moe_phase1_subset"
                ? "moe_phase1_gate_up_act_subset_u16load"
                : "moe_phase1_gate_up_act_u16load",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)

        let phase1Groups = (Int(topK) * Int(F) + phase1RowsPerTG - 1) / phase1RowsPerTG
        let phase1Bytes = UInt64(Int(topK) * 2 * Int(F) * Int(D) / 2)
        let phase2Bytes = UInt64(Int(topK) * Int(D) * Int(F) / 2)

        var Dv = D
        var Fv = F
        var TK = topK

        let cb = try requireCommandBuffer(context.queue)
        guard let enc = cb.makeComputeCommandEncoder() else {
            fatalError("could not create compute encoder")
        }
        let runPhase1 = kernelName == "moe" || kernelName.hasPrefix("moe_phase1")
        let phase1TG = MTLSize(width: phase1Threads, height: 1, depth: 1)
        let runPhase2 = kernelName == "moe_phase2" || kernelName == "moe"
        let runSubset = kernelName == "moe_phase1_subset"
        // active-slot buffer for the subset mode (all 8 experts active).
        var activeSlots = [UInt32](0..<topK)
        let activeSlotsBuf = try makeBuffer(Int(topK) * MemoryLayout<UInt32>.size, 0)
        activeSlotsBuf.contents().copyMemory(
            from: &activeSlots,
            byteCount: Int(topK) * MemoryLayout<UInt32>.size)
        var activeCount = topK
        for _ in 0..<iterations {
            if runSubset {
                enc.setComputePipelineState(subsetPSO)
                enc.setBuffer(argBuf, offset: 0, index: 0)
                enc.setBytes(&offsets, length: MemoryLayout<MoEBenchOffsets>.stride, index: 1)
                enc.setBuffer(x, offset: 0, index: 2)
                enc.setBuffer(acts, offset: 0, index: 3)
                enc.setBytes(&Dv, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBytes(&Fv, length: MemoryLayout<UInt32>.size, index: 5)
                enc.setBytes(&TK, length: MemoryLayout<UInt32>.size, index: 6)
                enc.setBuffer(activeSlotsBuf, offset: 0, index: 7)
                enc.setBytes(&activeCount, length: MemoryLayout<UInt32>.size, index: 8)
                enc.dispatchThreadgroups(
                    MTLSize(width: (Int(activeCount * F) + 15) / 16, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            }
            if runPhase1 {
                enc.setComputePipelineState(phase1PSO)
                enc.setBuffer(argBuf, offset: 0, index: 0)
                enc.setBytes(&offsets, length: MemoryLayout<MoEBenchOffsets>.stride, index: 1)
                enc.setBuffer(x, offset: 0, index: 2)
                enc.setBuffer(acts, offset: 0, index: 3)
                enc.setBytes(&Dv, length: MemoryLayout<UInt32>.size, index: 4)
                enc.setBytes(&Fv, length: MemoryLayout<UInt32>.size, index: 5)
                enc.setBytes(&TK, length: MemoryLayout<UInt32>.size, index: 6)
                enc.dispatchThreadgroups(
                    MTLSize(width: phase1Groups, height: 1, depth: 1),
                    threadsPerThreadgroup: phase1TG)
            }
            if runPhase2 {
                enc.setComputePipelineState(phase2PSO)
                enc.setBuffer(argBuf, offset: 0, index: 0)
                enc.setBytes(&offsets, length: MemoryLayout<MoEBenchOffsets>.stride, index: 1)
                enc.setBuffer(acts, offset: 0, index: 2)
                enc.setBuffer(routingW, offset: 0, index: 3)
                enc.setBuffer(residual, offset: 0, index: 4)
                enc.setBuffer(y, offset: 0, index: 5)
                enc.setBytes(&Dv, length: MemoryLayout<UInt32>.size, index: 6)
                enc.setBytes(&Fv, length: MemoryLayout<UInt32>.size, index: 7)
                enc.dispatchThreadgroups(
                    MTLSize(width: Int(D), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("COMMAND BUFFER ERROR: \(err)")
        }

        let totalSeconds = cb.gpuEndTime - cb.gpuStartTime
        let perIteration = totalSeconds / Double(iterations)
        let bytes: UInt64 = (runPhase1 ? phase1Bytes : 0) + (runPhase2 ? phase2Bytes : 0)
        let gbPerSec = Double(bytes) / perIteration / 1_000_000_000
        let theoretical = 100.0
        // Correctness probe: FNV-1a over the acts buffer (the phase-1 output)
        // and the y buffer (the phase-2 output).
        var hash: UInt32 = 0x811c_9dc5
        let actsPtr = acts.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<min(acts.length, 8192) {
            hash ^= UInt32(actsPtr[i])
            hash &*= 0x0100_0193
        }
        var yHash: UInt32 = 0x811c_9dc5
        let yPtr = y.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<min(y.length, 8192) {
            yHash ^= UInt32(yPtr[i])
            yHash &*= 0x0100_0193
        }
        let actsHalf = acts.contents().assumingMemoryBound(to: UInt16.self)
        let sample = (0..<min(16, acts.length / 2)).map { String(format: "%04x", actsHalf[$0]) }
            .joined(separator: " ")
        print(
            "kernel=\(kernelName) iterations=\(iterations) "
                + "total=\(String(format: "%.4f", totalSeconds))s "
                + "per_launch=\(String(format: "%.2f", perIteration * 1_000_000))us")
        print(
            "bytes/launch=\(bytes) (phase1=\(phase1Bytes) phase2=\(phase2Bytes)) "
                + "achieved=\(String(format: "%.1f", gbPerSec)) GB/s "
                + "efficiency=\(String(format: "%.0f", gbPerSec / theoretical * 100))% of ~100 GB/s peak "
                + "acts_fnv=\(String(format: "%08x", hash)) y_fnv=\(String(format: "%08x", yHash)) "
                + "acts_sample=\(sample)")
    }
}
