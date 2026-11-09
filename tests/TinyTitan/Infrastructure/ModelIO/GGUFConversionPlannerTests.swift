import Foundation
import Testing

@testable import TinyTitan

@Suite struct GGUFConversionPlannerTests {
    private static var realModelPath: String? {
        ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"]
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TINYTITAN_GGUF_MODEL"] != nil))
    func plansTheWholeInstallAndAgreesWithTheManifest() throws {
        let path = try #require(Self.realModelPath)
        let model = try GGUFModelReader.read(modelAt: URL(fileURLWithPath: path))
        let plan = try GGUFConversionPlanner.plan(from: model)

        #expect(plan.resident.count == 1_080)
        #expect(plan.expertBytes == 68_048_388_096)
        #expect(plan.residentBytes > 0)
        // The role classes the manifest's slots describe must get those widths.
        let byName = Dictionary(uniqueKeysWithValues: plan.resident.map { ($0.name, $0) })
        #expect(byName["token_embd.weight"]?.bits == 4)  // embedding slot: 4
        #expect(byName["output.weight"]?.bits == 4)  // with the embedding
        #expect(byName["blk.0.ffn_gate_shexp.weight"]?.bits == 4)  // sharedExpert slot
        #expect(byName["blk.0.ffn_gate_inp.weight"]?.bits == 8)  // router slot
        #expect(byName["blk.0.attn_qkv.weight"]?.bits == 8)  // attention slot
        #expect(byName["blk.3.attn_q.weight"]?.bits == 8)
        // Format constraints override the slot.
        #expect(byName["blk.0.hc_attn_norm.weight"]?.bits == 16)  // 1-D norm
        #expect(byName["blk.0.ssm_a"]?.bits == 16)  // 48 wide
        #expect(byName["blk.0.hc_attn_up.weight"]?.bits == 16)  // rank-320 pair

        // No affine tensor may be ungrouped.
        for tensor in plan.resident where tensor.bits != 16 {
            #expect(tensor.columns % 64 == 0, "\(tensor.name) at \(tensor.bits) bits")
        }

        print(
            "GGUF conversion plan: resident \(plan.residentBytes) bytes "
                + "(4-bit \(plan.bytesByBits[4] ?? 0), 8-bit \(plan.bytesByBits[8] ?? 0), "
                + "bf16 \(plan.bytesByBits[16] ?? 0)), experts \(plan.expertBytes), "
                + "total \(plan.totalBytes) bytes "
                + "(\(String(format: "%.1f", Double(plan.totalBytes) / 1e9)) GB)")
        // The PLE table is the format fallback that decides the install size:
        // 160-wide rows cannot be group-64 affine, and the int4 shader hardcodes
        // group 64, so the policy stores it bf16.
        let ple = try #require(
            plan.formatFallbacks.first { $0.name == "per_layer_token_embd.weight" },
            "the PLE table should be the format fallback")
        #expect(ple.columns == 160)
        #expect(ple.role == "pleTable")
        #expect(ple.group32Bytes < ple.bf16Bytes / 2)
        for fallback in plan.formatFallbacks {
            print(
                "GGUF format fallback: \(fallback.name) (\(fallback.role), "
                    + "\(fallback.columns) wide) intended \(fallback.intendedBits)-bit; "
                    + "bf16 \(fallback.bf16Bytes) vs group-32 \(fallback.group32Bytes)")
        }
        // With the fallback in place the install is 175.6 GB, not the ~100 GB
        // the probe projected; the difference is that one table.
        #expect(plan.totalBytes > 170_000_000_000)
        #expect(plan.totalBytes < 180_000_000_000)
    }
}
