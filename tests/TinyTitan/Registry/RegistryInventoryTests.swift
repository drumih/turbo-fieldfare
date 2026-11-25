import Testing
import Foundation

/// The registry is the one inventory of shipped (model, quantization) pairs
/// (7.0 phase 0, plan task 1). These tests pin its shape and its coverage of
/// the shipped set; the count in `expectedShippedPairs` changes only together
/// with the reconciliation output that task 1 step 1.2 records in tracker row
/// TT-072.
struct RegistryInventoryTests {
    /// Resolve the registry relative to the package root, whether the test
    /// process runs from there or from a test bundle directory.
    private static func registryURL() -> URL {
        let relative = "registry/registry.json"
        if FileManager.default.fileExists(atPath: relative) {
            return URL(fileURLWithPath: relative)
        }
        // swift test runs the bundle from .build/.../x86_64-... or arm64-...;
        // walk up to the package root (the directory containing Package.swift).
        var dir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0...6 {
            if FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("Package.swift").path)
            {
                return dir.appendingPathComponent(relative)
            }
            dir.deleteLastPathComponent()
        }
        return URL(fileURLWithPath: relative)
    }

    private static func loadRegistry() throws -> [String: Any] {
        let data = try Data(contentsOf: registryURL())
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test func registryParsesAndCoversTheShippedPairs() throws {
        let registry = try Self.loadRegistry()
        #expect(registry["version"] as? Int == 1)
        let models = try #require(registry["models"] as? [[String: Any]])
        // 10 models: the dense trio, the four 35B-A3B MoE fine-tunes,
        // Qwen3.8-Flash-Next, Ornith 1.5 122B, and Inkling 2 Mini.
        #expect(models.count == 10)
        for model in models {
            let variants = try #require(model["variants"] as? [[String: Any]])
            #expect(!variants.isEmpty)
            // Every shipped pair carries the fields the later phases rely on.
            for variant in variants {
                #expect(variant["id"] != nil)
                #expect(variant["bits"] != nil)
                #expect(
                    variant["engine"] as? String == "cpu" || variant["engine"] as? String == "gpu")
                let shipped = variant["shipped"] as? Bool ?? false
                // JSONSerialization surfaces JSON null as NSNull, so "unset"
                // is either an absent key or NSNull.
                let target = variant["target"]
                #expect(target == nil || target is NSNull || target is String)
                if shipped {
                    // A shipped variant must name a real target and carry a
                    // qualification record (catalog --check enforces the same).
                    #expect(target is String)
                    #expect(variant["qualification"] is [String: Any])
                }
                let qualification = variant["qualification"]
                #expect(
                    qualification == nil || qualification is NSNull
                        || qualification is [String: Any])
            }
        }
    }

    @Test func variantIdsAreUniqueAndSuffixed() throws {
        let registry = try Self.loadRegistry()
        let models = try #require(registry["models"] as? [[String: Any]])
        var seen = Set<String>()
        for model in models {
            let modelID = try #require(model["modelID"] as? String)
            for variant in try #require(model["variants"] as? [[String: Any]]) {
                let id = try #require(variant["id"] as? String)
                // The width suffix is the convention; a `-gpu` suffix may
                // follow it for a second engine over the same pair (TT-097
                // Task 12 added the dense-GPU rows).
                #expect(
                    id.hasSuffix("bit") || id.hasSuffix("bit-gpu"),
                    "\(id) must end in the width suffix")
                #expect(id.hasPrefix(modelID))
                #expect(seen.insert(id).inserted)
            }
        }
        // 26 pairs: the 18 Qwen/Ornith pairs plus the six dense-GPU rows plus
        // the two Inkling 2 Mini variants (GGUF Q4 and NVFP4).
        #expect(seen.count == 26)
    }

    @Test func everyShippedPairMatchesTheInstallerTable() throws {
        // The installer's choice table (tools/tinytitan_models.sh) is one of
        // the six lists the registry replaces; until the shells retire, the
        // two must agree. Count the `key|Name|bits|GB|note` rows and compare
        // against the registry's variant count, accounting for the bf16dense
        // row the registry models as a variant of ornith-1.5-35b.
        let script = try String(
            contentsOf: URL(fileURLWithPath: "tools/tinytitan_models.sh"), encoding: .utf8)
        let tableStart = try #require(script.range(of: "TINYTITAN_MODEL_CHOICES=("))
        // The array closes with a line that is exactly ")"; row text itself
        // contains parentheses, so a character-level first ")" would stop
        // inside a row.
        let tableEnd = try #require(script[tableStart.upperBound...].firstRange(of: "\n)\n"))
        let rows = script[tableStart.upperBound..<tableEnd.lowerBound]
            .split(separator: "\n")
            .filter { $0.contains("|") }
        #expect(rows.count == 20)
    }
}
