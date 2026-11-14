import Foundation
import Testing
import TinyTitanFormat
import TinyTitanCore

@testable import TinyTitanRepackCore

/// A MoE install's routed-expert width has to agree with its expert bytes, and
/// `--verify-install` has to refuse one that does not.
///
/// `packed_experts/layout.json` is the only place a packed-expert install records
/// how its expert bytes are packed. `validateQuantAgainstResident` deliberately
/// steps aside for those tensors -- they are the GPU path's, not the CPU reader's
/// -- which left the `routedExpert` slot compared against *nothing* for exactly the
/// shapes the product ships: the 35B-A3B and 125B-A6B builds. A manifest that
/// declared 8-bit over 4-bit expert bytes passed verification, and the writer's own
/// comment names what happens next: the dequantizer's word count and the Metal
/// pipeline constant both come from that declaration, so the model loads, answers
/// fluently, and is wrong.
///
/// The byte extent of a packed tensor determines its width exactly
/// (`size = elements * bits / 8`), so the payload is the authority and the
/// description is what gets checked. These tests hold that arithmetic against the
/// table of a real install on disk, and then against a real repack of a synthetic
/// one, so both halves are pinned: the check refuses a lie, and the writer does not
/// trip it.
@Suite struct PackedExpertWidthTests {

    // MARK: - The arithmetic, on a real expert table

    /// The 4-bit expert table of `qwen3.8-flash-next_125B_A6B_4Bit` verifies
    /// against its own declared width: 3 x 819,200 weight bytes plus
    /// 6 x 51,200 scale/bias bytes is 2,764,800, which pads to the recorded
    /// 2,768,896-byte stride (169 x 16,384).
    @Test func theRealExpertTablePassesAtItsOwnWidth() throws {
        try validate(
            tensors: qwen38Expert(weightBytes: 819_200, annotatedBits: 4),
            stride: 2_768_896,
            declaredWeightBits: 4)
    }

    /// The bug this task exists for: the same bytes declared 8-bit.
    ///
    /// Every other check still passes -- the shapes are well formed, the strides
    /// divide evenly, the hashes match the files -- and the install reads its
    /// experts with the wrong word count.
    @Test func aDeclaredWidthThatDisagreesWithTheExpertBytesIsRefused() throws {
        let message = refusal(
            tensors: qwen38Expert(weightBytes: 819_200, annotatedBits: 4),
            stride: 2_768_896,
            declaredWeightBits: 8)
        #expect(message.contains("are 4-bit"))
        #expect(message.contains("declares a routed-expert width of 8"))
    }

    /// The 8-bit analogue, so the check is not a 4-bit special case:
    /// 3 x 1,638,400 + 307,200 = 5,222,400, padding to 319 x 16,384.
    @Test func theEightBitTablePassesAgainstAnEightBitDeclaration() throws {
        try validate(
            tensors: qwen38Expert(weightBytes: 1_638_400, annotatedBits: 8),
            stride: 5_226_496,
            declaredWeightBits: 8)
        let message = refusal(
            tensors: qwen38Expert(weightBytes: 1_638_400, annotatedBits: 8),
            stride: 5_226_496,
            declaredWeightBits: 4)
        #expect(message.contains("are 8-bit"))
    }

    /// A tensor's own `bits` annotation is a second claim about the same bytes, and
    /// it is checked against them too: an expert can be internally inconsistent
    /// before it is inconsistent with the manifest.
    @Test func anAnnotationThatDisagreesWithItsOwnBytesIsRefused() throws {
        let message = refusal(
            tensors: qwen38Expert(weightBytes: 819_200, annotatedBits: 8),
            stride: 2_768_896,
            declaredWeightBits: 8)
        #expect(message.contains("annotated 8-bit"))
        #expect(message.contains("are 4-bit"))
    }

    /// 6-bit was withdrawn as a *format*, which is not the same as a 6-bit install
    /// being a lie. The bound here is the format layer's own (1...32), not the
    /// 4-or-8 the resident check insists on, so a truthful withdrawn install still
    /// verifies and only a mismatched one is refused. Being stricter than the load
    /// path would tell the user to re-download weights that are fine:
    /// 3 x 1,228,800 + 307,200 = 3,993,600, padding to 244 x 16,384.
    @Test func aTruthfulWithdrawnWidthIsNotRefusedButAMislabelledOneIs() throws {
        try validate(
            tensors: qwen38Expert(weightBytes: 1_228_800, annotatedBits: 6),
            stride: 3_997_696,
            declaredWeightBits: 6)
        let message = refusal(
            tensors: qwen38Expert(weightBytes: 1_228_800, annotatedBits: 6),
            stride: 3_997_696,
            declaredWeightBits: 4)
        #expect(message.contains("are 6-bit"))
    }

    /// An unannotated weight is judged on its bytes alone: 1,638,400 elements in
    /// 819,200 bytes is 4-bit, annotation or no annotation.
    @Test func aWidthDerivedFromBytesNeedsNoAnnotation() throws {
        try validate(
            tensors: qwen38Expert(weightBytes: 819_200, annotatedBits: nil),
            stride: 2_768_896,
            declaredWeightBits: 4)
    }

    /// A missing slice is the failure the stride sum is for: the file still divides
    /// evenly by the page and every shape is still legal.
    @Test func aSliceMissingFromAnExpertIsRefused() throws {
        var tensors = qwen38Expert(weightBytes: 819_200, annotatedBits: 4)
        tensors.removeValue(forKey: "down_biases")
        // 2,764,800 - 51,200 = 2,713,600, which pads to 166 x 16,384 = 2,719,744.
        let message = refusal(
            tensors: tensors, stride: 2_768_896, declaredWeightBits: 4)
        #expect(message.contains("its tensors total 2713600 bytes"))
        #expect(message.contains("which pads to 2719744"))
        #expect(message.contains("a slice is missing"))
    }

    /// The same sum catches a slice written twice, which a per-tensor range check
    /// cannot see because every range still fits inside the file.
    @Test func aSliceDuplicatedInAnExpertIsRefused() throws {
        let gate = try #require(qwen38Expert(weightBytes: 819_200, annotatedBits: 4)["gate"])
        var tensors = qwen38Expert(weightBytes: 819_200, annotatedBits: 4)
        tensors["gate_copy"] = gate
        let message = refusal(
            tensors: tensors, stride: 2_768_896, declaredWeightBits: 4)
        #expect(message.contains("duplicated"))
    }

    /// Padding is allowed at the end of an expert and nowhere else, and it is not
    /// required: a blob that fills its stride exactly is a whole number of pages.
    /// Two 256x64 weights at 4-bit are 8,192 bytes each, which is one page.
    @Test func anExpertThatFillsItsStrideExactlyPasses() throws {
        let weight = SSDAISubTensorV1(
            offset: 0, size: 8_192, dtype: "U32", shape: [256, 64], bits: 4)
        try validate(
            tensors: ["gate": weight, "up": weight],
            stride: 16_384,
            declaredWeightBits: 4)
    }

    /// A byte extent that is not a whole number of packed values per element is a
    /// broken layout, not a width to guess at.
    @Test func aByteExtentThatDoesNotDivideIsRefused() throws {
        var tensors = qwen38Expert(weightBytes: 819_200, annotatedBits: 4)
        tensors["gate"] = SSDAISubTensorV1(
            offset: 0, size: 819_201, dtype: "U32", shape: [640, 2560], bits: 4)
        let message = refusal(
            tensors: tensors, stride: 2_768_896, declaredWeightBits: 4)
        #expect(message.contains("does not divide"))
    }

    /// A zero-sized dimension describes no elements, so it implies no width.
    @Test func aTensorWithNoElementsIsRefused() throws {
        var tensors = qwen38Expert(weightBytes: 819_200, annotatedBits: 4)
        tensors["up"] = SSDAISubTensorV1(
            offset: 0, size: 819_200, dtype: "U32", shape: [640, 0], bits: 4)
        let message = refusal(
            tensors: tensors, stride: 2_768_896, declaredWeightBits: 4)
        #expect(message.contains("has no elements"))
    }

    /// A derived width has to be one a packed tensor can hold. 32-bit is the
    /// largest legal claim and it passes; 33 is the smallest one past the format
    /// layer's own bound, reached here by arithmetic rather than by a constant this
    /// check invents: 1,638,400 x 33 / 8 = 6,758,400 bytes.
    @Test func anImpliedWidthBeyondTheFormatRangeIsRefused() throws {
        var beyond = qwen38Expert(weightBytes: 819_200, annotatedBits: 4)
        beyond["gate"] = SSDAISubTensorV1(
            offset: 0, size: 6_758_400, dtype: "U32", shape: [640, 2560], bits: nil)
        let message = refusal(
            tensors: beyond, stride: 2_768_896, declaredWeightBits: 4)
        #expect(message.contains("outside the 1...32 bits"))

        var atTheBound = qwen38Expert(weightBytes: 819_200, annotatedBits: 4)
        // 1,638,400 x 32 / 8 = 6,553,600 bytes, which is a legal width even
        // though the install then fails for declaring 4.
        atTheBound["gate"] = SSDAISubTensorV1(
            offset: 0, size: 6_553_600, dtype: "U32", shape: [640, 2560], bits: 32)
        let widthMessage = refusal(
            tensors: atTheBound, stride: 2_768_896, declaredWeightBits: 4)
        #expect(widthMessage.contains("are 32-bit"))
    }

    /// A bf16 slice is two bytes an element and has no width to agree with, but its
    /// byte extent is still a claim about its shape.
    @Test func aScaleSliceOfTheWrongSizeIsRefused() throws {
        var tensors = qwen38Expert(weightBytes: 819_200, annotatedBits: 4)
        tensors["gate_scales"] = SSDAISubTensorV1(
            offset: 0, size: 51_201, dtype: "BF16", shape: [640, 40], bits: nil)
        let message = refusal(
            tensors: tensors, stride: 2_768_896, declaredWeightBits: 4)
        #expect(message.contains("is not the 51200 bytes a bf16 tensor of that shape is"))
    }

    /// The verifier decodes `layout.json` through its own mirror, so a dtype the
    /// format layer has never heard of reaches this code and must not be counted as
    /// if it were one of the two it knows.
    @Test func anUnknownDTypeIsRefused() throws {
        var tensors = qwen38Expert(weightBytes: 819_200, annotatedBits: 4)
        tensors["gate"] = SSDAISubTensorV1(
            offset: 0, size: 819_200, dtype: "F16", shape: [640, 2560], bits: nil)
        let message = refusal(
            tensors: tensors, stride: 2_768_896, declaredWeightBits: 4)
        #expect(message.contains("unknown dtype F16"))
    }

    /// The two overflow guards, each reached with the largest table that can be
    /// written: a byte extent that cannot be turned into a bit count, and a shape
    /// whose element count cannot be turned into a bf16 byte count. Neither may wrap
    /// into a width that looks plausible, and neither may be reported as one.
    @Test func arithmeticThatWouldOverflowIsRefusedRatherThanWrapped() throws {
        var hugeBits = qwen38Expert(weightBytes: 819_200, annotatedBits: 4)
        hugeBits["gate"] = SSDAISubTensorV1(
            offset: 0, size: UInt64.max, dtype: "U32", shape: [1], bits: nil)
        let bitMessage = refusal(
            tensors: hugeBits, stride: 2_768_896, declaredWeightBits: 4)
        #expect(bitMessage.contains("overflow a bit count"))

        var hugeElements = qwen38Expert(weightBytes: 819_200, annotatedBits: 4)
        hugeElements["gate_scales"] = SSDAISubTensorV1(
            offset: 0, size: 51_200, dtype: "BF16",
            shape: [UInt32.max, UInt32.max], bits: nil)
        let byteMessage = refusal(
            tensors: hugeElements, stride: 2_768_896, declaredWeightBits: 4)
        #expect(byteMessage.contains("too many bf16 values to count in bytes"))
    }

    /// Two machines must print the same complaint: the tensors are a dictionary, and
    /// the check walks them in key order rather than iteration order. `down` sorts
    /// first, so it is the tensor named when every weight in the table lies.
    @Test func theComplaintDoesNotDependOnDictionaryOrder() throws {
        for _ in 0..<10 {
            let message = refusal(
                tensors: qwen38Expert(weightBytes: 819_200, annotatedBits: 8),
                stride: 2_768_896,
                declaredWeightBits: 8)
            #expect(
                message.contains("expert 0 tensor down:"),
                Comment(rawValue: "the named tensor was not the first in key order: \(message)")
            )
        }
    }

    // MARK: - The whole layout check

    /// A dense install has no routed experts, so it has no routed-expert width to
    /// declare, and a missing `quant` block must not make it fail.
    ///
    /// This is the sibling risk of the check above: the dense Qwen 3.5 installs
    /// carry per-layer records with no experts in them. The empty-layer shortcut runs
    /// before the width is asked for, and this pins that ordering -- if the guard
    /// moved above it, every dense install would start failing `--verify-install`.
    @Test func aDenseLayoutHasNoWidthToDeclare() throws {
        let root = temporaryRoot("packed-width-dense")
        defer { try? FileManager.default.removeItem(atPath: root) }
        try writeLayout(
            [
                "expertStride": 0,
                "numLayers": 1,
                "expertsPerLayer": 0,
                "layers": [["layer": 0, "file": "layer_00.bin", "experts": []]],
            ],
            atRoot: root)
        try VerifiedInstallTool.validatePackedExpertLayout(
            access: try SSDAIDirectoryAccess(rootPath: root),
            manifest: Manifest(
                files: ["packed_experts/layout.json": fileEntry()],
                expertsPerLayer: 0,
                numLayers: 1,
                expertStride: 0,
                sourceSnapshotHash: nil,
                quant: nil))
    }

    /// An install that *does* carry expert bytes cannot escape the check by omitting
    /// the block the check reads.
    @Test func anExpertBearingLayoutWithoutAQuantBlockIsRefused() throws {
        let root = temporaryRoot("packed-width-no-quant")
        defer { try? FileManager.default.removeItem(atPath: root) }
        // One expert of two 4-bit weights that fill a 16,384-byte stride exactly.
        func slice(_ offset: Int) -> [String: Any] {
            ["offset": offset, "size": 8_192, "dtype": "U32", "shape": [256, 64], "bits": 4]
        }
        try writeLayout(
            [
                "expertStride": 16_384,
                "numLayers": 1,
                "expertsPerLayer": 1,
                "layers": [
                    [
                        "layer": 0,
                        "file": "layer_00.bin",
                        "experts": [
                            [
                                "expert": 0, "offset": 0, "size": 16_384,
                                "tensors": ["gate": slice(0), "up": slice(8_192)],
                            ]
                        ],
                    ]
                ],
            ],
            atRoot: root)
        try Data(repeating: 0, count: 16_384).write(
            to: URL(
                fileURLWithPath: (root as NSString)
                    .appendingPathComponent("packed_experts/layer_00.bin")))
        let access = try SSDAIDirectoryAccess(rootPath: root)
        let message = refusalMessage {
            try VerifiedInstallTool.validatePackedExpertLayout(
                access: access,
                manifest: Manifest(
                    files: [
                        "packed_experts/layout.json": fileEntry(),
                        "packed_experts/layer_00.bin": fileEntry(bytes: 16_384),
                    ],
                    expertsPerLayer: 1,
                    numLayers: 1,
                    expertStride: 16_384,
                    sourceSnapshotHash: nil,
                    quant: nil))
        }
        #expect(message.contains("has no quant block"))
    }

    /// The load path's routed cross-check names one expert per layer
    /// (`Model+SchemaValidation.validateRoutedExpertLayout` reads
    /// `layer.experts.first`), so a width lie confined to any later expert is one
    /// that can load and be answered from. The verifier's loop is what covers the
    /// rest of them, and this pins that it walks every expert rather than stopping
    /// at the first: a 125B layer is 512 experts, and only the second one lies.
    @Test func everyExpertInTheLayerIsCheckedNotOnlyTheFirst() throws {
        let root = temporaryRoot("packed-width-all-experts")
        defer { try? FileManager.default.removeItem(atPath: root) }
        try checkLayer(root: root, secondExpertBits: 4)
        let message = refusalMessage {
            try checkLayer(root: root, secondExpertBits: 8)
        }
        #expect(message.contains("expert 1 tensor gate: annotated 8-bit"))
    }

    // MARK: - Writer and verifier together

    /// A freshly repacked 8-bit MoE install verifies, which is the proof that the
    /// check above does not refuse the writer's own output at the other width.
    @Test func aFreshEightBitRepackVerifies() async throws {
        let output = try await repack(weightBits: 8, tag: "packed-width-e2e-8bit")
        let result = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputSSDAI: output))
        #expect(result.fileCount > 0)
    }

    /// The headline, end to end: edit the declared routed-expert width, leave the
    /// payload alone, and `--verify-install` refuses it by name.
    @Test func aManifestThatMisdeclaresTheRoutedExpertWidthIsRefused() async throws {
        let output = try await repack(weightBits: 4, tag: "packed-width-e2e-lie")
        var manifest = try readJSONObject(at: output, named: "manifest.json")
        var quant = try #require(manifest["quant"] as? [String: Any])
        var routed = try #require(quant["routedExpert"] as? [String: Any])
        // The bytes are 4-bit, so 4 is the truthful declaration and 8 is the lie.
        #expect(try #require(routed["weightBits"] as? Int) == 4)
        routed["weightBits"] = 8
        quant["routedExpert"] = routed
        manifest["quant"] = quant
        try writeJSONObject(manifest, at: output, named: "manifest.json")

        let message = refusalMessage {
            _ = try VerifiedInstallTool.run(options: VerifyInstallOptions(inputSSDAI: output))
        }
        #expect(message.contains("routed-expert width of 8"))
    }

    /// The expert annotation is the other place the width is claimed, and this says
    /// which check answers first: `layout.json` is listed in the manifest, so were
    /// the width check not run before the file hashes, the message here would be a
    /// plain SHA mismatch and a lying description would be reported as a corrupted
    /// file.
    @Test func aLayoutThatMisannotatesAnExpertIsRefusedBeforeTheHashes() async throws {
        let output = try await repack(weightBits: 4, tag: "packed-width-e2e-annotation")
        var layout = try readJSONObject(at: output, named: "packed_experts/layout.json")
        var layers = try #require(layout["layers"] as? [[String: Any]])
        var experts = try #require(layers[0]["experts"] as? [[String: Any]])
        var expert = experts[0]
        var tensors = try #require(expert["tensors"] as? [String: Any])
        var gate = try #require(tensors["gate"] as? [String: Any])
        gate["bits"] = 8
        tensors["gate"] = gate
        expert["tensors"] = tensors
        experts[0] = expert
        layers[0]["experts"] = experts
        layout["layers"] = layers
        try writeJSONObject(layout, at: output, named: "packed_experts/layout.json")

        let message = refusalMessage {
            _ = try VerifiedInstallTool.run(options: VerifyInstallOptions(inputSSDAI: output))
        }
        #expect(message.contains("annotated 8-bit"))
        #expect(!message.contains("SHA mismatch"))
    }

    // MARK: - Fixtures

    /// One packed expert of the 125B-A6B install on disk, as dumped from
    /// `models/qwen3.8-flash-next_125B_A6B_4Bit/packed_experts/layout.json`: three
    /// U32 weights of 1,638,400 elements each (gate and up at 640x2560, down at
    /// 2560x640) and six BF16 scale/bias tensors of 25,600 elements each.
    /// `weightBytes` is the one number a width changes. The real file's per-slice
    /// offsets are dropped here because this arithmetic never reads them.
    private func qwen38Expert(
        weightBytes: UInt64,
        annotatedBits: Int?
    ) -> [String: SSDAISubTensorV1] {
        func weight(_ shape: [UInt32]) -> SSDAISubTensorV1 {
            SSDAISubTensorV1(
                offset: 0, size: weightBytes, dtype: "U32", shape: shape, bits: annotatedBits)
        }
        func meta(_ shape: [UInt32]) -> SSDAISubTensorV1 {
            SSDAISubTensorV1(
                offset: 0, size: UInt64(shape[0]) * UInt64(shape[1]) * 2,
                dtype: "BF16", shape: shape, bits: nil)
        }
        return [
            "gate": weight([640, 2560]),
            "gate_scales": meta([640, 40]),
            "gate_biases": meta([640, 40]),
            "up": weight([640, 2560]),
            "up_scales": meta([640, 40]),
            "up_biases": meta([640, 40]),
            "down": weight([2560, 640]),
            "down_scales": meta([2560, 10]),
            "down_biases": meta([2560, 10]),
        ]
    }

    /// The expert table above, validated at `declaredWeightBits` against `stride`.
    private func validate(
        tensors: [String: SSDAISubTensorV1],
        stride: UInt64,
        declaredWeightBits: Int
    ) throws {
        try PackedExpertBytes.validate(
            expert(tensors, stride: stride),
            stride: stride,
            declaredWeightBits: declaredWeightBits,
            in: "packed_experts/layer_00.bin",
            rank: 0)
    }

    /// The same call, expected to refuse, returning its message.
    private func refusal(
        tensors: [String: SSDAISubTensorV1],
        stride: UInt64,
        declaredWeightBits: Int
    ) -> String {
        let packed = expert(tensors, stride: stride)
        return refusalMessage {
            try PackedExpertBytes.validate(
                packed,
                stride: stride,
                declaredWeightBits: declaredWeightBits,
                in: "packed_experts/layer_00.bin",
                rank: 0)
        }
    }

    private func refusalMessage(_ body: () throws -> Void) -> String {
        do {
            try body()
        } catch let error as RepackError {
            return error.description
        } catch {
            Issue.record("threw something other than a RepackError: \(error)")
            return ""
        }
        Issue.record("expected a refusal, and the check passed")
        return ""
    }

    /// An `Expert` holding `tensors` and sized to `stride`.
    ///
    /// The offsets stay as the fixture wrote them (zero) because this arithmetic
    /// reads only sizes, dtypes and shapes: slice placement inside the expert is the
    /// format layer's overlap check, and `SSDAIPackedExpertsLayoutV1` is what owns
    /// it.
    private func expert(
        _ tensors: [String: SSDAISubTensorV1],
        stride: UInt64
    ) -> Expert {
        Expert(expert: 0, offset: 0, size: stride, tensors: tensors)
    }

    /// A manifest entry, for the fields `validatePackedExpertLayout` reads.
    private func fileEntry(bytes: UInt64 = 0) -> TinyTitanRepackCore.ManifestFileEntry {
        TinyTitanRepackCore.ManifestFileEntry(
            size: bytes, sha256: String(repeating: "0", count: 64))
    }

    private func writeLayout(_ object: [String: Any], atRoot root: String) throws {
        let directory = (root as NSString).appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(
            atPath: directory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            .write(
                to: URL(
                    fileURLWithPath: (directory as NSString)
                        .appendingPathComponent("layout.json")))
    }

    /// Write a one-layer, two-expert layout into `root` and verify it, with expert
    /// 1's gate annotated at `secondExpertBits` over bytes that are 4-bit. Each
    /// expert is two 8,192-byte weights that fill its 16,384-byte stride exactly,
    /// and the layer file is the two experts end to end.
    private func checkLayer(root: String, secondExpertBits: Int) throws {
        func weights(_ bits: Int) -> [String: Any] {
            [
                "gate": [
                    "offset": 0, "size": 8_192, "dtype": "U32", "shape": [256, 64], "bits": bits,
                ],
                "up": [
                    "offset": 8_192, "size": 8_192, "dtype": "U32", "shape": [256, 64], "bits": 4,
                ],
            ]
        }
        try writeLayout(
            [
                "expertStride": 16_384,
                "numLayers": 1,
                "expertsPerLayer": 2,
                "layers": [
                    [
                        "layer": 0,
                        "file": "layer_00.bin",
                        "experts": [
                            ["expert": 0, "offset": 0, "size": 16_384, "tensors": weights(4)],
                            [
                                "expert": 1, "offset": 16_384, "size": 16_384,
                                "tensors": weights(secondExpertBits),
                            ],
                        ],
                    ]
                ],
            ],
            atRoot: root)
        try Data(repeating: 0, count: 32_768).write(
            to: URL(
                fileURLWithPath: (root as NSString)
                    .appendingPathComponent("packed_experts/layer_00.bin")))
        try VerifiedInstallTool.validatePackedExpertLayout(
            access: try SSDAIDirectoryAccess(rootPath: root),
            manifest: Manifest(
                files: [
                    "packed_experts/layout.json": fileEntry(),
                    "packed_experts/layer_00.bin": fileEntry(bytes: 32_768),
                ],
                expertsPerLayer: 2,
                numLayers: 1,
                expertStride: 16_384,
                sourceSnapshotHash: nil,
                quant: quant(routedExpert: 4)))
    }

    /// The five width slots, with the routed-expert one under test.
    private func quant(routedExpert: Int) -> SSDAIManifestQuantV1 {
        func slot(_ bits: Int) -> SSDAIManifestQuantSlotV1 {
            SSDAIManifestQuantSlotV1(
                weightBits: bits, scheme: "affine",
                scaleType: "BF16", biasType: "BF16", groupSize: 64)
        }
        return SSDAIManifestQuantV1(
            embedding: slot(8),
            attention: slot(4),
            router: slot(8),
            sharedExpert: slot(8),
            routedExpert: slot(routedExpert),
            overrides: nil)
    }

    /// Repack the synthetic MoE snapshot at `weightBits` and hand back the install.
    private func repack(weightBits: Int, tag: String) async throws -> String {
        let root = temporaryRoot(tag)
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.ssdai")
        _ = try SyntheticSnapshot.buildQwen(at: snapshot, weightBits: weightBits)
        // A local import copies these three files beside the config without parsing
        // them, and the synthetic builder writes no tokenizer.
        let stub = Data(#"{"version":"1.0"}"#.utf8)
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            try stub.write(
                to: URL(fileURLWithPath: (snapshot as NSString).appendingPathComponent(name)))
        }
        _ = try await RemoteStreamingRepacker.runLocalSnapshot(
            options: LocalSnapshotRepackOptions(
                inputSnapshotDir: snapshot,
                outputDir: output,
                modelID: "synthetic-qwen-\(weightBits)bit",
                minFreeReserveBytes: 0))
        return output
    }

    private func readJSONObject(at directory: String, named file: String) throws -> [String: Any] {
        let data = try Data(
            contentsOf: URL(
                fileURLWithPath: (directory as NSString).appendingPathComponent(file)))
        return try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func writeJSONObject(
        _ object: [String: Any],
        at directory: String,
        named file: String
    ) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            .write(
                to: URL(
                    fileURLWithPath: (directory as NSString).appendingPathComponent(file)))
    }

    private func temporaryRoot(_ tag: String) -> String {
        let base = (FileManager.default.currentDirectoryPath as NSString)
            .appendingPathComponent(".build/test-artifacts")
        let path = (base as NSString).appendingPathComponent(tag)
        try? FileManager.default.removeItem(atPath: path)
        try? FileManager.default.createDirectory(
            atPath: path, withIntermediateDirectories: true)
        return path
    }
}
