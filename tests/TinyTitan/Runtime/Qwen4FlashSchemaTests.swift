import Foundation
import Testing

@testable import TinyTitan

/// The Qwen 4.0 Flash tensor namespace, checked against the pinned 3.8 tensor
/// list.
///
/// The 4.0 checkpoint is not published, so the names are expected to agree with
/// the 3.8 preview today. The fixture is the only complete, real list in the
/// tree; a schema that merely compiles would prove nothing about a 174 GB
/// install. When the 4.0 index lands, point this suite at its own name list.
@Suite("Qwen4Flash tensor schema")
struct Qwen4FlashSchemaTests {
    static let names: Set<String> = {
        guard
            let url = Bundle.module.url(
                forResource: "qwen38_tensor_names",
                withExtension: "txt"),
            let text = try? String(contentsOf: url, encoding: .utf8)
        else { return [] }
        return Set(text.split(separator: "\n").map(String.init))
    }()

    private func exists(_ name: String) -> Bool { Self.names.contains(name) }

    @Test("The fixture loaded")
    func fixturePresent() {
        #expect(Self.names.count == 3164)
    }

    @Test("Model-level roles resolve")
    func modelLevelRoles() {
        let s = TensorSchema.schema(for: .qwen4flash)
        #expect(exists(s.embedding), "\(s.embedding)")
        #expect(exists(s.lmHead), "\(s.lmHead)")
        #expect(exists(s.finalNorm), "\(s.finalNorm)")
    }

    @Test("Every layer role resolves")
    func layerRoles() {
        let s = TensorSchema.schema(for: .qwen4flash)
        for l in 0..<48 {
            #expect(exists(s.router(l)))
            #expect(exists(s.sharedExpertGate(l)))
            #expect(exists(s.sharedExpertUp(l)))
            #expect(exists(s.sharedExpertDown(l)))
            #expect(exists(s.inputNorm(l)))
            #expect(exists(s.postAttnNorm(l)))
            #expect(exists(Qwen4FlashTensors.attnMixDown(l)))
            #expect(exists(Qwen4FlashTensors.attnMixUp(l)))
            #expect(exists(Qwen4FlashTensors.attnInject(l)))
            #expect(exists(Qwen4FlashTensors.mlpMixDown(l)))
            #expect(exists(Qwen4FlashTensors.mlpMixUp(l)))
            #expect(exists(Qwen4FlashTensors.mlpInject(l)))
        }
        for l in stride(from: 3, to: 48, by: 4) {
            #expect(exists(s.qProj(l)))
            #expect(exists(s.kProj(l)))
            #expect(exists(s.vProj(l)))
            #expect(exists(s.oProj(l)))
            #expect(exists(Qwen4FlashTensors.qNorm(l)))
            #expect(exists(Qwen4FlashTensors.kNorm(l)))
            #expect(exists(Qwen4FlashTensors.indexerQProj(l)))
            #expect(exists(Qwen4FlashTensors.indexerKProj(l)))
        }
        for l in 0..<48 where (l + 1) % 4 != 0 {
            #expect(exists(Qwen4FlashTensors.gdnQKV(l)))
            #expect(exists(Qwen4FlashTensors.gdnConv(l)))
            #expect(exists(Qwen4FlashTensors.gdnOut(l)))
        }
        #expect(exists(Qwen4FlashTensors.pleConv(1)))
        #expect(exists(Qwen4FlashTensors.mixerDown))
        #expect(exists(Qwen4FlashTensors.mixerUp))
    }

    @Test("The draft schema is the target's layer body")
    func draftIsTheTargetBody() {
        let target = TensorSchema.schema(for: .qwen4flash)
        let draft = TensorSchema.schema(for: .qwen4flashMTP)
        #expect(draft.embedding == target.embedding)
        #expect(draft.finalNorm == target.finalNorm)
        #expect(draft.qProj(0) == target.qProj(0))
    }
}
