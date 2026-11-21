import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2TextModelTests {
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func generatesFromAPromptWithCaches() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let config = try Inkling2Config.load(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        let tokenizer = try Inkling2Tokenizer(directory: directory)
        let model = try Inkling2TextModel(
            directory: directory, config: config, store: store)
        let prompt =
            "<|begin_of_text|><|message_system|><|content_text|>"
            + "Thinking effort level: 0.9<|end_message|><|message_user|>"
            + "The capital of France is<|end_message|><|message_model|>"
            + "<|content_text|>"
        let promptIds = try tokenizer.encode(prompt)
        var caches = model.makeCaches()
        for id in promptIds.dropLast() {
            _ = try model.forward(token: id, caches: &caches)
        }
        var next = promptIds.last ?? 0
        var generated: [Int] = []
        for _ in 0..<4 {
            let logits = try model.forward(token: next, caches: &caches)
            next = Inkling2TextModel.greedyToken(logits)
            generated.append(next)
        }
        print("prompt=\(promptIds) generated=\(generated) text=\(tokenizer.decode(generated))")
        #expect(generated.count == 4)
        #expect(generated.allSatisfy { (0..<200_058).contains($0) })
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func producesOneTokenOfLogits() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let config = try Inkling2Config.load(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        let initStart = Date()
        let model = try Inkling2TextModel(
            directory: directory, config: config, store: store)
        var caches = model.makeCaches()
        let forwardStart = Date()
        var trace: [[Float]] = []
        let logits = try model.forward(token: 200_028, caches: &caches) {
            _, hidden in
            trace.append(hidden)
        }
        for (index, hidden) in trace.enumerated() {
            print("trace \(index) \(hidden.prefix(5))")
        }
        let next = Inkling2TextModel.greedyToken(logits)
        let logits2 = try model.forward(token: next, caches: &caches)
        let forwardEnd = Date()
        print(
            "init=\(forwardStart.timeIntervalSince(initStart)) "
                + "forward=\(forwardEnd.timeIntervalSince(forwardStart))")
        #expect(logits.count == 200_058)
        #expect(logits.allSatisfy { $0.isFinite })
        let next2 = Inkling2TextModel.greedyToken(logits2)
        let tokenizer = try Inkling2Tokenizer(directory: directory)
        print(
            "next=\(next) text=\(tokenizer.decode([next])) "
                + "next2=\(next2) text2=\(tokenizer.decode([next2]))")
        #expect(logits2.count == 200_058)
        #expect(logits2.allSatisfy { $0.isFinite })
    }

    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"] != nil))
    func buildsTheFullLoop() throws {
        let path = try #require(
            ProcessInfo.processInfo.environment["TINYTITAN_INKLING2_MODEL"])
        let directory = URL(fileURLWithPath: path)
        let config = try Inkling2Config.load(directory: directory)
        let store = try Inkling2ExpertStore(directory: directory)
        let model = try Inkling2TextModel(
            directory: directory, config: config, store: store)
        #expect(model.config.text.layerCount == 24)
        #expect(
            model.visitIndices
                == Array(0...5) + Array(6...17) + Array(6...17) + Array(6...17)
                + Array(18...23))
    }
}
