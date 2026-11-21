import Foundation
import Testing

@testable import TinyTitanInkling2

@Suite struct Inkling2TextModelTests {
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
        let forwardStart = Date()
        let logits = try model.forward(token: 0)
        let forwardEnd = Date()
        print(
            "init=\(forwardStart.timeIntervalSince(initStart)) "
                + "forward=\(forwardEnd.timeIntervalSince(forwardStart))")
        #expect(logits.count == 201_024)
        #expect(logits.allSatisfy { $0.isFinite })
        let next = Inkling2TextModel.greedyToken(logits)
        let tokenizer = try Inkling2Tokenizer(directory: directory)
        print("next=\(next) text=\(tokenizer.decode([next]))")
        #expect((0..<201_024).contains(next))
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
