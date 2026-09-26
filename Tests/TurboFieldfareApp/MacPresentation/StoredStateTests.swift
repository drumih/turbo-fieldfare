import AppKit
import Observation
import SwiftUI
import Testing
import TurboFieldfareAppCore
@testable import TurboFieldfareMacPresentation

@Suite(.serialized) @MainActor struct StoredStateTests {
    @Test func bindingsAndValueMutationRedrawTheMountedView() async throws {
        let driver = StateDriver()
        let bridge = StateBridge()
        let host = StateHost(driver: driver, bridge: bridge)
        defer { host.close() }
        try await rendered { bridge.value?.text == "initial" }
        #expect(bridge.value?.selection == nil)
        #expect(bridge.value?.flag == false)
        let binding = try #require(bridge.binding)
        binding.wrappedValue = "child edit"
        try await rendered { bridge.value?.text == "child edit" }
        bridge.mutate?()
        try await rendered { bridge.value?.items == [7] }
        #expect(bridge.value?.text == "child edit!")
        #expect(binding.wrappedValue == "child edit!")
        #expect(bridge.value?.selection == driver.selection)
        #expect(bridge.value?.flag == true)
        #expect(bridge.value?.height == 42)
    }

    @Test func stableIdentityKeepsEditsAndAnimationOriginWhenParentChanges() async throws {
        let driver = StateDriver()
        let bridge = StateBridge()
        let host = StateHost(driver: driver, bridge: bridge)
        defer { host.close() }
        try await rendered { bridge.value != nil }
        bridge.mutate?()
        try await rendered { bridge.value?.items == [7] }
        let before = try #require(bridge.value)
        driver.seed = "new parent input"
        driver.revision += 1
        try await rendered { bridge.revision == 1 }
        #expect(bridge.value == before)
    }

    @Test func changedIdentityAndReinsertionResetOnlyThatView() async throws {
        let driver = StateDriver()
        let bridge = StateBridge()
        let host = StateHost(driver: driver, bridge: bridge)
        defer { host.close() }
        try await rendered { bridge.value != nil }
        bridge.mutate?()
        try await rendered { bridge.value?.flag == true }
        let retiredBinding = try #require(bridge.binding)
        driver.seed = "replacement"
        driver.identity += 1
        try await rendered { bridge.value?.text == "replacement" }
        #expect(bridge.value?.flag == false)
        #expect(bridge.value?.selection == nil)
        #expect(bridge.value?.items.isEmpty == true)
        // A late image/copy task holding the old binding cannot edit a new identity.
        retiredBinding.wrappedValue = "late old completion"
        driver.revision += 1
        try await rendered { bridge.revision == 1 }
        #expect(bridge.value?.text == "replacement")
        driver.visible = false
        try await rendered("conditional removal") { !bridge.mounted }
        driver.seed = "reinserted"
        driver.visible = true
        try await rendered { bridge.value?.text == "reinserted" }
        #expect(bridge.value?.items.isEmpty == true)
    }

    @Test func separateViewInstancesNeverShareState() async throws {
        let first = StateBridge()
        let second = StateBridge()
        let a = StateHost(driver: StateDriver(), bridge: first)
        let b = StateHost(driver: StateDriver(), bridge: second)
        defer { a.close(); b.close() }
        try await rendered { first.value != nil && second.value != nil }
        first.mutate?()
        try await rendered { first.value?.flag == true }
        #expect(second.value?.text == "initial")
        #expect(second.value?.flag == false)
    }

    @Test func observableReferenceKeepsIdentityAndNestedChangesRedraw() async throws {
        let driver = StateDriver()
        let original = driver.model
        let bridge = StateBridge()
        let host = StateHost(driver: driver, bridge: bridge)
        defer { host.close() }
        try await rendered { bridge.model === original }
        original.count = 9
        try await rendered { bridge.modelCount == 9 }
        driver.model = StateReference()
        driver.revision += 1
        try await rendered { bridge.revision == 1 }
        #expect(bridge.model === original)
        #expect(bridge.modelCount == 9)
        driver.identity += 1
        try await rendered { bridge.model === driver.model }
        #expect(bridge.modelCount == 0)
    }

    @Test func stateReferenceIsReleasedAfterItsHostAndBindingsAreReleased() async throws {
        weak var retained: StateReference?
        do {
            let driver = StateDriver()
            retained = driver.model
            let bridge = StateBridge()
            let host = StateHost(driver: driver, bridge: bridge)
            try await rendered { bridge.model != nil }
            driver.model = StateReference()
            driver.revision += 1
            try await rendered { bridge.revision == 1 }
            host.close()
            bridge.binding = nil
            bridge.mutate = nil
            bridge.model = nil
        }
        try await rendered("reference teardown") { retained == nil }
    }

    @Test func appModelKeepsItsExplicitReferenceAndHistoryAcrossRedraws() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {
            do { try FileManager.default.removeItem(at: directory) }
            catch { Issue.record("Could not remove isolated model fixture: \(error)") }
        }
        let delegateModel = AppModel(
            modelDirectory: directory.appendingPathComponent("missing.gturbo"),
            client: NoGenerationClient(), visionRuntimeSupported: false,
            settingsPersistenceEnabled: true)
        defer { delegateModel.shutdownForTermination() }
        let driver = StateDriver()
        let bridge = AppModelBridge()
        let host = StateHost(rootView: AppModelRoot(model: delegateModel, driver: driver, bridge: bridge))
        defer { host.close() }
        try await rendered { bridge.model === delegateModel }
        let history = delegateModel.history
        delegateModel.promptText = "draft"
        try await rendered { bridge.text == "draft" }
        driver.revision += 1
        try await rendered { bridge.revision == 1 }
        #expect(bridge.model === delegateModel)
        #expect(bridge.model?.history === history)
        #expect(delegateModel.modelPathText == directory.appendingPathComponent("missing.gturbo").path)
    }

    @Test func wrapperInitializationIsEagerAndExplicitInitialValueIsRetained() {
        var constructions = 0
        func makeReference() -> StateReference {
            constructions += 1
            return StateReference()
        }
        let first = StoredState(wrappedValue: makeReference())
        let second = StoredState(wrappedValue: makeReference())
        #expect(constructions == 2)
        #expect(first.wrappedValue !== second.wrappedValue)
        let reference = StateReference()
        let explicit = StoredState(initialValue: reference)
        #expect(explicit.wrappedValue === reference)
    }

    private func rendered(_ label: String = "state update", _ predicate: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(3)
        while !predicate(), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(predicate(), "Mounted SwiftUI view did not reach the expected state: \(label)")
    }
}

@MainActor @Observable private final class StateReference {
    var count = 0
}

@MainActor @Observable private final class StateDriver {
    var seed = "initial"
    var revision = 0
    var identity = 0
    var visible = true
    var model = StateReference()
    let selection = UUID()
}

private struct StateValue: Equatable {
    var text: String
    var flag: Bool
    var selection: UUID?
    var height: CGFloat
    var items: [Int]
    var origin: Date
}

@MainActor private final class StateBridge {
    var value: StateValue?
    var binding: Binding<String>?
    var mutate: (() -> Void)?
    var model: StateReference?
    var modelCount = -1
    var revision = -1
    var mounted = false
}

@MainActor private final class StateHost {
    private let window: NSWindow

    convenience init(driver: StateDriver, bridge: StateBridge) {
        self.init(rootView: StateRoot(driver: driver, bridge: bridge))
    }

    init<Content: View>(rootView: Content) {
        _ = NSApplication.shared
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 100, height: 100),
                          styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = NSHostingView(rootView: rootView)
        window.orderFront(nil)
    }

    func close() {
        window.orderOut(nil)
        window.contentView = nil
        window.close()
    }
}

private struct StateRoot: View {
    let driver: StateDriver
    let bridge: StateBridge

    var body: some View {
        if driver.visible {
            StateFixture(seed: driver.seed, revision: driver.revision,
                         selection: driver.selection, model: driver.model, bridge: bridge)
                .id(driver.identity)
                .onAppear { bridge.mounted = true }
                .onDisappear { bridge.mounted = false }
        }
    }
}

private struct StateFixture: View {
    let revision: Int
    let selection: UUID
    let bridge: StateBridge
    @StoredState private var text: String
    @StoredState private var flag = false
    @StoredState private var selected: UUID?
    @StoredState private var height: CGFloat = 0
    @StoredState private var items: [Int] = []
    @StoredState private var origin = Date()
    @StoredState private var model: StateReference
    @StoredState private var image: NSImage?
    @FocusState private var focused: Bool

    init(seed: String, revision: Int, selection: UUID, model: StateReference, bridge: StateBridge) {
        self.revision = revision
        self.selection = selection
        self.bridge = bridge
        _text = StoredState(initialValue: seed)
        _model = StoredState(initialValue: model)
    }

    var body: some View {
        StateBindingChild(
            text: $text,
            value: StateValue(text: text, flag: flag, selection: selected,
                              height: height, items: items, origin: origin),
            model: model, count: model.count, revision: revision, bridge: bridge,
            mutate: {
                text += "!"
                flag.toggle()
                selected = selection
                height = 42
                items.append(7)
            })
    }
}

private struct StateBindingChild: View {
    @Binding var text: String
    let value: StateValue
    let model: StateReference
    let count: Int
    let revision: Int
    let bridge: StateBridge
    let mutate: () -> Void

    var body: some View {
        StateRecorder(value: value, binding: $text, model: model, count: count,
                      revision: revision, bridge: bridge, mutate: mutate)
    }
}

private struct StateRecorder: NSViewRepresentable {
    let value: StateValue
    let binding: Binding<String>
    let model: StateReference
    let count: Int
    let revision: Int
    let bridge: StateBridge
    let mutate: () -> Void

    func makeCoordinator() -> StateBridge { bridge }
    func makeNSView(context: Context) -> NSView { NSView() }
    func updateNSView(_ view: NSView, context: Context) {
        bridge.value = value
        bridge.binding = binding
        bridge.mutate = mutate
        bridge.model = model
        bridge.modelCount = count
        bridge.revision = revision
    }
}

private struct NoGenerationClient: AppInferenceClient {
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        Issue.record("The state ownership fixture must not generate")
        return AsyncThrowingStream { $0.finish() }
    }
    func cancel() {}
}

@MainActor private final class AppModelBridge {
    var model: AppModel?
    var text = ""
    var revision = -1
}

private struct AppModelRoot: View {
    let model: AppModel
    let driver: StateDriver
    let bridge: AppModelBridge

    var body: some View {
        AppModelFixture(model: model, revision: driver.revision, bridge: bridge)
    }
}

private struct AppModelFixture: View {
    @StoredState private var model: AppModel
    let revision: Int
    let bridge: AppModelBridge

    init(model: AppModel, revision: Int, bridge: AppModelBridge) {
        self.revision = revision
        self.bridge = bridge
        _model = StoredState(initialValue: model)
    }

    var body: some View {
        AppModelRecorder(model: model, text: model.promptText, revision: revision, bridge: bridge)
    }
}

private struct AppModelRecorder: NSViewRepresentable {
    let model: AppModel
    let text: String
    let revision: Int
    let bridge: AppModelBridge

    func makeNSView(context: Context) -> NSView { NSView() }
    func updateNSView(_ view: NSView, context: Context) {
        bridge.model = model
        bridge.text = text
        bridge.revision = revision
    }
}
