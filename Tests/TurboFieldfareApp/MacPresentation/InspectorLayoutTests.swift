import AppKit
import SwiftUI
import Testing
@testable import TurboFieldfareAppCore
@testable import TurboFieldfareMac

@Suite(.serialized) @MainActor struct InspectorLayoutTests {
    @Test func runtimeControlsStayInsideTheInspector() async throws {
        _ = NSApplication.shared
        let modelURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("inspector-layout-\(UUID()).gturbo")
        let model = AppModel(modelDirectory: modelURL, environment: [:])
        model.installationStatus = .complete
        model.applyLoadState(.ready(modelDirectory: modelURL, loadSeconds: 0))
        let host = NSHostingView(rootView: InspectorView(model: model)
            .frame(width: RootView.inspectorWidth))
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: RootView.inspectorWidth, height: 780),
            styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.orderFront(nil)
        defer {
            window.orderOut(nil)
            window.contentView = nil
            window.close()
        }
        let deadline = ContinuousClock.now + .seconds(3)
        while scrollViews(in: host).isEmpty, ContinuousClock.now < deadline {
            await Task.yield()
            host.layoutSubtreeIfNeeded()
        }
        let scroll = try #require(scrollViews(in: host).first)
        for policy in AppRDAdvicePolicy.allCases {
            model.runtimeOptions.rdadvisePolicy = policy
            for topK in [false, true] {
                model.topKEnabled = topK
                await Task.yield()
                host.layoutSubtreeIfNeeded()
                // The segmented picker forced this scroll view wider than its
                // fixed panel; clicking it widened it again and clipped every row.
                let frame = scroll.convert(scroll.bounds, to: host)
                #expect(frame.width <= RootView.inspectorWidth + 0.5)
                #expect(frame.minX >= -0.5)
                #expect(frame.maxX <= host.bounds.width + 0.5)
            }
        }
    }

    private func scrollViews(in view: NSView) -> [NSScrollView] {
        (view as? NSScrollView).map { [$0] } ?? view.subviews.flatMap(scrollViews)
    }
}
