import AppKit
import SwiftUI

public enum TransientPopoverPresentation {
    public static let behavior: NSPopover.Behavior = .transient
}

@MainActor
public struct TransientPopoverButton<Content: View>: NSViewRepresentable {
    private let systemImage: String
    private let help: String
    private let content: Content

    public init(systemImage: String, help: String,
                @ViewBuilder content: () -> Content) {
        self.systemImage = systemImage
        self.help = help
        self.content = content()
    }

    public func makeCoordinator() -> Coordinator { Coordinator(content: content) }

    public func makeNSView(context: Context) -> NSButton {
        let button = NSButton(image: NSImage(
            systemSymbolName: systemImage, accessibilityDescription: help) ?? NSImage(),
            target: context.coordinator, action: #selector(Coordinator.toggle(_:)))
        button.isBordered = false
        button.toolTip = help
        button.setAccessibilityLabel(help)
        return button
    }

    public func updateNSView(_ button: NSButton, context: Context) {
        context.coordinator.host.rootView = content
    }

    public static func dismantleNSView(_ button: NSButton, coordinator: Coordinator) {
        coordinator.popover.close()
    }

    @MainActor
    public final class Coordinator: NSObject {
        let popover = NSPopover()
        let host: NSHostingController<Content>

        init(content: Content) {
            host = NSHostingController(rootView: content)
            super.init()
            popover.behavior = TransientPopoverPresentation.behavior
            popover.contentViewController = host
        }

        @objc func toggle(_ sender: NSButton) {
            if popover.isShown {
                popover.close()
            } else {
                popover.show(relativeTo: sender.bounds, of: sender, preferredEdge: .maxY)
            }
        }
    }
}
