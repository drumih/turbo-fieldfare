import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareCLICore

/// The CLI plans and encodes images in dictionary order, which Swift derives
/// from a per-process hash seed — so the same command line did the work in a
/// different order on every run, and a failure named a different image each
/// time.
@Suite struct CLIImageOrderTests {
    @Test func imagesAreProcessedInPromptOrder() {
        let ids = (0..<6).map { _ in UUID() }
        var images: [UUID: URL] = [:]
        for (index, id) in ids.enumerated() {
            images[id] = URL(fileURLWithPath: "/tmp/image-\(index).png")
        }
        let parts: [MultimodalContentPart] = ids.map { .image(id: $0) }
        let messages = [MultimodalMessage(role: .user, content: parts)]

        // Stable across repeated calls, and equal to the prompt's own order.
        let first = orderedImageIDs(messages: messages, images: images)
        #expect(first == ids, "images were not planned in the order the prompt lists them")
        #expect(orderedImageIDs(messages: messages, images: images) == first)

        // An image the parts never mention is still processed, not dropped.
        let orphan = UUID()
        images[orphan] = URL(fileURLWithPath: "/tmp/orphan.png")
        let withOrphan = orderedImageIDs(messages: messages, images: images)
        #expect(withOrphan.count == ids.count + 1)
        #expect(withOrphan.last == orphan)
        #expect(Set(withOrphan) == Set(images.keys))
    }
}
