import AppKit
import ImageIO
import SwiftUI

/// Decodes only a bounded thumbnail for conversation UI; the original image
/// remains on disk for the decode service.
struct LocalImageThumbnailView: View {
    private struct SendableCGImage: @unchecked Sendable {
        let value: CGImage
    }

    let fileURL: URL?
    var maximumPixelSize: Int = 640

    @State private var thumbnail: NSImage?

    var body: some View {
        Group {
            if let thumbnail {
                Image(nsImage: thumbnail)
                    .resizable()
                    .scaledToFit()
            } else {
                ZStack {
                    Color.secondary.opacity(0.08)
                    Image(systemName: "photo")
                        .font(.title2)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .task(id: fileURL) {
            thumbnail = nil
            let loaded = await loadThumbnail()
            guard !Task.isCancelled else { return }
            thumbnail = loaded
        }
    }

    private func loadThumbnail() async -> NSImage? {
        guard let fileURL else { return nil }
        let maximumPixelSize = max(64, maximumPixelSize)
        let result = await Task.detached(priority: .utility) {
            let sourceOptions = [
                kCGImageSourceShouldCache: false,
            ] as CFDictionary
            guard let source = CGImageSourceCreateWithURL(
                fileURL as CFURL,
                sourceOptions) else { return nil as SendableCGImage? }
            let options = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceShouldCacheImmediately: true,
                kCGImageSourceThumbnailMaxPixelSize: maximumPixelSize,
            ] as CFDictionary
            guard let image = CGImageSourceCreateThumbnailAtIndex(
                source,
                0,
                options) else { return nil as SendableCGImage? }
            return SendableCGImage(value: image)
        }.value
        guard let result else { return nil }
        return NSImage(
            cgImage: result.value,
            size: NSSize(
                width: result.value.width,
                height: result.value.height))
    }
}
