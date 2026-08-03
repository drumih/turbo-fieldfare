import CryptoKit
import Darwin
import Foundation
import ImageIO

enum AppChatAttachmentStoreError: Error, CustomStringConvertible, Sendable {
    case sourceMustBeLocal
    case sourceUnavailable
    case sourceIsNotRegular
    case emptyFile
    case fileTooLarge(maximumBytes: UInt64)
    case unsupportedFormat
    case invalidDimensions
    case imageTooLarge(maximumPixels: UInt64)
    case invalidStoredReference
    case storedFileMissing
    case storedFileChanged
    case copyFailed(String)

    var description: String {
        switch self {
        case .sourceMustBeLocal:
            return "Choose an image stored on this Mac."
        case .sourceUnavailable:
            return "The selected image could not be opened."
        case .sourceIsNotRegular:
            return "The selected item is not a regular image file."
        case .emptyFile:
            return "The selected image is empty."
        case .fileTooLarge(let maximumBytes):
            return "Choose an image smaller than \(maximumBytes / 1_048_576) MB."
        case .unsupportedFormat:
            return "Choose a PNG, JPEG, HEIC, or WebP image."
        case .invalidDimensions:
            return "The selected image has invalid dimensions."
        case .imageTooLarge(let maximumPixels):
            return "Choose an image smaller than \(maximumPixels / 1_000_000) megapixels."
        case .invalidStoredReference:
            return "The saved image reference is invalid."
        case .storedFileMissing:
            return "The attached image is no longer available."
        case .storedFileChanged:
            return "The attached image changed after it was added."
        case .copyFailed(let detail):
            return "The image could not be saved: \(detail)"
        }
    }
}

/// Owns bounded import and resolution of images referenced by chat history.
/// Files are copied beside the history document and never embedded in JSON or
/// decode-service frames.
enum AppChatAttachmentStore {
    static let maximumFileBytes: UInt64 = 25 * 1_024 * 1_024
    static let maximumPixelCount: UInt64 = 64_000_000
    static let copyChunkBytes = 1 * 1_024 * 1_024

    private static let directoryName = "mac-app-chat-attachments"
    private static let supportedTypes: [String: String] = [
        "public.png": "png",
        "public.jpeg": "jpg",
        "public.heic": "heic",
        "public.heif": "heic",
        "org.webmproject.webp": "webp",
    ]

    static func rootURL(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(directoryName, isDirectory: true)
    }

    static func importImage(
        from sourceURL: URL,
        chatID: UUID,
        forModelDirectory modelDirectory: URL,
        fileManager: FileManager = .default
    ) throws -> AppImageAttachment {
        guard sourceURL.isFileURL else {
            throw AppChatAttachmentStoreError.sourceMustBeLocal
        }

        let sourcePath = sourceURL.standardizedFileURL.path
        let sourceDescriptor = open(sourcePath, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard sourceDescriptor >= 0 else {
            throw AppChatAttachmentStoreError.sourceUnavailable
        }
        defer { close(sourceDescriptor) }

        var sourceStat = stat()
        guard fstat(sourceDescriptor, &sourceStat) == 0 else {
            throw AppChatAttachmentStoreError.sourceUnavailable
        }
        guard (sourceStat.st_mode & S_IFMT) == S_IFREG else {
            throw AppChatAttachmentStoreError.sourceIsNotRegular
        }
        guard sourceStat.st_size > 0 else {
            throw AppChatAttachmentStoreError.emptyFile
        }
        let sourceBytes = UInt64(sourceStat.st_size)
        guard sourceBytes <= maximumFileBytes else {
            throw AppChatAttachmentStoreError.fileTooLarge(
                maximumBytes: maximumFileBytes)
        }

        let attachmentID = UUID()
        let chatComponent = chatID.uuidString.lowercased()
        let attachmentComponent = attachmentID.uuidString.lowercased()
        let chatDirectory = rootURL(forModelDirectory: modelDirectory)
            .appendingPathComponent(chatComponent, isDirectory: true)
        try fileManager.createDirectory(
            at: chatDirectory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        let partialURL = chatDirectory.appendingPathComponent(
            ".\(attachmentComponent).partial",
            isDirectory: false)
        var importCompleted = false
        defer {
            try? fileManager.removeItem(at: partialURL)
            if !importCompleted,
               (try? fileManager.contentsOfDirectory(
                    atPath: chatDirectory.path).isEmpty) == true {
                try? fileManager.removeItem(at: chatDirectory)
            }
        }

        let digest: String
        do {
            digest = try copyAndHash(
                from: sourceDescriptor,
                to: partialURL,
                expectedBytes: sourceBytes)
        } catch let error as AppChatAttachmentStoreError {
            throw error
        } catch {
            throw AppChatAttachmentStoreError.copyFailed("\(error)")
        }

        let metadata = try inspectImage(at: partialURL)
        let destinationURL = chatDirectory.appendingPathComponent(
            "\(attachmentComponent).\(metadata.extensionName)",
            isDirectory: false)
        do {
            try fileManager.moveItem(at: partialURL, to: destinationURL)
        } catch {
            throw AppChatAttachmentStoreError.copyFailed("\(error)")
        }

        let relativePath = "\(chatComponent)/\(destinationURL.lastPathComponent)"
        importCompleted = true
        return AppImageAttachment(
            id: attachmentID,
            relativePath: relativePath,
            originalFilename: String(sourceURL.lastPathComponent.prefix(255)),
            mediaTypeIdentifier: metadata.typeIdentifier,
            pixelWidth: metadata.width,
            pixelHeight: metadata.height,
            byteCount: sourceBytes,
            sha256: digest)
    }

    static func fileURL(
        for attachment: AppImageAttachment,
        modelDirectory: URL
    ) throws -> URL {
        guard attachment.hasValidStoredMetadata else {
            throw AppChatAttachmentStoreError.invalidStoredReference
        }
        let components = attachment.relativePath.split(separator: "/")
        guard components.count == 2,
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw AppChatAttachmentStoreError.invalidStoredReference
        }
        let root = rootURL(forModelDirectory: modelDirectory).standardizedFileURL
        let candidate = root
            .appendingPathComponent(attachment.relativePath, isDirectory: false)
            .standardizedFileURL
        let rootPrefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard candidate.path.hasPrefix(rootPrefix) else {
            throw AppChatAttachmentStoreError.invalidStoredReference
        }
        return candidate
    }

    static func validateStoredImage(
        _ attachment: AppImageAttachment,
        forModelDirectory modelDirectory: URL,
        fileManager: FileManager = .default
    ) throws {
        let fileURL = try fileURL(
            for: attachment,
            modelDirectory: modelDirectory)
        var isDirectory = ObjCBool(false)
        guard fileManager.fileExists(
            atPath: fileURL.path,
            isDirectory: &isDirectory),
              !isDirectory.boolValue else {
            throw AppChatAttachmentStoreError.storedFileMissing
        }
        let values = try fileURL.resourceValues(forKeys: [
            .isRegularFileKey,
            .isSymbolicLinkKey,
            .fileSizeKey,
        ])
        guard values.isRegularFile == true,
              values.isSymbolicLink != true else {
            throw AppChatAttachmentStoreError.invalidStoredReference
        }
        guard let fileSize = values.fileSize,
              fileSize >= 0,
              UInt64(fileSize) == attachment.byteCount else {
            throw AppChatAttachmentStoreError.storedFileChanged
        }
    }

    static func remove(
        _ attachment: AppImageAttachment,
        forModelDirectory modelDirectory: URL,
        fileManager: FileManager = .default
    ) {
        guard let fileURL = try? fileURL(
            for: attachment,
            modelDirectory: modelDirectory) else { return }
        try? fileManager.removeItem(at: fileURL)
        let parent = fileURL.deletingLastPathComponent()
        if (try? fileManager.contentsOfDirectory(atPath: parent.path).isEmpty) == true {
            try? fileManager.removeItem(at: parent)
        }
    }

    static func removeAll(
        forChatID chatID: UUID,
        modelDirectory: URL,
        fileManager: FileManager = .default
    ) {
        let root = rootURL(forModelDirectory: modelDirectory)
        let directory = root.appendingPathComponent(
            chatID.uuidString.lowercased(),
            isDirectory: true)
        try? fileManager.removeItem(at: directory)
    }

    private static func copyAndHash(
        from sourceDescriptor: Int32,
        to destinationURL: URL,
        expectedBytes: UInt64
    ) throws -> String {
        let destinationDescriptor = open(
            destinationURL.path,
            O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
            S_IRUSR | S_IWUSR)
        guard destinationDescriptor >= 0 else {
            throw AppChatAttachmentStoreError.copyFailed(
                String(cString: strerror(errno)))
        }
        defer { close(destinationDescriptor) }

        var hasher = SHA256()
        var buffer = [UInt8](repeating: 0, count: copyChunkBytes)
        var copied: UInt64 = 0
        while true {
            let count = buffer.withUnsafeMutableBytes { bytes -> Int in
                read(sourceDescriptor, bytes.baseAddress, bytes.count)
            }
            if count == 0 { break }
            guard count > 0 else {
                throw AppChatAttachmentStoreError.copyFailed(
                    String(cString: strerror(errno)))
            }
            copied += UInt64(count)
            guard copied <= maximumFileBytes else {
                throw AppChatAttachmentStoreError.fileTooLarge(
                    maximumBytes: maximumFileBytes)
            }
            guard copied <= expectedBytes else {
                throw AppChatAttachmentStoreError.copyFailed(
                    "the source changed while it was being copied")
            }

            try buffer.withUnsafeBytes { bytes in
                let chunk = UnsafeRawBufferPointer(
                    start: bytes.baseAddress,
                    count: count)
                hasher.update(bufferPointer: chunk)
                var offset = 0
                while offset < count {
                    let written = write(
                        destinationDescriptor,
                        bytes.baseAddress!.advanced(by: offset),
                        count - offset)
                    guard written > 0 else {
                        throw AppChatAttachmentStoreError.copyFailed(
                            String(cString: strerror(errno)))
                    }
                    offset += written
                }
            }
        }
        guard copied == expectedBytes else {
            throw AppChatAttachmentStoreError.copyFailed(
                "the source changed while it was being copied")
        }
        guard fsync(destinationDescriptor) == 0 else {
            throw AppChatAttachmentStoreError.copyFailed(
                String(cString: strerror(errno)))
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func inspectImage(at fileURL: URL) throws
        -> (typeIdentifier: String, extensionName: String, width: Int, height: Int) {
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithURL(fileURL as CFURL, options),
              CGImageSourceGetCount(source) > 0,
              let rawType = CGImageSourceGetType(source) else {
            throw AppChatAttachmentStoreError.unsupportedFormat
        }
        let typeIdentifier = rawType as String
        guard let extensionName = supportedTypes[typeIdentifier] else {
            throw AppChatAttachmentStoreError.unsupportedFormat
        }
        guard let properties = CGImageSourceCopyPropertiesAtIndex(
            source,
            0,
            options) as? [CFString: Any],
              let widthNumber = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let heightNumber = properties[kCGImagePropertyPixelHeight] as? NSNumber else {
            throw AppChatAttachmentStoreError.invalidDimensions
        }
        let width = widthNumber.intValue
        let height = heightNumber.intValue
        guard width > 0, height > 0 else {
            throw AppChatAttachmentStoreError.invalidDimensions
        }
        let pixels = UInt64(width).multipliedReportingOverflow(by: UInt64(height))
        guard !pixels.overflow,
              pixels.partialValue <= maximumPixelCount else {
            throw AppChatAttachmentStoreError.imageTooLarge(
                maximumPixels: maximumPixelCount)
        }
        return (typeIdentifier, extensionName, width, height)
    }
}
