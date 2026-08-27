import AppKit
import TurboFieldfare
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI
import UniformTypeIdentifiers

struct PromptComposerView: View {
    @Bindable var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false
    @State private var showingImagePicker = false
    @State private var isImageDropTargeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.imageAttachments.isEmpty || model.imageAttachmentError != nil {
                attachmentStrip
            }
            editor
            footer
        }
        .fileImporter(
            isPresented: $showingImagePicker,
            allowedContentTypes: VisionImageLimits().allowedContentTypes,
            allowsMultipleSelection: true
        ) { result in
            switch result {
            case .success(let urls): model.addImages(urls)
            case .failure(let error): model.reportImageAttachmentError(error)
            }
        }
        .padding(14)
        .background {
            RoundedRectangle(cornerRadius: 22)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 22)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
    }

    private var editor: some View {
        PromptTextEditor(
            text: $model.promptText,
            isFocused: Binding(
                get: { promptFocused },
                set: { promptFocused = $0 }),
            newlineShortcut: model.newlineShortcut,
            canRun: model.canRun,
            canAcceptImages: model.isImageInputAvailable
                && !model.isRunning
                && !model.isAddingImages,
            onSubmit: model.run,
            onImagesDropped: { model.addImages($0) },
            onImageDataPasted: model.addImageData,
            onPromisedImagesReceived: { urls, directory in
                model.addImages(urls, discardingSourceDirectory: directory)
            },
            onPromisedImagesFailed: {
                model.reportImageAttachmentError(
                    "That drag did not deliver a file TurboFieldfare could read.")
            },
            onUnsupportedImagePaste: {
                model.reportImageAttachmentError(
                    "That clipboard content is not an image TurboFieldfare can read.")
            },
            onDropTargeted: { isImageDropTargeted = $0 })
            .accessibilityLabel("Message")
            .frame(height: editorHeight)
            .overlay(alignment: .topLeading) {
                if model.promptText.isEmpty {
                    // Matches the NSTextView text origin: 5pt line fragment
                    // padding, no vertical inset.
                    Text("Message")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }
            .overlay {
                if isImageDropTargeted {
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.tint, lineWidth: 2)
                        .allowsHitTesting(false)
                }
            }
    }

    private var editorHeight: CGFloat {
        model.shouldShowPromptExamples ? 46 : 84
    }

    private var footer: some View {
        HStack(spacing: 10) {
            if model.isImageInputAvailable {
                Button {
                    showingImagePicker = true
                } label: {
                    Label("Add images", systemImage: "photo.badge.plus")
                        .labelStyle(.iconOnly)
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.borderless)
                .disabled(model.isRunning || model.isAddingImages
                    || model.imageAttachments.count
                        >= model.maximumImageAttachments)
                .help(model.maximumImageAttachments == 0
                    ? "Start a new chat to make room for images."
                    : "Add images")
            }
            promptTips
            Spacer()
            clearAction
            GenerateControl(model: model)
        }
    }

    private var attachmentStrip: some View {
        VStack(alignment: .leading, spacing: 6) {
            ScrollView(.horizontal) {
                // Top-aligned so the tiles share one edge; they are all the
                // same size now, but alignment left to chance is how the row
                // drifted out of line in the first place.
                HStack(alignment: .top, spacing: 8) {
                    ForEach(model.imageAttachments, id: \.id) { attachment in
                        SubmittedImageThumbnail(attachment: attachment)
                            .overlay(alignment: .topTrailing) {
                            Button {
                                model.removeImage(id: attachment.id)
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                            }
                            .buttonStyle(.plain)
                            .background(.regularMaterial, in: Circle())
                            .offset(x: 5, y: -5)
                            .accessibilityLabel("Remove \(attachment.displayName)")
                        }
                        .padding(.top, 5)
                        .padding(.trailing, 5)
                    }
                }
            }
            .frame(height: 58)
            if let error = model.imageAttachmentError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
    }

    private var promptTips: some View {
        Button {
            showingPromptTips.toggle()
        } label: {
            Label("Prompt tips", systemImage: "questionmark.circle")
                .labelStyle(.iconOnly)
                .frame(width: 28, height: 28)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help("Prompt tips")
        .popover(isPresented: $showingPromptTips,
                 attachmentAnchor: .point(.top),
                 arrowEdge: .top) {
            promptGuide
        }
    }

    private var promptGuide: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prompting this model")
                .font(.headline)

            tipSection("Ask for a clear task",
                       "Say what you want the model to create, explain, plan, or transform. Put the essential context in the same prompt.")
            tipSection("Shape the answer",
                       "Specify a useful length, sections, tone, or output format. Concrete constraints work better than a long list of vague preferences.")
            tipSection("Anchor important facts",
                       "Include facts the answer must preserve and say what should be checked. Generated factual claims can still be wrong or outdated.")
            tipSection("For code and calculations",
                       "Provide types, dimensions, interfaces, edge cases, or a small scaffold. Compile or run the result before relying on it.")
            tipSection("Try a focused revision",
                       "If the answer drifts, shorten the task and make the missing requirement explicit. The default temperature is 0.20 for steadier responses.")
        }
        .font(.callout)
        .frame(width: 390, alignment: .leading)
        .padding(18)
    }

    private func tipSection(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .fontWeight(.semibold)
            Text(detail)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var clearAction: some View {
        if !model.isRunning
            && (!model.promptText.isEmpty || !model.imageAttachments.isEmpty) {
            Button {
                model.promptText = ""
                model.clearImages()
                promptFocused = true
            } label: {
                Label("Clear input", systemImage: "xmark.circle.fill")
                    .labelStyle(.iconOnly)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Clear text and images")
        } else if !model.isRunning && model.hasOutputTranscript {
            Button {
                model.newChat()
            } label: {
                Label("New chat", systemImage: "trash")
                    .labelStyle(.iconOnly)
                    .frame(width: 28, height: 28)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .help("Start a new chat")
        }
    }
}

private struct PromptTextEditor: NSViewRepresentable {
    @Binding var text: String
    let isFocused: Binding<Bool>
    let newlineShortcut: AppNewlineShortcut
    let canRun: Bool
    let canAcceptImages: Bool
    let onSubmit: () -> Void
    let onImagesDropped: ([URL]) -> Void
    let onImageDataPasted: (Data, String) -> Void
    let onPromisedImagesReceived: ([URL], URL) -> Void
    let onPromisedImagesFailed: () -> Void
    let onUnsupportedImagePaste: () -> Void
    let onDropTargeted: (Bool) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let textView = ImageDropTextView()
        textView.delegate = context.coordinator
        textView.string = text
        textView.font = .systemFont(ofSize: NSFont.systemFontSize)
        textView.textColor = .labelColor
        textView.drawsBackground = false
        textView.isRichText = false
        textView.allowsUndo = true
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainerInset = .zero
        textView.textContainer?.lineFragmentPadding = 5
        textView.textContainer?.widthTracksTextView = true
        textView.setAccessibilityLabel("Message")
        // A promise-only drag — Photos, Mail, most browsers — never reaches
        // `draggingEntered` unless its types are registered here.
        textView.registerForDraggedTypes(
            textView.registeredDraggedTypes
                + NSFilePromiseReceiver.readableDraggedTypes.map {
                    NSPasteboard.PasteboardType(rawValue: $0)
                })

        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.borderType = .noBorder
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.documentView = textView
        context.coordinator.textView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? ImageDropTextView else { return }
        context.coordinator.parent = self
        textView.canAcceptImages = canAcceptImages
        textView.onImagesDropped = onImagesDropped
        textView.onImageDataPasted = onImageDataPasted
        textView.onPromisedImagesReceived = onPromisedImagesReceived
        textView.onPromisedImagesFailed = onPromisedImagesFailed
        textView.onUnsupportedImagePaste = onUnsupportedImagePaste
        textView.onDropTargeted = onDropTargeted
        if textView.string != text {
            textView.string = text
        }
        if isFocused.wrappedValue,
           textView.window?.firstResponder !== textView {
            DispatchQueue.main.async {
                textView.window?.makeFirstResponder(textView)
            }
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: PromptTextEditor
        weak var textView: NSTextView?

        init(_ parent: PromptTextEditor) {
            self.parent = parent
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            parent.text = textView.string
        }

        func textDidBeginEditing(_ notification: Notification) {
            parent.isFocused.wrappedValue = true
        }

        func textDidEndEditing(_ notification: Notification) {
            parent.isFocused.wrappedValue = false
        }

        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            guard commandSelector == #selector(NSResponder.insertNewline(_:)) else {
                return false
            }
            let event = NSApp.currentEvent
            switch PromptSubmissionPolicy.decision(
                newlineShortcut: parent.newlineShortcut,
                modifiers: Self.modifiers(event?.modifierFlags ?? []),
                canRun: parent.canRun,
                hasMarkedText: textView.hasMarkedText(),
                isRepeat: event?.isARepeat == true) {
            case .submit:
                parent.onSubmit()
                return true
            case .consume:
                return true
            case .deferToEditor:
                return false
            }
        }

        private static func modifiers(_ flags: NSEvent.ModifierFlags) -> EventModifiers {
            var modifiers: EventModifiers = []
            if flags.contains(.command) { modifiers.insert(.command) }
            if flags.contains(.shift) { modifiers.insert(.shift) }
            if flags.contains(.option) { modifiers.insert(.option) }
            if flags.contains(.control) { modifiers.insert(.control) }
            return modifiers
        }
    }
}

private final class ImageDropTextView: NSTextView {
    var canAcceptImages = false
    var onImagesDropped: (([URL]) -> Void)?
    /// Promised files arrive in a directory we made and must not keep.
    var onPromisedImagesReceived: (([URL], URL) -> Void)?
    var onImageDataPasted: ((Data, String) -> Void)?
    var onUnsupportedImagePaste: (() -> Void)?
    var onPromisedImagesFailed: (() -> Void)?
    var onDropTargeted: ((Bool) -> Void)?

    override func paste(_ sender: Any?) {
        guard canAcceptImages else {
            super.paste(sender)
            return
        }
        let pasteboard = NSPasteboard.general
        let urls = fileURLs(from: pasteboard)
        if !urls.isEmpty {
            onImagesDropped?(urls)
            return
        }
        // Text on the pasteboard wins over a picture that came with it.
        // Copying a range of cells from Numbers or Excel, or a text box from
        // Keynote, writes TIFF alongside the RTF and plain text; so does a
        // rich selection from many other apps. Taking the picture and
        // returning swallowed the text the user actually selected, in a text
        // editor, with nothing said about it. A screenshot or an image copied
        // from an image editor carries no text, so it still attaches.
        //
        // Only the byte-carrying branches need this. A file paste is handled
        // above, and Finder puts the path on the pasteboard as a string too.
        if pasteboard.string(forType: .string)?.isEmpty == false {
            super.paste(sender)
            return
        }
        // An image copied from another app arrives as bytes with no file
        // behind it, which used to be refused with an instruction to go and
        // find one.
        if let image = imageData(from: pasteboard) {
            onImageDataPasted?(image.data, image.name)
            return
        }
        if containsImage(in: pasteboard) {
            onUnsupportedImagePaste?()
            return
        }
        super.paste(sender)
    }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        guard carriesImages(sender.draggingPasteboard) else {
            return super.draggingEntered(sender)
        }
        guard canAcceptImages else { return [] }
        onDropTargeted?(true)
        return .copy
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        guard carriesImages(sender.draggingPasteboard) else {
            return super.draggingUpdated(sender)
        }
        return canAcceptImages ? .copy : []
    }

    override func draggingExited(_ sender: NSDraggingInfo?) {
        onDropTargeted?(false)
        super.draggingExited(sender)
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        let urls = fileURLs(from: sender.draggingPasteboard)
        if !urls.isEmpty {
            onDropTargeted?(false)
            guard canAcceptImages else { return false }
            onImagesDropped?(urls)
            return true
        }
        let promises = filePromises(from: sender.draggingPasteboard)
        guard !promises.isEmpty else { return super.performDragOperation(sender) }
        onDropTargeted?(false)
        guard canAcceptImages else { return false }
        receive(promises)
        return true
    }

    /// Drags from Photos, Mail and most browsers carry a promise rather than a
    /// file: the source writes the bytes only once a destination asks for them.
    /// Without this the drop was simply refused.
    private func receive(_ promises: [NSFilePromiseReceiver]) {
        let destination = FileManager.default.temporaryDirectory
            .appendingPathComponent("TurboFieldfare-Promises", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        do {
            try FileManager.default.createDirectory(
                at: destination, withIntermediateDirectories: true)
        } catch {
            // Failing to make a staging directory is a disk or permission
            // failure, not a statement about what was dropped. Reporting it as
            // unsupported content sent someone off converting file formats
            // because the volume was full.
            onPromisedImagesFailed?()
            return
        }
        let queue = OperationQueue()
        // One completion arrives per promised FILE, not per receiver: a legacy
        // promiser puts several file names on a single pasteboard item. Counting
        // receivers finished the batch at the first file and handed the
        // directory straight to `addImages`, whose `defer` deleted it while
        // AppKit was still writing the rest — so a three-image drag attached one
        // and destroyed two, with nothing reporting the loss.
        let expected = promises.reduce(0) { $0 + max(1, $1.fileNames.count) }
        let received = ReceivedPromises(expected: expected)
        for promise in promises {
            promise.receivePromisedFiles(
                atDestination: destination,
                options: [:],
                operationQueue: queue
            ) { [weak self] url, error in
                // The staged copy is what the request uses, so the promised
                // file is only needed until then.
                let finished = received.record(url: error == nil ? url : nil)
                guard let finished else { return }
                DispatchQueue.main.async {
                    if finished.isEmpty {
                        // A promise that failed is not "unsupported content";
                        // saying so sends the user looking for the wrong thing.
                        self?.onPromisedImagesFailed?()
                        try? FileManager.default.removeItem(at: destination)
                    } else if let handler = self?.onPromisedImagesReceived {
                        handler(finished, destination)
                    } else {
                        // The text view went away before the promises resolved,
                        // so nothing downstream will ever own this directory:
                        // the optional chain was a no-op and the sweep covers
                        // only the staging root, so a closed window left the
                        // full-size copies behind for every future launch.
                        try? FileManager.default.removeItem(at: destination)
                    }
                }
            }
        }
    }

    /// Only files in a format the vision pack can actually decode. Without the
    /// conformance filter any file URL was staged: copying a document in Finder
    /// and pressing Cmd-V attached it, replacing the editor's own paste, and the
    /// failure surfaced only once the run reached image decoding.
    ///
    /// The list is the runtime's, not `public.image`: conforming to
    /// `public.image` admits camera RAW, EXR and WebP, which are copied at full
    /// size and decoded for a thumbnail before the run refuses them. The picker
    /// reads the same list, and paste, drop and the drag-accept highlight all
    /// read through here, so the four stay in agreement.
    private func fileURLs(from pasteboard: NSPasteboard) -> [URL] {
        let objects = pasteboard.readObjects(
            forClasses: [NSURL.self],
            options: [
                .urlReadingFileURLsOnly: true,
                .urlReadingContentsConformToTypes:
                    Array(VisionImageLimits().allowedTypeIdentifiers),
            ]) as? [NSURL]
        return objects?.map { $0 as URL } ?? []
    }

    private func filePromises(from pasteboard: NSPasteboard) -> [NSFilePromiseReceiver] {
        pasteboard.readObjects(
            forClasses: [NSFilePromiseReceiver.self],
            options: nil) as? [NSFilePromiseReceiver] ?? []
    }

    private func carriesImages(_ pasteboard: NSPasteboard) -> Bool {
        !fileURLs(from: pasteboard).isEmpty || !filePromises(from: pasteboard).isEmpty
    }

    /// The first pasteboard representation the vision pack can actually decode.
    /// PNG and JPEG come through unchanged; anything else is re-encoded as PNG
    /// rather than handing the decoder a format it was never validated against.
    private func imageData(from pasteboard: NSPasteboard) -> (data: Data, name: String)? {
        if let data = pasteboard.data(forType: .png) {
            return (data, "Pasted image.png")
        }
        if let data = pasteboard.data(forType: NSPasteboard.PasteboardType("public.jpeg")) {
            return (data, "Pasted image.jpeg")
        }
        guard let tiff = pasteboard.data(forType: .tiff),
              let representation = NSBitmapImageRep(data: tiff),
              let png = representation.representation(using: .png, properties: [:])
        else { return nil }
        return (png, "Pasted image.png")
    }

    private func containsImage(in pasteboard: NSPasteboard) -> Bool {
        (pasteboard.types ?? []).contains { type in
            UTType(type.rawValue)?.conforms(to: .image) == true
        }
    }
}

/// Collects the results of a multi-file promise drop, which arrive one
/// completion at a time on an arbitrary queue.
private final class ReceivedPromises: @unchecked Sendable {
    private let lock = NSLock()
    private let expected: Int
    private var urls: [URL] = []
    private var completed = 0

    init(expected: Int) {
        self.expected = expected
    }

    /// Returns the collected URLs once every promise has reported, and nil
    /// until then.
    func record(url: URL?) -> [URL]? {
        lock.lock()
        defer { lock.unlock() }
        if let url { urls.append(url) }
        completed += 1
        return completed == expected ? urls : nil
    }
}
