import SwiftUI
import UniformTypeIdentifiers
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation

/// View showing attached documents and allowing more to be added
struct DocumentAttachmentView: View {
    @Bindable var model: AppModel
    @State private var isFileImporterPresented = false
    @State private var isHovering = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header with add button
            HStack {
                Label("Documents", systemImage: "doc.on.doc")
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.secondary)
                
                Spacer()
                
                if !model.attachments.isEmpty {
                    Text("\(model.attachments.count) file\(model.attachments.count > 1 ? "s" : "")")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                
                Button {
                    isFileImporterPresented = true
                } label: {
                    Label("Add", systemImage: "plus.circle.fill")
                        .labelStyle(.iconOnly)
                        .font(.callout)
                }
                .buttonStyle(.borderless)
                .help("Add a document (PDF, Word, TXT, MD, RTF)")
                .disabled(model.isExtractingAttachment)
            }
            
            // Loading indicator
            if model.isExtractingAttachment {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.7)
                    Text("Extracting text…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            
            // List of attached documents
            if !model.attachments.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(model.attachments) { attachment in
                        AttachmentRow(
                            attachment: attachment,
                            onRemove: { model.removeAttachment(attachment) }
                        )
                    }
                }
                
                // Footer with total size
                HStack {
                    Text("Total size: \(formattedTotalSize)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                    
                    Spacer()
                    
                    Button("Clear all") {
                        model.clearAttachments()
                    }
                    .font(.caption2)
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                }
            }
            
            // Error messages
            if !model.attachmentErrors.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Label("Errors", systemImage: "exclamationmark.triangle.fill")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.orange)
                        Spacer()
                        Button("Clear errors") {
                            model.clearAttachmentErrors()
                        }
                        .font(.caption2)
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                    }
                    ForEach(model.attachmentErrors.indices, id: \.self) { index in
                        HStack(alignment: .top, spacing: 6) {
                            Text(model.attachmentErrors[index])
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                            Spacer(minLength: 4)
                            Button {
                                model.dismissAttachmentError(at: index)
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.caption)
                                    .symbolRenderingMode(.hierarchical)
                            }
                            .buttonStyle(.borderless)
                            .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            }
        }
        .fileImporter(
            isPresented: $isFileImporterPresented,
            allowedContentTypes: DocumentExtractor.supportedTypes,
            allowsMultipleSelection: true
        ) { result in
            switch result {
            case .success(let urls):
                model.attachDocuments(from: urls)
            case .failure(let error):
                model.attachmentErrors = [error.localizedDescription]
            }
        }
        .onDrop(of: DocumentExtractor.supportedTypes, isTargeted: $isHovering) { providers in
            handleDrop(providers: providers)
        }
        .overlay {
            if isHovering {
                RoundedRectangle(cornerRadius: 12)
                    .stroke(TurboFieldfareMacTheme.accentColor, lineWidth: 2)
                    .background(TurboFieldfareMacTheme.accentColor.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
            }
        }
    }
    
    private var formattedTotalSize: String {
        ByteCountFormatter.string(fromByteCount: Int64(model.totalAttachmentBytes), countStyle: .file)
    }
    
    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        Task {
            var urls: [URL] = []
            for provider in providers {
                if let url = await withCheckedContinuation({ continuation in
                    _ = provider.loadObject(ofClass: URL.self) { url, _ in
                        continuation.resume(returning: url)
                    }
                }) {
                    urls.append(url)
                }
            }
            if !urls.isEmpty {
                await MainActor.run {
                    model.attachDocuments(from: urls)
                }
            }
        }
        return true
    }
}

/// Row displaying an attached document
struct AttachmentRow: View {
    let attachment: DocumentAttachment
    let onRemove: () -> Void
    @State private var isHovering = false
    @State private var showingPreview = false
    
    var body: some View {
        HStack(spacing: 10) {
            // Document type icon on a tinted tile
            Image(systemName: attachment.type.icon)
                .font(.title3)
                .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                .frame(width: 32, height: 32)
                .background(
                    TurboFieldfareMacTheme.accentColor.opacity(0.12),
                    in: RoundedRectangle(cornerRadius: 8)
                )
                .accessibilityHidden(true)
            
            // File information
            VStack(alignment: .leading, spacing: 2) {
                Text(attachment.filename)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                
                HStack(spacing: 6) {
                    Text(attachment.formattedSize)
                    if let pages = attachment.pageCount {
                        Text("•")
                        Text("\(pages) page\(pages > 1 ? "s" : "")")
                    }
                    if attachment.truncated {
                        Text("•")
                        Text("truncated")
                            .foregroundStyle(.orange)
                    }
                    Text("•")
                    Text("\(attachment.extractedText.count) characters")
                }
                .font(.caption2)
                .foregroundStyle(.tertiary)
            }
            
            Spacer()
            
            // Action buttons
            HStack(spacing: 4) {
                Button {
                    showingPreview.toggle()
                } label: {
                    Image(systemName: "eye")
                        .font(.caption)
                }
                .buttonStyle(.borderless)
                .help("Preview text")
                
                Button {
                    onRemove()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.caption)
                        .symbolRenderingMode(.hierarchical)
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.secondary)
                .help("Remove this document")
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background {
            RoundedRectangle(cornerRadius: 12)
                .fill(TurboFieldfareAppearance.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(isHovering
                                ? AnyShapeStyle(TurboFieldfareMacTheme.accentColor.opacity(0.4))
                                : AnyShapeStyle(.separator.opacity(0.35)),
                                lineWidth: 1)
                }
        }
        .onHover { hovering in
            isHovering = hovering
        }
        .popover(isPresented: $showingPreview, arrowEdge: .bottom) {
            AttachmentPreview(attachment: attachment)
        }
    }
}

/// Preview of the text extracted from a document
struct AttachmentPreview: View {
    let attachment: DocumentAttachment
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: attachment.type.icon)
                    .font(.title2)
                VStack(alignment: .leading) {
                    Text(attachment.filename)
                        .font(.headline)
                    Text(attachment.type.displayName)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            
            Divider()
            
            ScrollView {
                Text(attachment.preview(maxLength: 2000))
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 200, maxHeight: 300)
            
            if attachment.truncated {
                Label("Document truncated to \(DocumentExtractor.maximumTextLength / 1000)k characters", systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .padding()
        .frame(width: 400)
    }
}

#Preview {
    DocumentAttachmentView(model: AppModel())
        .frame(width: 500)
        .padding()
}