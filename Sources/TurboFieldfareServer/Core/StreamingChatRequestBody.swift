import Foundation
import NIOCore

struct ParsedChatRequestBody {
    let json: Data
    let stagedImages: [String: ServerStagedImage]
    let lease: ServerAttachmentLease?
}

final class StreamingChatRequestBody {
    static let maximumWireBytes = 96 * 1_024 * 1_024
    static let maximumSanitizedBytes = 5 * 1_024 * 1_024
    /// JSON nesting is decoded recursively downstream, on the event-loop thread
    /// whose stack is the 512 KiB pthread default, so deep nesting exhausts it
    /// and kills the process rather than failing the request. Foundation only
    /// rejects beyond 512, which is already past the crash threshold.
    static let maximumContainerDepth = 64

    private enum ContainerKind { case object, array }
    private struct Container {
        let kind: ContainerKind
        let contextKey: String?
        var currentKey: String?
    }

    private enum StringMode {
        /// An object key: copied through and collected so the key can be
        /// decoded when the string closes.
        case key
        /// Any other string: copied through untouched.
        case plain
        /// A messages[].content[].image_url.url value before its data-URL
        /// header's comma has decided between staging and passthrough.
        case urlPrefix
        /// The base64 payload of an image data URL, streamed to disk.
        case staging
    }

    private struct StringState {
        var mode: StringMode
        /// A backslash was seen in `key`/`plain` mode, where escapes are copied
        /// through raw; the flag only stops an escaped quote from terminating
        /// the string.
        var escaped = false
        var escape = JSONStringEscapeDecoder()
        var keyBytes: [UInt8] = []
        /// The exact wire bytes of a `urlPrefix` value, re-emitted verbatim if
        /// the value turns out not to be an image data URL.
        var rawPrefix: [UInt8] = []
        /// The same prefix with JSON escapes decoded, which is what the header
        /// match and the comma detection must run on.
        var decodedPrefix: [UInt8] = []
        var writer: StreamingBase64FileWriter?
        var replacementToken: String?
    }

    private var wireBytes = 0
    private var json = Data()
    private var containers: [Container] = []
    private var string: StringState?
    private var stagedImages: [String: ServerStagedImage] = [:]
    private var totalImageBytes = 0
    private var lease: ServerAttachmentLease?
    /// Where this request stages its images. Injectable so a test can own its
    /// staging root instead of sharing one with every other server in the
    /// process.
    private let attachmentRoot: URL
    /// Staging writes decoded image bytes to disk, so a server that cannot
    /// serve images must refuse them here — before any byte is staged — not
    /// after the request has queued behind a generation.
    private let visionCapability: String

    init(attachmentRoot: URL = ServerAttachmentDirectory.root,
         visionCapability: String = "ready") {
        self.attachmentRoot = attachmentRoot
        self.visionCapability = visionCapability
    }

    func feed(_ buffer: inout ByteBuffer) throws {
        guard let bytes = buffer.readBytes(length: buffer.readableBytes) else { return }
        wireBytes += bytes.count
        guard wireBytes <= Self.maximumWireBytes else {
            throw ServerRequestError.invalid(
                message: "request body is too large",
                param: nil, code: "request_too_large")
        }
        var index = 0
        while index < bytes.count {
            if let mode = string?.mode {
                switch mode {
                case .key, .plain:
                    index = try consumeCopiedString(bytes, from: index)
                case .urlPrefix:
                    index = try consumeURLPrefix(bytes, at: index)
                case .staging:
                    index = try consumeStagingPayload(bytes, from: index)
                }
            } else {
                index = try consumeStructural(bytes, at: index)
            }
        }
    }

    func finish() throws -> ParsedChatRequestBody {
        guard string == nil else {
            throw ServerRequestError.invalid(
                message: "unterminated JSON string",
                param: nil, code: "invalid_json")
        }
        guard json.count <= Self.maximumSanitizedBytes else {
            throw ServerRequestError.invalid(
                message: "non-image request content is too large",
                param: nil, code: "request_too_large")
        }
        return ParsedChatRequestBody(
            json: json,
            stagedImages: stagedImages,
            lease: lease)
    }

    /// Only a URL at exactly messages[].content[].image_url.url is intercepted.
    /// An image_url-shaped object anywhere else — a tool schema, a text field —
    /// is someone else's data and must pass through byte-identical.
    private var isMessageImageURLValue: Bool {
        containers.count == 6
            && containers[0].kind == .object && containers[0].contextKey == nil
            && containers[1].kind == .array && containers[1].contextKey == "messages"
            && containers[2].kind == .object
            && containers[3].kind == .array && containers[3].contextKey == "content"
            && containers[4].kind == .object
            && containers[5].kind == .object
            && containers[5].contextKey == "image_url"
            && containers[5].currentKey == "url"
    }

    private func consumeStructural(_ bytes: [UInt8], at index: Int) throws -> Int {
        let byte = bytes[index]
        if byte == UInt8(ascii: "\"") {
            try append(byte)
            if isMessageImageURLValue {
                string = StringState(mode: .urlPrefix)
            } else if containers.last?.kind == .object,
                      containers.last?.currentKey == nil {
                string = StringState(mode: .key)
            } else {
                string = StringState(mode: .plain)
            }
            return index + 1
        }
        try append(byte)
        switch byte {
        case UInt8(ascii: "{"), UInt8(ascii: "["):
            guard containers.count < Self.maximumContainerDepth else {
                throw ServerRequestError.invalid(
                    message: "request JSON nests deeper than "
                        + "\(Self.maximumContainerDepth) levels",
                    param: nil, code: "invalid_request_body")
            }
            containers.append(Container(
                kind: byte == UInt8(ascii: "{") ? .object : .array,
                contextKey: containers.last?.kind == .object
                    ? containers.last?.currentKey : containers.last?.contextKey,
                currentKey: nil))
        case UInt8(ascii: "}"), UInt8(ascii: "]"):
            if !containers.isEmpty { containers.removeLast() }
        case UInt8(ascii: ","):
            if !containers.isEmpty, containers[containers.count - 1].kind == .object {
                containers[containers.count - 1].currentKey = nil
            }
        default:
            break
        }
        return index + 1
    }

    /// A `key` or `plain` string: whole spans up to the next quote or backslash
    /// are copied at once, so the sanitized body never pays a per-byte state
    /// machine for its bulk.
    private func consumeCopiedString(_ bytes: [UInt8], from start: Int) throws -> Int {
        var state = string!
        var index = start
        if state.escaped {
            state.escaped = false
            if state.mode == .key { state.keyBytes.append(bytes[index]) }
            try append(bytes[index])
            string = state
            return index + 1
        }
        let end = Self.nextQuoteOrBackslash(bytes, from: index)
        if end > index {
            try append(contentsOf: bytes[index..<end])
            if state.mode == .key {
                state.keyBytes.append(contentsOf: bytes[index..<end])
            }
            index = end
        }
        guard index < bytes.count else {
            string = state
            return index
        }
        let byte = bytes[index]
        try append(byte)
        if byte == UInt8(ascii: "\\") {
            state.escaped = true
            if state.mode == .key { state.keyBytes.append(byte) }
            string = state
            return index + 1
        }
        if state.mode == .key, !containers.isEmpty,
           containers[containers.count - 1].kind == .object {
            let quoted = Data([UInt8(ascii: "\"")] + state.keyBytes + [UInt8(ascii: "\"")])
            containers[containers.count - 1].currentKey =
                try JSONDecoder().decode(String.self, from: quoted)
        }
        string = nil
        return index + 1
    }

    /// The head of an image URL, one byte at a time — it is at most 64 decoded
    /// bytes before it either becomes a data-URL payload or passes through.
    /// Escapes are decoded because a spec-valid serializer may write the same
    /// URL as `data:image\/png;...`; the raw bytes are kept alongside so a
    /// passthrough re-emits exactly what arrived.
    private func consumeURLPrefix(_ bytes: [UInt8], at index: Int) throws -> Int {
        var state = string!
        let byte = bytes[index]
        if state.escape.isActive {
            state.rawPrefix.append(byte)
            let decoded = try state.escape.consume(byte)
            try acceptPrefix(decoded: decoded, state: &state)
            return index + 1
        }
        if byte == UInt8(ascii: "\"") {
            try append(contentsOf: state.rawPrefix)
            try append(byte)
            string = nil
            return index + 1
        }
        if byte == UInt8(ascii: "\\") {
            state.rawPrefix.append(byte)
            state.escape.begin()
            string = state
            return index + 1
        }
        state.rawPrefix.append(byte)
        try acceptPrefix(decoded: [byte], state: &state)
        return index + 1
    }

    private func acceptPrefix(decoded: [UInt8], state: inout StringState) throws {
        for byte in decoded where byte == UInt8(ascii: ",") {
            let header = String(decoding: state.decodedPrefix, as: UTF8.self)
                .lowercased()
            if header.hasPrefix("data:image/") {
                guard visionCapability == "ready" else {
                    throw ServerRequestError.invalid(
                        message: visionCapability == "disabled"
                            ? "image support is disabled"
                            : "image support is unavailable",
                        param: "messages", code: "vision_unavailable")
                }
                guard ServerAttachmentStore.allowedDataURLHeaders.contains(header) else {
                    throw ServerRequestError.invalid(
                        message: "image_url must be a JPEG, PNG, HEIC, or HEIF base64 data URL",
                        param: "messages", code: "unsupported_image")
                }
                guard stagedImages.count < ServerAttachmentStore.maximumRequestImages else {
                    throw ServerRequestError.invalid(
                        message: "at most \(ServerAttachmentStore.maximumRequestImages) images "
                            + "per request",
                        param: "messages", code: "too_many_images")
                }
                let lease = try attachmentLease()
                let token = UUID().uuidString.lowercased()
                let writer = try StreamingBase64FileWriter(
                    directoryURL: lease.directoryURL)
                try append(contentsOf: Array("turbofieldfare-attachment:\(token)".utf8))
                string = StringState(mode: .staging,
                                     writer: writer,
                                     replacementToken: token)
            } else {
                try append(contentsOf: state.rawPrefix)
                string = StringState(mode: .plain)
            }
            return
        }
        state.decodedPrefix.append(contentsOf: decoded)
        if state.decodedPrefix.count > 64 {
            try append(contentsOf: state.rawPrefix)
            string = StringState(mode: .plain)
            return
        }
        string = state
    }

    /// The base64 payload: whole spans up to the next quote or backslash go to
    /// the writer at once. Escaped payload bytes are decoded and fed
    /// individually; anything that decodes outside the base64 alphabet fails in
    /// the writer exactly as it would have arrived raw.
    private func consumeStagingPayload(_ bytes: [UInt8], from start: Int) throws -> Int {
        var state = string!
        if state.escape.isActive {
            let decoded = try state.escape.consume(bytes[start])
            if !decoded.isEmpty { try state.writer!.feed(contentsOf: decoded) }
            string = state
            return start + 1
        }
        let byte = bytes[start]
        if byte == UInt8(ascii: "\"") {
            let staged = try state.writer!.finish()
            totalImageBytes += staged.encodedBytes
            guard totalImageBytes <= ServerAttachmentStore.maximumRequestImageBytes else {
                throw ServerRequestError.invalid(
                    message: "inline image bytes exceed the server limit",
                    param: "messages", code: "image_too_large")
            }
            stagedImages[state.replacementToken!] = staged
            try append(byte)
            string = nil
            return start + 1
        }
        if byte == UInt8(ascii: "\\") {
            state.escape.begin()
            string = state
            return start + 1
        }
        let end = Self.nextQuoteOrBackslash(bytes, from: start + 1)
        try state.writer!.feed(contentsOf: bytes[start..<end])
        string = state
        return end
    }

    private static func nextQuoteOrBackslash(_ bytes: [UInt8], from start: Int) -> Int {
        var index = start
        while index < bytes.count {
            let byte = bytes[index]
            if byte == UInt8(ascii: "\"") || byte == UInt8(ascii: "\\") { return index }
            index += 1
        }
        return index
    }

    private func attachmentLease() throws -> ServerAttachmentLease {
        if let lease { return lease }
        let created = try ServerAttachmentDirectory.makeLease(in: attachmentRoot)
        lease = created
        return created
    }

    private func append(_ byte: UInt8) throws {
        json.append(byte)
        try checkSanitizedSize()
    }

    private func append(contentsOf bytes: some Sequence<UInt8>) throws {
        json.append(contentsOf: bytes)
        try checkSanitizedSize()
    }

    private func checkSanitizedSize() throws {
        guard json.count <= Self.maximumSanitizedBytes else {
            throw ServerRequestError.invalid(
                message: "non-image request content is too large",
                param: nil, code: "request_too_large")
        }
    }
}

/// Decodes JSON string escape sequences inside an intercepted image URL, where
/// the sanitized body keeps a replacement token and the raw bytes are gone —
/// the URL has to be matched and staged from what the escapes mean, not how
/// they were written.
private struct JSONStringEscapeDecoder {
    private enum State {
        case idle
        case started
        case unicode([UInt8])
    }

    private var state = State.idle

    var isActive: Bool {
        if case .idle = state { return false }
        return true
    }

    /// Enters the sequence; the caller has just consumed the backslash.
    mutating func begin() {
        state = .started
    }

    /// Feeds the next raw byte of the sequence. Returns the decoded bytes once
    /// the sequence completes, and an empty array while it is still open.
    mutating func consume(_ byte: UInt8) throws -> [UInt8] {
        switch state {
        case .idle:
            preconditionFailure("consume called outside an escape sequence")
        case .started:
            switch byte {
            case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"):
                state = .idle
                return [byte]
            case UInt8(ascii: "b"): state = .idle; return [8]
            case UInt8(ascii: "f"): state = .idle; return [12]
            case UInt8(ascii: "n"): state = .idle; return [10]
            case UInt8(ascii: "r"): state = .idle; return [13]
            case UInt8(ascii: "t"): state = .idle; return [9]
            case UInt8(ascii: "u"):
                state = .unicode([])
                return []
            default:
                throw Self.invalidEscape
            }
        case .unicode(var digits):
            guard Self.hexValue(byte) != nil else { throw Self.invalidEscape }
            digits.append(byte)
            guard digits.count == 4 else {
                state = .unicode(digits)
                return []
            }
            state = .idle
            let value = digits.reduce(UInt32(0)) { $0 << 4 | UInt32(Self.hexValue($1)!) }
            // Surrogate pairs only encode characters no data URL may contain,
            // so they are refused rather than paired across two sequences.
            guard let scalar = Unicode.Scalar(value) else { throw Self.invalidEscape }
            return Array(String(scalar).utf8)
        }
    }

    private static let invalidEscape = ServerRequestError.invalid(
        message: "image_url contains an unsupported escape sequence",
        param: "messages", code: "invalid_image_url")

    private static func hexValue(_ byte: UInt8) -> UInt8? {
        switch byte {
        case UInt8(ascii: "0")...UInt8(ascii: "9"): byte - UInt8(ascii: "0")
        case UInt8(ascii: "a")...UInt8(ascii: "f"): byte - UInt8(ascii: "a") + 10
        case UInt8(ascii: "A")...UInt8(ascii: "F"): byte - UInt8(ascii: "A") + 10
        default: nil
        }
    }
}
