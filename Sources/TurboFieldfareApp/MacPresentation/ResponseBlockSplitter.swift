import Foundation

/// Splits a response into the top-level blocks that can be styled while the
/// answer is still arriving, and names the trailing block that is still being
/// written.
///
/// Blank lines are almost the only splitter, and a fence line is the one
/// exception. Everything else stays with the block above it: `Title` followed
/// by `---` is a setext heading, a `|` row that follows a paragraph line
/// belongs to that paragraph, a list item that follows another keeps its
/// ordinal from the list it is in — and a progressive render is only worth
/// having if a completed block looks exactly like the final one. A fenced
/// block therefore absorbs the blank lines inside it, and a list absorbs the
/// blank lines between its items, because the parser reads both as one block.
/// A fence opener has to split, for the same reason: CommonMark lets it
/// interrupt a paragraph, so a parser handed the two together reads the
/// listing as code where this scanner would have read prose.
///
/// One thing a block cannot carry is a reference to something outside it. A
/// `[label][ref]` whose `[ref]: url` definition sits in a later block renders
/// with its brackets showing, because the block that uses the label is parsed
/// on its own. Resolving it would mean scanning the whole response on every
/// tick, which is the shape the streaming timing gate exists to catch, and
/// doing it only at finalize would put the flip back that the per-block
/// finalize removed. The target models emit reference links rarely.
///
/// Boundaries are UTF-8 offsets and the scan is resumable so a tick costs the
/// bytes that arrived, not the bytes that are there: re-scanning a 10 KB fenced
/// block on each of the ~1,200 ticks it takes to stream is the quadratic shape
/// the streaming timing gate exists to catch.
enum ResponseBlockSplitter {
    enum Kind: Equatable {
        case paragraph
        case heading
        case fencedCode
        case indentedCode
        case table
        case list
        case quote
        case displayMath
        case thematicBreak
    }

    struct Fence: Equatable {
        let marker: UInt8
        let length: Int
        let indent: Int
        /// First byte after the opening fence line.
        let bodyStart: Int
        /// First byte of the closing fence line, or the end of the block.
        let bodyEnd: Int
        let isClosed: Bool
    }

    struct Block: Equatable {
        /// UTF-8 offset of the block's first byte.
        let start: Int
        /// UTF-8 offset one past the block's last non-blank byte; the line
        /// terminator that follows is not part of the block.
        let end: Int
        let kind: Kind
        let fence: Fence?

        var utf8Range: Range<Int> { start..<end }
    }

    /// Scanner position that survives an append. Only complete lines commit,
    /// so nothing already in `completed` can change when more text arrives.
    struct ScanState: Equatable {
        var cursor = 0
        var current: Current?
        /// A blank line has ended the current block's content, but a list or an
        /// indented code block may still continue past it.
        var pendingBlank = false
    }

    struct Current: Equatable {
        var start: Int
        var contentEnd: Int
        var kind: Kind
        var fenceMarker: UInt8 = 0
        var fenceLength = 0
        var fenceIndent = 0
        var fenceBodyStart = 0
        var fenceBodyEnd = 0
        var fenceClosed = false
        /// Content column of the outermost item a list block is inside. A
        /// fence at or past it is the item's own listing; one before it ends
        /// the list.
        var listContentColumn = 0
        /// A fence opened inside a list item. The block stays a list — the
        /// whole thing goes through the markdown pass — but until the fence
        /// closes its lines are code, so no blank line inside it commits the
        /// block and no line inside it opens anything.
        var innerFenceMarker: UInt8 = 0
        var innerFenceLength = 0
    }

    struct Split: Equatable {
        var completed: [Block] = []
        var open: Block?
        var scan = ScanState()

        var blocks: [Block] { completed + (open.map { [$0] } ?? []) }
    }

    static func split(_ response: String) -> Split {
        scan(response, from: Split())
    }

    /// Resumes `previous`. The caller must have established that `response`
    /// extends the string `previous` was produced from; a shorter or diverging
    /// response has to go through `split(_:)` instead, because the committed
    /// blocks are not re-examined.
    static func split(_ response: String, resuming previous: Split) -> Split {
        scan(response, from: previous)
    }

    static func text(_ range: Range<Int>, in response: String) -> String {
        withUTF8(response) { bytes in
            let lower = min(max(range.lowerBound, 0), bytes.count)
            let upper = min(max(range.upperBound, lower), bytes.count)
            return String(decoding: UnsafeBufferPointer(
                rebasing: bytes[lower..<upper]), as: UTF8.self)
        }
    }

    // MARK: - Scanning

    private static func scan(_ response: String, from previous: Split) -> Split {
        withUTF8(response) { bytes in
            var state = previous.scan
            var completed = previous.completed
            var index = min(state.cursor, bytes.count)
            while index < bytes.count {
                guard let terminator = newlineIndex(bytes, from: index) else { break }
                consume(
                    bytes,
                    line: index..<terminator,
                    lineEnd: terminator + 1,
                    terminated: true,
                    state: &state,
                    completed: &completed)
                index = terminator + 1
                state.cursor = index
            }

            // The trailing partial line is scanned into a copy: a block may
            // only be completed by a line that has already ended, or a block
            // the caller has drawn as finished could re-open on the next tick.
            var derived = state
            var ignored = completed
            if index < bytes.count {
                consume(
                    bytes,
                    line: index..<bytes.count,
                    lineEnd: bytes.count,
                    terminated: false,
                    state: &derived,
                    completed: &ignored)
            }

            var split = Split()
            split.completed = completed
            split.scan = state
            if let current = derived.current {
                split.open = block(from: current)
            }
            return split
        }
    }

    private static func consume(
        _ bytes: UnsafeBufferPointer<UInt8>,
        line: Range<Int>,
        lineEnd: Int,
        terminated: Bool,
        state: inout ScanState,
        completed: inout [Block]
    ) {
        let content = contentRange(bytes, line)

        if var current = state.current, current.kind == .fencedCode, !current.fenceClosed {
            if let run = FenceLine.run(bytes, in: content),
               run.closes(marker: current.fenceMarker, length: current.fenceLength) {
                current.fenceClosed = true
                current.fenceBodyEnd = line.lowerBound
                current.contentEnd = content.upperBound
                if terminated {
                    completed.append(block(from: current))
                    state.current = nil
                    state.pendingBlank = false
                } else {
                    state.current = current
                }
                return
            }
            current.contentEnd = content.upperBound
            current.fenceBodyEnd = lineEnd
            state.current = current
            return
        }

        if var current = state.current, current.innerFenceMarker != 0 {
            let indent = indentWidth(bytes, content)
            let blank = isBlank(bytes, content)
            // An unclosed fence inside an item cannot hold the rest of the
            // answer: a line indented before the item's content column has
            // left the item, and the fence with it.
            if blank || indent >= current.listContentColumn {
                if let run = FenceLine.run(
                    bytes,
                    in: content,
                    containerIndent: current.listContentColumn),
                   run.closes(
                    marker: current.innerFenceMarker,
                    length: current.innerFenceLength) {
                    current.innerFenceMarker = 0
                    current.innerFenceLength = 0
                }
                if !blank { current.contentEnd = content.upperBound }
                state.current = current
                state.pendingBlank = false
                return
            }
            current.innerFenceMarker = 0
            current.innerFenceLength = 0
            state.current = current
        }

        guard !isBlank(bytes, content) else {
            guard let current = state.current else { return }
            if terminated, current.kind != .list, current.kind != .indentedCode {
                completed.append(block(from: current))
                state.current = nil
                state.pendingBlank = false
            } else {
                state.pendingBlank = true
            }
            return
        }

        let indent = indentWidth(bytes, content)
        guard var current = state.current else {
            state.current = opening(bytes, content: content, lineEnd: lineEnd, indent: indent)
            state.pendingBlank = false
            return
        }

        if state.pendingBlank {
            let continues = continuation(
                of: current.kind,
                bytes: bytes,
                content: content,
                indent: indent)
            // An unterminated line may still turn into a list item, so the
            // block it would continue stays open until the line ends.
            guard continues || !terminated else {
                completed.append(block(from: current))
                state.current = opening(
                    bytes, content: content, lineEnd: lineEnd, indent: indent)
                state.pendingBlank = false
                return
            }
            state.pendingBlank = false
        }

        if current.kind == .list {
            // A nested item is inside the item above it; a sibling or an outer
            // one is the item the block's fences are measured against now.
            if let item = ContainerPrefix.listItem(bytes, in: content),
               item.markerIndent < current.listContentColumn {
                current.listContentColumn = item.contentColumn
            }
            if terminated, let run = FenceLine.run(
                bytes,
                in: content,
                containerIndent: current.listContentColumn) {
                current.innerFenceMarker = run.marker
                current.innerFenceLength = run.length
                current.contentEnd = content.upperBound
                state.current = current
                return
            }
        }

        // The one boundary a blank line does not draw. A fence interrupts a
        // paragraph in CommonMark, and absorbing it instead hands the rest of
        // the listing to the prose block: the blank line inside the code then
        // commits it, and the lines below open an indented-code block that
        // swallows the closing fence and the sentence after it. A list is
        // interrupted the same way once the fence is left of its content
        // column, which is where the item's own listing would have started.
        if terminated, current.kind != .indentedCode,
           FenceLine.run(bytes, in: content) != nil {
            completed.append(block(from: current))
            state.current = opening(bytes, content: content, lineEnd: lineEnd, indent: indent)
            return
        }

        current.contentEnd = content.upperBound
        state.current = current
    }

    private static func opening(
        _ bytes: UnsafeBufferPointer<UInt8>,
        content: Range<Int>,
        lineEnd: Int,
        indent: Int
    ) -> Current {
        var current = Current(
            start: content.lowerBound,
            contentEnd: content.upperBound,
            kind: kind(bytes, content: content, indent: indent))
        if current.kind == .fencedCode, let fence = FenceLine.run(bytes, in: content) {
            current.fenceMarker = fence.marker
            current.fenceLength = fence.length
            current.fenceIndent = fence.indent
            current.fenceBodyStart = lineEnd
            current.fenceBodyEnd = lineEnd
        }
        if current.kind == .list, let item = ContainerPrefix.listItem(bytes, in: content) {
            current.listContentColumn = item.contentColumn
        }
        return current
    }

    private static func block(from current: Current) -> Block {
        var fence: Fence?
        if current.kind == .fencedCode {
            fence = Fence(
                marker: current.fenceMarker,
                length: current.fenceLength,
                indent: current.fenceIndent,
                bodyStart: current.fenceBodyStart,
                bodyEnd: max(current.fenceBodyStart, current.fenceBodyEnd),
                isClosed: current.fenceClosed)
        }
        return Block(
            start: current.start,
            end: current.contentEnd,
            kind: current.kind,
            fence: fence)
    }

    // MARK: - Line classification

    private static func kind(
        _ bytes: UnsafeBufferPointer<UInt8>,
        content: Range<Int>,
        indent: Int
    ) -> Kind {
        if indent >= 4 { return .indentedCode }
        if FenceLine.run(bytes, in: content) != nil { return .fencedCode }
        let start = firstNonSpace(bytes, content)
        guard start < content.upperBound else { return .paragraph }
        if isThematicBreak(bytes, content) { return .thematicBreak }
        switch bytes[start] {
        case UInt8(ascii: "#"):
            var hashes = 0
            var index = start
            while index < content.upperBound, bytes[index] == UInt8(ascii: "#") {
                hashes += 1
                index += 1
            }
            if hashes <= 6, index >= content.upperBound || isSpace(bytes[index]) {
                return .heading
            }
        case UInt8(ascii: ">"):
            return .quote
        case UInt8(ascii: "|"):
            return .table
        case UInt8(ascii: "$"):
            if start + 1 < content.upperBound, bytes[start + 1] == UInt8(ascii: "$") {
                return .displayMath
            }
        default:
            break
        }
        if ContainerPrefix.listItem(bytes, in: content) != nil { return .list }
        return .paragraph
    }

    private static func continuation(
        of kind: Kind,
        bytes: UnsafeBufferPointer<UInt8>,
        content: Range<Int>,
        indent: Int
    ) -> Bool {
        switch kind {
        case .list:
            return indent >= 2 || ContainerPrefix.listItem(bytes, in: content) != nil
        case .indentedCode:
            return indent >= 4
        default:
            return false
        }
    }

    private static func isThematicBreak(
        _ bytes: UnsafeBufferPointer<UInt8>,
        _ content: Range<Int>
    ) -> Bool {
        var index = firstNonSpace(bytes, content)
        guard index < content.upperBound else { return false }
        let marker = bytes[index]
        guard marker == UInt8(ascii: "-") || marker == UInt8(ascii: "*")
            || marker == UInt8(ascii: "_") else {
            return false
        }
        var count = 0
        while index < content.upperBound {
            let byte = bytes[index]
            if byte == marker {
                count += 1
            } else if !isSpace(byte) {
                return false
            }
            index += 1
        }
        return count >= 3
    }

    // MARK: - Bytes

    private static func newlineIndex(
        _ bytes: UnsafeBufferPointer<UInt8>,
        from index: Int
    ) -> Int? {
        var cursor = index
        while cursor < bytes.count {
            if bytes[cursor] == UInt8(ascii: "\n") { return cursor }
            cursor += 1
        }
        return nil
    }

    private static func contentRange(
        _ bytes: UnsafeBufferPointer<UInt8>,
        _ line: Range<Int>
    ) -> Range<Int> {
        var end = line.upperBound
        while end > line.lowerBound, bytes[end - 1] == UInt8(ascii: "\r") { end -= 1 }
        return line.lowerBound..<end
    }

    private static func isBlank(
        _ bytes: UnsafeBufferPointer<UInt8>,
        _ content: Range<Int>
    ) -> Bool {
        for index in content where !isSpace(bytes[index]) { return false }
        return true
    }

    /// Byte offset of the first byte that is not indentation.
    private static func firstNonSpace(
        _ bytes: UnsafeBufferPointer<UInt8>,
        _ content: Range<Int>
    ) -> Int {
        var index = content.lowerBound
        while index < content.upperBound,
              bytes[index] == UInt8(ascii: " ") || bytes[index] == UInt8(ascii: "\t") {
            index += 1
        }
        return index
    }

    private static func indentWidth(
        _ bytes: UnsafeBufferPointer<UInt8>,
        _ content: Range<Int>
    ) -> Int {
        var width = 0
        for index in content {
            if bytes[index] == UInt8(ascii: " ") {
                width += 1
            } else if bytes[index] == UInt8(ascii: "\t") {
                width += 4
            } else {
                break
            }
        }
        return width
    }

    private static func isSpace(_ byte: UInt8) -> Bool {
        byte == UInt8(ascii: " ") || byte == UInt8(ascii: "\t") || byte == UInt8(ascii: "\r")
    }

    private static func withUTF8<R>(
        _ text: String,
        _ body: (UnsafeBufferPointer<UInt8>) -> R
    ) -> R {
        if let result = text.utf8.withContiguousStorageIfAvailable(body) { return result }
        var copy = text
        return copy.withUTF8 { body($0) }
    }
}
