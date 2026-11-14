import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanMemory
import TinyTitanCore

/// Bridges the memory subsystem to the server's own types.
///
/// `TinyTitanMemory` knows nothing about chat messages, tool definitions or
/// requests, and the serving code knows nothing about how memory is stored.
/// Everything that has to speak both lives here: deriving a session identity
/// from a conversation, turning memory tools into the tokenizer's function
/// definitions, and turning a model's tool call into a memory operation and
/// its result back into a message.
enum ServerMemory {
    /// The working directory a client declared in its system prompt, if any.
    ///
    /// Claude Code and Codex both tell the model where they are running:
    /// Claude Code as a "Working directory:" line, Codex as a `<cwd>` element
    /// in its environment block. That is exactly the project the session is
    /// about, and it arrives on every request without anyone configuring a
    /// header. Only absolute paths count, and only in system messages: a
    /// user pasting a transcript must not be able to move their own memory.
    static func declaredWorkingDirectory(in messages: [GFTokenizer.Message]) -> String? {
        // Built here rather than held statically: `Regex` is not Sendable,
        // and a literal costs nothing worth caching against a request that
        // is about to run a model.
        let patterns: [Regex<(Substring, Substring)>] = [
            // Codex: <environment_context><cwd>/path</cwd>
            /<cwd>\s*(\/[^<\n]+?)\s*<\/cwd>/,
            // Claude Code: "Working directory: /path", "Primary working
            // directory: /path", usually as a bulleted line in an
            // environment block.
            /(?im)^[ \t]*[-*]?[ \t]*(?:primary[ \t]+)?working[ \t]+directory:[ \t]*(\/\S+)/,
            // A plain "cwd: /path" line, which some tools emit.
            /(?im)^[ \t]*[-*]?[ \t]*cwd:[ \t]*(\/\S+)/,
        ]
        for message in messages where message.role == .system {
            guard let content = message.content, !content.isEmpty else { continue }
            for pattern in patterns {
                if let range = content.firstMatch(of: pattern)?.1 {
                    let path = String(range).trimmingCharacters(in: .whitespaces)
                    if path.hasPrefix("/") { return path }
                }
            }
        }
        return nil
    }

    /// The request that distils a session into facts.
    ///
    /// Its own conversation, with no tools and no memory fragment: it is not
    /// a turn of the session, and it must not open one. The transcript is the
    /// journal's filtered turns, so tool results and file dumps are already
    /// gone. Existing keys are shown so an update lands on the address it
    /// changes rather than beside it.
    static func consolidationRequest(
        turns: [JournalTurn],
        existing: [MemoryRecord],
        workspace: String
    ) -> ValidatedChatRequest {
        var transcript = ""
        for turn in turns {
            transcript += "USER: \(turn.prompt)\n\nASSISTANT: \(turn.reply)\n\n---\n\n"
        }
        // Keys by name only for the namespaces this session touches; the
        // other namespaces as one line each with a count. Values only for
        // keys the session mentions. Measured, listing every key by name was
        // the whole overhead of the feature and it grew with the store: v2
        // cost 15% more than v1 on a hundred-chapter novel for it.
        let haystack = transcript.lowercased()
        let mentioned = existing.filter { Self.isMentioned($0.key.rawValue, in: haystack) }
        var known = Self.keyListing(existing, mentioned: mentioned)
        if !mentioned.isEmpty {
            known +=
                "\n\nCurrent values of the keys this session touches:\n"
                + mentioned.prefix(40).map { record in
                    let value = record.value.replacingOccurrences(of: "\n", with: " ")
                    return "- \(record.key.rawValue) = \(value.prefix(160))"
                }.joined(separator: "\n")
        }
        let system =
            "You distil a finished working session into durable facts for a memory "
            + "store scoped to the project `\(workspace)`. Later sessions will see these "
            + "facts and nothing else from this conversation, so record exactly what a "
            + "future session must not contradict: decisions and the reasons for them, "
            + "fixed attributes, rules and constraints, current state, and what changed. "
            + "Do not record conversation, reasoning, code, or anything a future session "
            + "can re-derive.\n\n"
            + "You are shown what memory already holds. Write ONLY facts this session "
            + "added or changed. Do not restate a fact that is unchanged, and never "
            + "write a key whose value you cannot take from this session: no \"not "
            + "specified\", \"unknown\", \"N/A\" or guesses -- omit the key instead. "
            + "When this session changes a state that memory holds, reuse that key "
            + "exactly and write the new state; the old one is kept as history "
            + "automatically. Never create a second key for a fact memory already "
            + "holds under another name: if `state/inn` exists, the inn's state goes "
            + "to `state/inn`, not to `continuity/inn_status`. One fact per key: a "
            + "group of numbers or attributes is several keys, not one blob. Booleans "
            + "are true or false.\n\n"
            + "Output only a JSON array, in a ```json block, of objects with keys "
            + "\"key\", \"value\", \"importance\" (0 to 1; fixed attributes and "
            + "rules high, passing state lower) and \"source\". Set \"source\" to "
            + "\"user\" when the fact is something the USER stated or required, and "
            + "\"assistant\" when it comes from the assistant's own output. This is a "
            + "labelling question about where the fact appears in the transcript "
            + "above, not a judgement about whether it is true: if the USER wrote it, "
            + "it is \"user\". Keys are lowercase path-like names "
            + "such as `characters/marcus/eyes`, `decisions/storage`, `state/inn` or "
            + "`rules/weather`. A fact about the PERSON rather than this project -- a "
            + "convention they want everywhere, their language, their tone, a tool "
            + "they always use -- also carries \"global\": true, and goes under "
            + "`preferences/...`; project facts never do. Return [] if nothing durable "
            + "was added or changed."
        let user = "Memory already holds these keys:\n\(known)\n\nThe session:\n\n\(transcript)"
        return ValidatedChatRequest(
            messages: [
                GFTokenizer.Message(role: .system, content: system),
                GFTokenizer.Message(role: .user, content: user),
            ],
            tools: [],
            stream: false,
            includeUsage: false,
            // Sixteen facts ran to about 700 tokens; a session with a plot
            // event ran into the old 900 cap and came back unparseable.
            generationConfig: GenerationConfig(maxNewTokens: 2000),
            maximumCompletionTokens: 2000,
            // B6: the engine's own generation, and not watched.
            isEngineInternal: true)
    }

    /// The facts a consolidation produced, or none if it produced nothing
    /// usable. A malformed key is skipped rather than failing the batch: one
    /// bad name must not cost the other twenty facts.
    static func consolidationRecords(from text: String) -> [MemoryRecord] {
        var candidates: [String] = []
        let fenced = try? NSRegularExpression(pattern: "```(?:json)?\\s*(\\[[\\s\\S]*?\\])\\s*```")
        if let fenced,
            let match = fenced.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
            let range = Range(match.range(at: 1), in: text)
        {
            candidates.append(String(text[range]))
        }
        if let open = text.firstIndex(of: "["), let close = text.lastIndex(of: "]"), open < close {
            candidates.append(String(text[open...close]))
        }
        // A truncated array -- the output cap landed mid-object -- still has
        // every complete object before the cut. Losing sixteen facts to a
        // seventeenth that was cut off is the failure this recovers from.
        if let open = text.firstIndex(of: "["), let lastClose = text.lastIndex(of: "}"),
            open < lastClose
        {
            candidates.append(String(text[open...lastClose]) + "]")
        }
        // No array at all: one bare object, or several in a row. A session
        // with a single fact came back as `{...}` and was scored as having
        // produced nothing, which lost the one fact that session was for.
        if let open = text.firstIndex(of: "{"), let close = text.lastIndex(of: "}"), open < close {
            let objects = String(text[open...close])
            candidates.append(
                "["
                    + objects.replacingOccurrences(
                        of: #"\}\s*\{"#, with: "},{", options: .regularExpression) + "]")
        }
        for candidate in candidates {
            guard let data = candidate.data(using: .utf8),
                let parsed = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
            else { continue }
            var records: [MemoryRecord] = []
            for entry in parsed {
                guard let rawKey = entry["key"] as? String,
                    let key = try? MemoryKey(validating: rawKey.lowercased()),
                    let raw = entry["value"], !(raw is NSNull)
                else { continue }
                let value: String
                if let flag = raw as? Bool {
                    value = flag ? "true" : "false"
                } else {
                    value = "\(raw)".trimmingCharacters(in: .whitespacesAndNewlines)
                }
                guard !value.isEmpty, !Self.isPlaceholder(value) else { continue }
                let importance =
                    (entry["importance"] as? Double)
                    ?? (entry["importance"] as? Int).map(Double.init)
                var record = MemoryRecord(key: key, value: value, importance: importance)
                record.isGlobal = (entry["global"] as? Bool) ?? false
                // Anything but an explicit "user" is the model's own: an
                // extraction that omits the field, or answers something
                // unexpected, degrades to today's behaviour rather than to a
                // protected fact. Guessing the other way would let a
                // confused label make an invention permanent.
                record.isUserAsserted =
                    (entry["source"] as? String)?.lowercased() == "user"
                records.append(record)
            }
            if !records.isEmpty || parsed.isEmpty { return records }
        }
        return []
    }

    /// The key names an extraction is shown.
    ///
    /// Namespaces the session mentions are listed in full, so an update lands
    /// on the key it changes. Every other namespace is one line with a count:
    /// enough to tell the model the namespace exists and how it is spelled,
    /// which is what stops it inventing a parallel one. Capped, because a
    /// list that grows with the store is the cost curve this replaces.
    static func keyListing(
        _ existing: [MemoryRecord], mentioned: [MemoryRecord],
        maximumListed: Int = 60
    ) -> String {
        guard !existing.isEmpty else { return "(none yet)" }
        let touched = Set(mentioned.map { namespace(of: $0.key.rawValue) })
        var lines: [String] = []
        var listed = 0
        var byNamespace: [String: [String]] = [:]
        for record in existing {
            byNamespace[namespace(of: record.key.rawValue), default: []].append(record.key.rawValue)
        }
        for name in byNamespace.keys.sorted() {
            let keys = (byNamespace[name] ?? []).sorted()
            if touched.contains(name), listed + keys.count <= maximumListed {
                lines.append(contentsOf: keys.map { "- \($0)" })
                listed += keys.count
            } else if touched.contains(name) {
                let room = max(0, maximumListed - listed)
                lines.append(contentsOf: keys.prefix(room).map { "- \($0)" })
                lines.append("- \(name)/ ... and \(keys.count - room) more keys")
                listed = maximumListed
            } else {
                lines.append(
                    "- \(name)/ (\(keys.count) key\(keys.count == 1 ? "" : "s"), not touched by this session)"
                )
            }
        }
        return lines.joined(separator: "\n")
    }

    private static func namespace(of key: String) -> String {
        key.split(separator: "/").first.map(String.init) ?? key
    }

    /// Whether a key's subject appears in the transcript: any segment of
    /// three or more characters, split on `/`, `_` and `-`, matched as a
    /// whole word. `state/inn_status` matches a session that mentions the
    /// inn; `characters/rosa/eyes` matches one that mentions Rosa. Whole
    /// words, because "inn" must not match "beginning" and "ferry" must not
    /// match nothing but itself.
    static func isMentioned(_ key: String, in haystack: String) -> Bool {
        let parts = key.lowercased()
            .split(whereSeparator: { $0 == "/" || $0 == "_" || $0 == "-" })
            .map(String.init)
            .filter { $0.count >= 3 }
        guard !parts.isEmpty else { return false }
        return parts.contains { part in
            let pattern =
                "(?<![a-z0-9])" + NSRegularExpression.escapedPattern(for: part) + "(?![a-z0-9])"
            return haystack.range(of: pattern, options: .regularExpression) != nil
        }
    }

    /// Routes a new fact to the key memory already uses for it.
    ///
    /// The extraction is told to reuse keys and still invents parallel
    /// namespaces under sampling: `continuity/inn_status` beside
    /// `state/inn_status`. Two keys for one fact put a contradiction in the
    /// bootstrap, and the model then answers whichever it read last.
    ///
    /// The rule is narrow on purpose: a new key is routed only when
    /// everything after its first segment matches exactly one existing key
    /// whose first segment differs. That is the shape of a renamed
    /// namespace, and nothing else. A first version matched on the final
    /// segment alone and routed `characters/ines/knows_photo_content` onto
    /// `characters/marcus/knows_photo_content`, which gave Marcus a fact he
    /// was not allowed to have until chapter 60.
    static func reconcile(
        _ records: [MemoryRecord],
        existing: [MemoryRecord]
    ) -> (
        records: [MemoryRecord],
        merged: [(from: String, to: String)]
    ) {
        var byPath: [String: [MemoryKey]] = [:]
        for record in existing {
            if let path = pathAfterNamespace(record.key.rawValue) {
                byPath[path, default: []].append(record.key)
            }
        }
        let existingKeys = Set(existing.map(\.key.rawValue))
        var out: [MemoryRecord] = []
        var merged: [(from: String, to: String)] = []
        for record in records {
            let key = record.key.rawValue
            guard !existingKeys.contains(key),
                let path = pathAfterNamespace(key), path.count >= 4,
                let targets = byPath[path], targets.count == 1,
                let target = targets.first, target.rawValue != key
            else {
                out.append(record)
                continue
            }
            var routed = record
            routed.key = target
            out.append(routed)
            merged.append((from: key, to: target.rawValue))
        }
        return (out, merged)
    }

    /// `state/inn_status` gives `inn_status`; `characters/ines/eyes` gives
    /// `ines/eyes`; a single-segment key gives nil, because it has no
    /// namespace to have been renamed.
    private static func pathAfterNamespace(_ key: String) -> String? {
        let segments = key.split(separator: "/", omittingEmptySubsequences: false)
        guard segments.count >= 2 else { return nil }
        return segments.dropFirst().joined(separator: "/")
    }

    /// Values that are the absence of a fact. Writing one over a real value
    /// is worse than writing nothing, and a model shown a key it has no
    /// information for will produce exactly these.
    static func isPlaceholder(_ value: String) -> Bool {
        let folded = value.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: " .\"'"))
        return [
            "not specified", "unspecified", "unknown", "n/a", "na", "none", "null",
            "tbd", "not mentioned", "not stated", "not given", "no change", "unchanged",
        ]
        .contains(folded)
    }

    /// A stable session id for a conversation.
    ///
    /// The API is stateless and clients send the whole history each turn, so
    /// there is no session id to read. The first user message plus the
    /// workspace identifies a conversation well enough to keep one session's
    /// memory continuous across its turns, and it changes when a new
    /// conversation starts, which is when a new session should begin.
    static func sessionIdentifier(
        messages: [GFTokenizer.Message],
        workspace: String
    ) -> String {
        let seed = messages.first { $0.role == .user }?.content ?? ""
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in (workspace + "\u{0}" + seed).utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return String(format: "s-%016llx", hash)
    }

    /// True when this request is the opening turn of a conversation, which is
    /// where the bootstrap belongs. Later turns reuse the session.
    @available(
        *, deprecated,
        message: "No caller (TT-051): sessions are identified by sessionIdentifier."
    )
    static func isFirstTurn(_ messages: [GFTokenizer.Message]) -> Bool {
        !messages.contains { $0.role == .assistant }
    }

    /// Memory tools in the tokenizer's own definition type.
    static func functionDefinitions(_ definitions: [MemoryToolDefinition])
        -> [GFTokenizer.FunctionDefinition]
    {
        definitions.map { definition in
            GFTokenizer.FunctionDefinition(
                name: definition.name,
                description: definition.description,
                parameters: jsonValue(from: definition.parameters.jsonObject))
        }
    }

    /// Adds memory tools to a request's own, without displacing them.
    ///
    /// A client's tool of the same name wins: the client executes its tools
    /// and we execute ours, and two definitions of one name would make the
    /// model's call ambiguous.
    static func merging(
        tools: [GFTokenizer.FunctionDefinition],
        memory: [GFTokenizer.FunctionDefinition]
    )
        -> [GFTokenizer.FunctionDefinition]
    {
        let existing = Set(tools.map(\.name))
        return tools + memory.filter { !existing.contains($0.name) }
    }

    /// Converts a parsed tool call's arguments into memory tool values.
    static func arguments(from json: JSONValue) -> [String: MemoryToolValue] {
        guard case .object(let fields) = json else { return [:] }
        var result: [String: MemoryToolValue] = [:]
        for (name, value) in fields {
            result[name] = toolValue(from: value)
        }
        return result
    }

    private static func toolValue(from json: JSONValue) -> MemoryToolValue {
        switch json {
        case .string(let text): return .string(text)
        case .number(let number): return .number(number)
        case .integer(let number): return .number(Double(number))
        case .unsignedInteger(let number): return .number(Double(number))
        case .decimal(let number): return .number(NSDecimalNumber(decimal: number).doubleValue)
        case .bool(let flag): return .bool(flag)
        case .array(let items):
            return .stringArray(
                items.compactMap { item in
                    if case .string(let text) = item { return text }
                    return nil
                })
        case .null: return .null
        case .object:
            // Objects are not a memory argument type; rendering it back to
            // text keeps a malformed call debuggable instead of silent.
            return .string(String(describing: json))
        }
    }

    /// The assistant turn that made a set of tool calls, as history.
    static func assistantMessage(content: String, calls: [ParsedToolCall])
        -> GFTokenizer.Message
    {
        GFTokenizer.Message(
            role: .assistant,
            content: content.isEmpty ? nil : content,
            toolCalls: calls.map {
                GFTokenizer.HistoricalToolCall(id: $0.id, name: $0.name, arguments: $0.arguments)
            })
    }

    /// One tool result, as the message the model reads next.
    static func toolResultMessage(
        call: ParsedToolCall,
        result: MemoryToolResult
    ) -> GFTokenizer.Message {
        GFTokenizer.Message(
            role: .tool,
            content: result.jsonString(),
            toolCallID: call.id,
            name: call.name)
    }

    private static func jsonValue(from object: Any) -> JSONValue {
        switch object {
        case let dictionary as [String: Any]:
            var mapped: [String: JSONValue] = [:]
            for (key, value) in dictionary { mapped[key] = jsonValue(from: value) }
            return .object(mapped)
        case let array as [Any]:
            return .array(array.map(jsonValue(from:)))
        case let text as String:
            return .string(text)
        case let flag as Bool:
            return .bool(flag)
        case let number as Int:
            return .integer(Int64(number))
        case let number as Double:
            return .number(number)
        default:
            return .null
        }
    }
}
