import Foundation

/// Rewrites the LaTeX commands language models emit constantly but the pinned
/// typesetter rejects, into forms it accepts. Every rule either preserves the
/// meaning (delimiter spellings, environment aliases) or drops
/// presentation-only decoration (`\boxed`, `\tag`, `\color`) rather than losing
/// the whole equation to a raw-text fallback. A command the pinned revision
/// already renders gets no rule, because rewriting it would trade real
/// notation for an approximation; a command with no rule passes through and
/// fails visibly instead of being silently mangled.
enum MathCommandNormalizer {
    /// A rewritten argument is rewritten in turn, so `\boxed{\boxed{...}}`
    /// costs one stack frame per level. Real answers nest a handful deep;
    /// thousands deep is a runaway generation, and overflowing the stack there
    /// takes the app down. Past this depth the span is handed back untouched
    /// and renders as raw text instead.
    private static let maximumDepth = 256

    static func normalize(_ latex: String) -> String {
        var overflowed = false
        let rewritten = rewrite(Array(latex), depth: 0, overflowed: &overflowed)
        return overflowed ? latex : rewritten
    }

    private enum Rule {
        /// Replace the command token itself; any arguments stay in the stream.
        case token(String)
        /// Consume `count` brace groups and emit the one at `keep`.
        case keepArgument(keep: Int, count: Int)
        /// Consume `count` brace groups and emit `replacement` instead.
        case replaceArguments(String, count: Int)
        /// `\underbrace{x}_{y}`: keep the group, drop the annotation script.
        case keepArgumentDroppingScript
        /// `\overset{a}{b}`: `{b}^{a}` when `a` is one token, otherwise `b`.
        case script(String)
    }

    private static let rules: [String: Rule] = [
        "boxed": .keepArgument(keep: 0, count: 1),
        "fbox": .keepArgument(keep: 0, count: 1),
        "mbox": .keepArgument(keep: 0, count: 1),
        "cancel": .keepArgument(keep: 0, count: 1),
        "xcancel": .keepArgument(keep: 0, count: 1),
        "sout": .keepArgument(keep: 0, count: 1),
        "textcolor": .keepArgument(keep: 1, count: 2),
        "colorbox": .keepArgument(keep: 1, count: 2),
        "color": .replaceArguments("", count: 1),
        "tag": .replaceArguments("", count: 1),
        "label": .replaceArguments("", count: 1),
        "xrightarrow": .replaceArguments("\\rightarrow", count: 1),
        "xleftarrow": .replaceArguments("\\leftarrow", count: 1),
        "underbrace": .keepArgumentDroppingScript,
        "overbrace": .keepArgumentDroppingScript,
        "overset": .script("^"),
        "underset": .script("_"),
        "stackrel": .script("^"),
        "nonumber": .token(""),
        "middle": .token(""),
        "hline": .token(""),
        "lvert": .token("|"),
        "rvert": .token("|"),
        "lVert": .token("\\|"),
        "rVert": .token("\\|"),
        "dots": .token("\\ldots"),
        "dotsc": .token("\\ldots"),
        "therefore": .token("\\text{\u{2234}}"),
        "because": .token("\\text{\u{2235}}"),
        ":": .token("\\,"),
        "bmod": .token("\\ \\mathrm{mod}\\ "),
        "pmb": .token("\\mathbf"),
        "mathscr": .token("\\mathcal"),
        "coloneqq": .token(":="),
        "eqqcolon": .token("=:"),
    ]

    /// Mathematical characters a model writes as Unicode instead of as
    /// commands. The pinned typesetter rejects every one of these outright, so
    /// an answer that says "x ≤ 5" lost the whole equation to raw text; the
    /// command spellings render identically and cost nothing. A character the
    /// typesetter already draws gets no entry, by the same rule the command
    /// table follows.
    static let symbols: [Character: String] = [
        "α": #"\alpha"#,
        "β": #"\beta"#,
        "γ": #"\gamma"#,
        "δ": #"\delta"#,
        "ε": #"\epsilon"#,
        "ϵ": #"\epsilon"#,
        "ζ": #"\zeta"#,
        "η": #"\eta"#,
        "θ": #"\theta"#,
        "ϑ": #"\theta"#,
        "ι": #"\iota"#,
        "κ": #"\kappa"#,
        "λ": #"\lambda"#,
        "μ": #"\mu"#,
        "ν": #"\nu"#,
        "ξ": #"\xi"#,
        "π": #"\pi"#,
        "ρ": #"\rho"#,
        "σ": #"\sigma"#,
        "ς": #"\sigma"#,
        "τ": #"\tau"#,
        "υ": #"\upsilon"#,
        "φ": #"\phi"#,
        "ϕ": #"\phi"#,
        "χ": #"\chi"#,
        "ψ": #"\psi"#,
        "ω": #"\omega"#,
        "Γ": #"\Gamma"#,
        "Δ": #"\Delta"#,
        "Θ": #"\Theta"#,
        "Λ": #"\Lambda"#,
        "Ξ": #"\Xi"#,
        "Π": #"\Pi"#,
        "Σ": #"\Sigma"#,
        "Υ": #"\Upsilon"#,
        "Φ": #"\Phi"#,
        "Ψ": #"\Psi"#,
        "Ω": #"\Omega"#,
        "×": #"\times"#,
        "÷": #"\div"#,
        "±": #"\pm"#,
        "∓": #"\mp"#,
        "·": #"\cdot"#,
        "⋅": #"\cdot"#,
        "∗": #"\ast"#,
        "≤": #"\le"#,
        "≥": #"\ge"#,
        "≠": #"\ne"#,
        "≈": #"\approx"#,
        "≡": #"\equiv"#,
        "∼": #"\sim"#,
        "∝": #"\propto"#,
        "→": #"\to"#,
        "←": #"\leftarrow"#,
        "↔": #"\leftrightarrow"#,
        "⇒": #"\Rightarrow"#,
        "⇐": #"\Leftarrow"#,
        "⇔": #"\Leftrightarrow"#,
        "∞": #"\infty"#,
        "∑": #"\sum"#,
        "∏": #"\prod"#,
        "∫": #"\int"#,
        "∂": #"\partial"#,
        "∇": #"\nabla"#,
        "∠": #"\angle"#,
        "∈": #"\in"#,
        "∉": #"\notin"#,
        "⊂": #"\subset"#,
        "⊆": #"\subseteq"#,
        "∪": #"\cup"#,
        "∩": #"\cap"#,
        "∅": #"\emptyset"#,
        "∀": #"\forall"#,
        "∃": #"\exists"#,
        "−": #"-"#,
        "…": #"\ldots"#,
        "½": #"\frac{1}{2}"#,
        "¼": #"\frac{1}{4}"#,
        "¾": #"\frac{3}{4}"#,
        "ℝ": #"\mathbb{R}"#,
        "ℕ": #"\mathbb{N}"#,
        "ℤ": #"\mathbb{Z}"#,
        "ℚ": #"\mathbb{Q}"#,
        "ℂ": #"\mathbb{C}"#,
        // The pinned build drops every one of these without an error, so an
        // answer that used them typeset with the operator simply missing.
        "%": "\\%",
        "#": "\\#",
        "$": "\\$",
        "°": #"^{\circ}"#,
        "¬": #"\neg"#,
        "∨": #"\lor"#,
        "∧": #"\land"#,
        "∘": #"\circ"#,
        "⊕": #"\oplus"#,
        "⊗": #"\otimes"#,
        "≪": #"\ll"#,
        "≫": #"\gg"#,
        "⟨": #"\langle"#,
        "⟩": #"\rangle"#,
        "⌊": #"\lfloor"#,
        "⌋": #"\rfloor"#,
        "⌈": #"\lceil"#,
        "⌉": #"\rceil"#,
        "ℓ": #"\ell"#,
        "ℏ": #"\hbar"#,
        "ℵ": #"\aleph"#,
        "ℜ": #"\Re"#,
        "ℑ": #"\Im"#,
        "∮": #"\oint"#,
        "∬": #"\iint"#,
        "⋯": #"\cdots"#,
        "⋮": #"\vdots"#,
        "⋱": #"\ddots"#,
        "↦": #"\mapsto"#,
        "↑": #"\uparrow"#,
        "↓": #"\downarrow"#,
        "∖": #"\setminus"#,
        "∣": #"\mid"#,
        "∥": #"\parallel"#,
        "⊥": #"\perp"#,
        "⊤": #"\top"#,
        "≅": #"\cong"#,
        "≃": #"\simeq"#,
        "⊃": #"\supset"#,
        "⊇": #"\supseteq"#,
        // U+2206 and U+00B5 and U+03F1 are the look-alikes of characters the
        // table already carries, and the build treats them as different.
        "∆": #"\Delta"#,
        "µ": #"\mu"#,
        "ϱ": #"\rho"#,
        "□": #"\square"#,
        "∴": "\\text{\u{2234}}",
        "∵": "\\text{\u{2235}}",
    ]

    /// Spacing the build drops. These are not in `symbols` because a lone
    /// space produces a zero-size image, which the conformer rejects: they are
    /// pinned in context instead.
    static let spacing: [Character: String] = [
        "~": #"\ "#,
        "\u{00A0}": #"\ "#,
        "\u{2002}": #"\ "#,
        "\u{2003}": #"\ "#,
        "\u{2009}": #"\,"#,
    ]

    /// Invisible characters a model pastes in from formatted text. They mean
    /// nothing in an equation and the build drops them anyway. A zero-width
    /// joiner or non-joiner is not here: Unicode grapheme breaking keeps those
    /// inside the character before them, so they never arrive on their own and
    /// the build draws the grapheme they made.
    static let removed: Set<Character> = ["\u{200B}", "\u{FEFF}"]

    /// Prime marks, and how many primes each one is.
    static let primes: [Character: Int] = ["\u{2032}": 1, "\u{2019}": 1, "\u{2033}": 2, "\u{2034}": 3]

    /// Environments the pinned typesetter rejects, mapped onto the closest one
    /// it renders. `aligned` is the workhorse: it accepts `&` and `\\` rows,
    /// which is all the rejected alignment environments need.
    private static let environments: [String: String] = [
        "align": "aligned",
        "align*": "aligned",
        "aligned*": "aligned",
        "alignat": "aligned",
        "alignat*": "aligned",
        "gather*": "aligned",
        "equation": "aligned",
        "equation*": "aligned",
        "array": "matrix",
        "rcases": "cases",
    ]

    /// Environments whose `\begin` carries an extra brace argument that the
    /// replacement does not take: `array`'s column spec, `alignat`'s count.
    private static let environmentsWithArgument: Set<String> = [
        "array", "alignat", "alignat*",
    ]

    private static func rewrite(
        _ chars: [Character],
        depth: Int,
        overflowed: inout Bool
    ) -> String {
        guard depth <= maximumDepth else {
            overflowed = true
            return ""
        }
        var out = ""
        var index = 0
        while index < chars.count {
            guard chars[index] == "\\", index + 1 < chars.count else {
                index = appendLiteral(
                    chars,
                    at: index,
                    depth: depth,
                    overflowed: &overflowed,
                    out: &out)
                continue
            }
            let command = name(chars, after: index)
            if command.name == "begin" || command.name == "end" {
                if let rewritten = environment(chars, command: command, out: &out) {
                    index = rewritten
                    continue
                }
            }
            guard let rule = rules[command.name] else {
                out.append(contentsOf: chars[index..<command.end])
                index = command.end
                continue
            }
            index = apply(
                rule,
                chars: chars,
                command: command,
                start: index,
                depth: depth,
                overflowed: &overflowed,
                out: &out)
        }
        return out
    }

    private static func appendLiteral(
        _ chars: [Character],
        at index: Int,
        depth: Int,
        overflowed: inout Bool,
        out: inout String
    ) -> Int {
        let character = chars[index]
        if removed.contains(character) { return index + 1 }
        if let space = spacing[character] {
            out += space
            return index + 1
        }
        if primes[character] != nil {
            var count = 0
            var cursor = index
            while cursor < chars.count, let width = primes[chars[cursor]] {
                count += width
                cursor += 1
            }
            out += "^{" + String(repeating: #"\prime"#, count: count) + "}"
            return cursor
        }
        if character == "\u{221A}" {
            return appendRadical(
                chars,
                at: index,
                depth: depth,
                overflowed: &overflowed,
                out: &out)
        }
        return appendSymbol(chars, at: index, out: &out)
    }

    /// `\u{221A}` is a character, not a command: the build has no atom for it,
    /// so `\u{221A}2` typeset pixel-identical to `2`. What follows it is the
    /// radicand — a group, a parenthesised expression, a run of digits, one
    /// letter, or one command.
    private static func appendRadical(
        _ chars: [Character],
        at index: Int,
        depth: Int,
        overflowed: inout Bool,
        out: inout String
    ) -> Int {
        var cursor = index + 1
        while cursor < chars.count, chars[cursor] == " " { cursor += 1 }
        guard cursor < chars.count else {
            out += #"\sqrt{}"#
            return cursor
        }
        if chars[cursor] == "{", let group = arguments(chars, from: cursor, count: 1) {
            out += #"\sqrt{"# + rewrite(group.values[0], depth: depth + 1, overflowed: &overflowed) + "}"
            return group.end
        }
        if chars[cursor] == "(", let close = closingParenthesis(chars, from: cursor) {
            let inner = Array(chars[(cursor + 1)..<close])
            out += #"\sqrt{"# + rewrite(inner, depth: depth + 1, overflowed: &overflowed) + "}"
            return close + 1
        }
        if chars[cursor].isNumber {
            var end = cursor
            while end < chars.count, chars[end].isNumber || chars[end] == "." { end += 1 }
            out += #"\sqrt{"# + String(chars[cursor..<end]) + "}"
            return end
        }
        if chars[cursor].isLetter {
            out += #"\sqrt{"# + String(chars[cursor]) + "}"
            return cursor + 1
        }
        if chars[cursor] == "\\" {
            let command = name(chars, after: cursor)
            out += #"\sqrt{"# + String(chars[cursor..<command.end]) + "}"
            return command.end
        }
        out += #"\sqrt{}"#
        return cursor
    }

    private static func closingParenthesis(_ chars: [Character], from start: Int) -> Int? {
        var depth = 0
        var index = start
        while index < chars.count {
            if chars[index] == "\\" {
                index += 2
                continue
            }
            if chars[index] == "(" { depth += 1 }
            if chars[index] == ")" {
                depth -= 1
                if depth == 0 { return index }
            }
            index += 1
        }
        return nil
    }

    private static func appendSymbol(
        _ chars: [Character],
        at index: Int,
        out: inout String
    ) -> Int {
        guard let replacement = symbols[chars[index]] else {
            out.append(chars[index])
            return index + 1
        }
        out += replacement
        // A command may not run into the letter after it: `α` before `b` would
        // otherwise spell the unknown command `\alphab`.
        if replacement.last?.isLetter == true,
           index + 1 < chars.count, chars[index + 1].isLetter {
            out.append(" ")
        }
        return index + 1
    }

    private struct Command {
        let name: String
        /// Index just past the command token.
        let end: Int
    }

    private static func name(_ chars: [Character], after backslash: Int) -> Command {
        var index = backslash + 1
        guard index < chars.count else { return Command(name: "", end: index) }
        guard chars[index].isLetter else {
            return Command(name: String(chars[index]), end: index + 1)
        }
        while index < chars.count, chars[index].isLetter { index += 1 }
        var text = String(chars[(backslash + 1)..<index])
        if index < chars.count, chars[index] == "*" {
            text += "*"
            index += 1
        }
        return Command(name: text, end: index)
    }

    private static func apply(
        _ rule: Rule,
        chars: [Character],
        command: Command,
        start: Int,
        depth: Int,
        overflowed: inout Bool,
        out: inout String
    ) -> Int {
        switch rule {
        case .token(let replacement):
            out += replacement
            return command.end
        case .keepArgument(let keep, let count):
            guard let groups = arguments(chars, from: command.end, count: count) else {
                return verbatim(chars, command: command, start: start, out: &out)
            }
            out += rewrite(groups.values[keep], depth: depth + 1, overflowed: &overflowed)
            return groups.end
        case .replaceArguments(let replacement, let count):
            guard let groups = arguments(chars, from: command.end, count: count) else {
                return verbatim(chars, command: command, start: start, out: &out)
            }
            out += replacement
            return groups.end
        case .keepArgumentDroppingScript:
            guard let groups = arguments(chars, from: command.end, count: 1) else {
                return verbatim(chars, command: command, start: start, out: &out)
            }
            out += rewrite(groups.values[0], depth: depth + 1, overflowed: &overflowed)
            return skippingScript(chars, from: groups.end)
        case .script(let operatorCharacter):
            guard let groups = arguments(chars, from: command.end, count: 2) else {
                return verbatim(chars, command: command, start: start, out: &out)
            }
            let annotation = rewrite(
                groups.values[0], depth: depth + 1, overflowed: &overflowed)
            let base = rewrite(groups.values[1], depth: depth + 1, overflowed: &overflowed)
            guard isSingleToken(annotation) else {
                out += "{" + base + "}"
                return groups.end
            }
            out += "{" + base + "}" + operatorCharacter + "{" + annotation + "}"
            return groups.end
        }
    }

    private static func verbatim(
        _ chars: [Character],
        command: Command,
        start: Int,
        out: inout String
    ) -> Int {
        out.append(contentsOf: chars[start..<command.end])
        return command.end
    }

    private static func environment(
        _ chars: [Character],
        command: Command,
        out: inout String
    ) -> Int? {
        guard let groups = arguments(chars, from: command.end, count: 1) else { return nil }
        let raw = String(groups.values[0]).trimmingCharacters(in: .whitespaces)
        guard let mapped = environments[raw] else { return nil }
        out += "\\" + command.name + "{" + mapped + "}"
        guard command.name == "begin", environmentsWithArgument.contains(raw) else {
            return groups.end
        }
        guard let spec = arguments(chars, from: groups.end, count: 1) else { return groups.end }
        return spec.end
    }

    private struct Arguments {
        let values: [[Character]]
        /// Index just past the last consumed group.
        let end: Int
    }

    private static func arguments(
        _ chars: [Character],
        from start: Int,
        count: Int
    ) -> Arguments? {
        var values: [[Character]] = []
        var index = start
        for _ in 0..<count {
            while index < chars.count, chars[index] == " " { index += 1 }
            guard index < chars.count, chars[index] == "{" else { return nil }
            var depth = 0
            var cursor = index
            var closing: Int?
            while cursor < chars.count {
                if chars[cursor] == "\\" {
                    cursor += 2
                    continue
                }
                if chars[cursor] == "{" { depth += 1 }
                if chars[cursor] == "}" {
                    depth -= 1
                    if depth == 0 {
                        closing = cursor
                        break
                    }
                }
                cursor += 1
            }
            guard let closing else { return nil }
            values.append(Array(chars[(index + 1)..<closing]))
            index = closing + 1
        }
        return Arguments(values: values, end: index)
    }

    /// Drops the `_{...}` or `^{...}` annotation that follows a brace command
    /// whose decoration was removed, so the annotation does not reattach to
    /// the surviving base and change what the equation says.
    private static func skippingScript(_ chars: [Character], from start: Int) -> Int {
        guard start < chars.count, chars[start] == "_" || chars[start] == "^" else {
            return start
        }
        if let group = arguments(chars, from: start + 1, count: 1) { return group.end }
        let next = start + 1
        guard next < chars.count else { return start }
        guard chars[next] == "\\" else { return next + 1 }
        // The unbraced form `_\text{total}` is one script token: the command
        // and the group it takes. Dropping only the command left the group
        // behind, so `\underbrace{a+b}_\text{total}` became `a+b{total}` and
        // typeset as "a+btotal". A group separated by a space is not the
        // command's argument and stays part of the equation.
        var end = name(chars, after: next).end
        while end < chars.count, chars[end] == "{",
              let group = arguments(chars, from: end, count: 1) {
            end = group.end
        }
        return end
    }

    private static func isSingleToken(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        if trimmed.count == 1 { return true }
        guard trimmed.hasPrefix("\\") else { return false }
        let body = trimmed.dropFirst()
        return !body.isEmpty && body.allSatisfy { $0.isLetter }
    }
}
