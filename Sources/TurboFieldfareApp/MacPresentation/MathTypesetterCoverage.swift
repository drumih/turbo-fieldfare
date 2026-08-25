import Foundation

/// What the pinned typesetter can draw, character by character.
///
/// `MTMathListBuilder` asks `MTMathAtomFactory.atom(forCharacter:)` for every
/// literal character and, in math mode, *skips* the ones it has no atom for —
/// no error, no diagnostic, nothing missing from the returned image except the
/// character itself. So `√2` typeset pixel-identical to `2` and `f′(x)` to
/// `f(x)`, and the reader had no way to know an operator had gone.
///
/// This mirrors that function's own order: the Cyrillic range and the accented
/// table first, then everything outside printable ASCII refused, then the set
/// LaTeX reserves, then the ranges each remaining character falls in. The order
/// is why a base character carrying a combining mark is drawn — its string
/// sorts inside the range its base falls in — and the empirical probe measures
/// every scalar of this against the build.
enum PinnedMathCoverage {
    static func renders(_ character: Character) -> Bool {
        let text = String(character)
        if text >= "\u{0410}", text <= "\u{044F}" { return true }
        if accented.contains(character) { return true }
        guard let first = character.unicodeScalars.first,
              first.value >= 0x21, first.value <= 0x7E else {
            return false
        }
        guard !reserved.contains(text) else { return false }
        if singles.contains(text) { return true }
        return ranges.contains { text >= $0.0 && text <= $0.1 }
    }

    /// The first character in `latex` the pinned build would drop, or nil.
    ///
    /// Command names are skipped, because the letters after a backslash are a
    /// command rather than literal text, and so are `\text{...}` groups, which
    /// is the one place the builder accepts anything at all.
    static func firstDroppedCharacter(in latex: String) -> Character? {
        let characters = Array(latex)
        var index = 0
        while index < characters.count {
            let character = characters[index]
            guard character != "\\" else {
                index = skippingCommand(characters, from: index)
                continue
            }
            if structural.contains(character) {
                index += 1
                continue
            }
            guard renders(character) else { return character }
            index += 1
        }
        return nil
    }

    private static func skippingCommand(_ characters: [Character], from start: Int) -> Int {
        var index = start + 1
        guard index < characters.count else { return index }
        guard characters[index].isLetter else { return index + 1 }
        var name = ""
        while index < characters.count, characters[index].isLetter {
            name.append(characters[index])
            index += 1
        }
        // `spacesAllowed` is set for `\text` alone in the pinned builder, and
        // that is the only mode in which an unrecognised character survives.
        guard name == "text", index < characters.count, characters[index] == "{" else {
            return index
        }
        var depth = 0
        while index < characters.count {
            if characters[index] == "\\" {
                index += 2
                continue
            }
            if characters[index] == "{" { depth += 1 }
            if characters[index] == "}" {
                depth -= 1
                if depth == 0 { return index + 1 }
            }
            index += 1
        }
        return index
    }

    /// Characters the builder consumes as structure before it ever asks for an
    /// atom: scripts, groups, the alignment separator, the escape itself, and
    /// ASCII whitespace, which math mode drops on purpose — the spacing comes
    /// from `\ ` and `\,`. A non-breaking or thin space is not in here: those
    /// the builder drops without meaning to.
    ///
    /// The empirical probe skips these, because what they draw is a script or
    /// a group rather than the character itself.
    static let structural: Set<Character> = ["^", "_", "{", "}", "&", "\\", " ", "\t", "\n", "\r"]

    /// What `atom(forCharacter:)` returns nil for by name. `'` is not here: the
    /// pinned build routes it through the accented table and draws it.
    private static let reserved: Set<String> = ["$", "%", "#", "&", "~", "^", "_", "{", "}", "\\"]

    private static let singles: Set<String> = [
        "(", "[", ")", "]", "!", "?", ",", ";", "=", ">", "<", ":",
        "-", "+", "*", ".", "\"", "/", "@", "`", "|",
    ]

    private static let ranges: [(String, String)] = [("0", "9"), ("a", "z"), ("A", "Z")]

    /// `MTMathAtomFactory.supportedAccentedCharacters`, which is consulted
    /// before the ASCII rules and so also carries `'`.
    private static let accented: Set<Character> = [
        "á", "é", "í", "ó", "ú", "ý",
        "à", "è", "ì", "ò", "ù",
        "â", "ê", "î", "ĵ", "ô", "û",
        "ä", "ë", "ï", "ö", "ü", "ÿ",
        "ã", "ñ", "õ",
        "ç", "ø", "å", "æ", "œ", "ß", "'",
        "Á", "É", "Í", "Ó", "Ú", "Ý",
        "À", "È", "Ì", "Ò", "Ù",
        "Â", "Ê", "Î", "Ô", "Û",
        "Ä", "Ë", "Ï", "Ö", "Ü",
        "Ã", "Ñ", "Õ",
        "Ç", "Ø", "Å", "Æ", "Œ",
    ]
}
