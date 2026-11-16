import Foundation

// The number and string arms of the JSON grammar state machine: number
// continuation, key and string reading, escapes and unicode escapes.
//
// Split out of `JSONGrammar.swift` (2026-09-28) under the 500-line-per-file
// rule (Task 8 of the cleanup runbook) as pure code motion. The six arms
// widened from `private` to internal because `step` stays behind.
extension JSONGrammar {

    mutating func startNumber(_ byte: UInt8) -> Bool {
        switch byte {
        case 0x2D:
            state = .number(.minus)
        case 0x30:
            state = .number(.zero)
        default:
            guard JSONGrammar.isDigit(byte) else { return false }
            state = .number(.integer)
        }
        return true
    }

    mutating func continueNumber(_ byte: UInt8, number: NumberState) -> Bool {
        let full = !node.forbidsFraction
        switch number {
        case .minus:
            guard JSONGrammar.isDigit(byte) else { return false }
            state = .number(byte == 0x30 ? .zero : .integer)
            return true
        case .zero:
            // A leading zero may not be followed by another digit.
            if byte == 0x2E, full {
                state = .number(.fractionStart)
                return true
            }
            if byte == 0x65 || byte == 0x45, full {
                state = .number(.exponent)
                return true
            }
            return finishValue(consuming: byte)
        case .integer:
            if JSONGrammar.isDigit(byte) { return true }
            if byte == 0x2E, full {
                state = .number(.fractionStart)
                return true
            }
            if byte == 0x65 || byte == 0x45, full {
                state = .number(.exponent)
                return true
            }
            return finishValue(consuming: byte)
        case .fractionStart:
            guard JSONGrammar.isDigit(byte) else { return false }
            state = .number(.fraction)
            return true
        case .fraction:
            if JSONGrammar.isDigit(byte) { return true }
            if byte == 0x65 || byte == 0x45 {
                state = .number(.exponent)
                return true
            }
            return finishValue(consuming: byte)
        case .exponent:
            if byte == 0x2B || byte == 0x2D {
                state = .number(.exponentSign)
                return true
            }
            if JSONGrammar.isDigit(byte) {
                state = .number(.exponentDigits)
                return true
            }
            return false
        case .exponentSign:
            guard JSONGrammar.isDigit(byte) else { return false }
            state = .number(.exponentDigits)
            return true
        case .exponentDigits:
            if JSONGrammar.isDigit(byte) { return true }
            return finishValue(consuming: byte)
        }
    }

    // MARK: - Strings

    mutating func startKey(_ byte: UInt8) -> Bool {
        if case .object(let frame) = stack.last, !frame.additional {
            let remaining = frame.properties.keys.filter { !frame.seen.contains($0) }.sorted()
            guard !remaining.isEmpty else { return false }
            return beginEnumeration(remaining.map { "\"\($0)\"" }, role: .key, first: byte)
        }
        guard byte == 0x22 else { return false }
        keyBytes.removeAll(keepingCapacity: true)
        state = .string(.key)
        return true
    }

    mutating func continueString(_ byte: UInt8, role: StringRole) -> Bool {
        if byte == 0x5C {
            state = .escape(role)
            return true
        }
        // A raw control character is not legal inside a JSON string.
        guard byte >= 0x20 else { return false }
        if byte == 0x22 {
            guard role == .key else { return finishValue() }
            return finishKey(keyBytes.lossyUTF8String)
        }
        if role == .key { keyBytes.append(byte) }
        return true
    }

    mutating func continueEscape(_ byte: UInt8, role: StringRole) -> Bool {
        if byte == 0x75 {
            unicodeDigits.removeAll(keepingCapacity: true)
            state = .unicode(4, role)
            return true
        }
        let decoded: UInt8
        switch byte {
        case 0x22: decoded = 0x22
        case 0x5C: decoded = 0x5C
        case 0x2F: decoded = 0x2F
        case 0x62: decoded = 0x08
        case 0x66: decoded = 0x0C
        case 0x6E: decoded = 0x0A
        case 0x72: decoded = 0x0D
        case 0x74: decoded = 0x09
        default: return false
        }
        if role == .key { keyBytes.append(decoded) }
        state = .string(role)
        return true
    }

    mutating func continueUnicode(_ byte: UInt8, remaining: Int, role: StringRole) -> Bool {
        guard JSONGrammar.isHexDigit(byte) else { return false }
        unicodeDigits.append(byte)
        if remaining > 1 {
            state = .unicode(remaining - 1, role)
            return true
        }
        if role == .key, let scalar = UInt32(unicodeDigits.lossyUTF8String, radix: 16),
            let unicode = Unicode.Scalar(scalar)
        {
            keyBytes.append(contentsOf: Array(String(Character(unicode)).utf8))
        }
        state = .string(role)
        return true
    }
}
