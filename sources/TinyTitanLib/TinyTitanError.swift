// The typed errors the embedded engine reports.
//
// Part of the supported facade (phase A1 of `docs/plan-embedded-library.md`,
// §4). A caller fixing a path and a caller waiting for support need to be told
// apart, so the cases are specific rather than one string.
//
// The loader's own `ModelError` is `package` rather than internal, so this
// target can classify it instead of rethrowing an opaque `Error` — which is
// what a load failure used to be. The two cases that carry free text carry the
// loader's wording verbatim, because inventing a digest-shaped value for a
// failure the loader describes in words would be a worse lie than prose.
import Foundation
import TinyTitanCore

public enum TinyTitanError: Error, Sendable {
    /// The directory does not exist.
    case modelNotFound(URL)
    /// The directory is there but is not a readable TinyTitan install.
    case notAnInstall(URL)
    /// The manifest declares a family this build does not implement.
    case unsupportedFamily(family: String)
    /// The install's format is not one this build reads — a manifest whose
    /// magic is neither `SSDAI` nor the legacy `GTURBO`, an unknown version, or
    /// a flag this build does not know. `detail` is the loader's wording.
    case unsupportedFormat(detail: String)
    /// The install's bytes are not what its manifest and receipt record.
    /// `path` is what the loader was reading when it found out.
    case integrityFailure(path: String, detail: String)
    /// A tool the caller offered is not one the engine can render: a name that
    /// is not `[A-Za-z0-9_-]{1,64}`, parameters that are not an object schema,
    /// or parameters that are not JSON at all.
    case invalidToolDefinition(tool: String, detail: String)
    /// The prompt plus the requested completion do not fit the context window.
    case contextWindowExceeded(prompt: Int, window: Int)
    /// No usable Metal device: none on the machine, or the caller's device
    /// refused a command queue.
    case metalUnavailable(reason: String)
    /// The caller cancelled the generation.
    case cancelled
    /// The engine this session belonged to was unloaded.
    case engineShutDown
}

extension TinyTitanError: CustomStringConvertible {
    /// Human-readable, so a front end that prints `"\(error)"` reports
    /// something a person can act on rather than a reflected enum.
    ///
    /// `modelNotFound` deliberately keeps the tokenizer loader's wording
    /// ("model directory not found: <path>", path standardized), which the
    /// CLI's own test pins and which is what a user who mistyped `--model`
    /// needs to read.
    public var description: String {
        switch self {
        case .modelNotFound(let directory):
            return "model directory not found: \(directory.standardizedFileURL.path)"
        case .notAnInstall(let directory):
            return "not a TinyTitan install: \(directory.standardizedFileURL.path)"
        case .unsupportedFamily(let family):
            return "unsupported model family: \(family)"
        case .unsupportedFormat(let detail):
            return "unsupported model format: \(detail)"
        case .integrityFailure(let path, let detail):
            return "integrity failure for \(path): \(detail)"
        case .contextWindowExceeded(let prompt, let window):
            return "context window exceeded: prompt \(prompt) tokens, window \(window)"
        case .invalidToolDefinition(let tool, let detail):
            return "invalid tool \"\(tool)\": \(detail)"
        case .metalUnavailable(let reason):
            return "no usable Metal device: \(reason)"
        case .cancelled:
            return "cancelled"
        case .engineShutDown:
            return "the engine was unloaded"
        }
    }
}
