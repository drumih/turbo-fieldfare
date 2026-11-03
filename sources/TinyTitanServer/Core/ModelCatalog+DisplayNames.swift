import Foundation

/// The display-name table, split out of `ModelCatalog.swift` (2026-10-25) as
/// pure code motion under the 500-line-per-file rule.
extension ModelCatalog {
    /// Human names for the installs this project ships. A manifest carries an
    /// id, not a name; an install outside this table is listed by its id.
    static let displayNames: [String: String] = [
        "qwen3.6-35b-a3b": "Qwen 3.6 35B-A3B",
        "ornith-1.5-35b-a3b": "Ornith 1.5 35B-A3B",
        "ornith-1.5-35b-a3b-bf16dense": "Ornith 1.5 35B-A3B (bf16 dense)",
        "qwen-agentworld": "Qwen AgentWorld 35B-A3B",
        "kat-coder-v2.5": "KAT-Coder-V2.5-Dev 35B-A3B",
        "qwen3.8-flash-next": "Qwen 3.8 Flash Next 125B-A6B",
        "qwen4-omni-120b-think": "Qwen 4.0 Flash (Qwen4-Omni-120B-Think)",
        // The dense CPU models. Their snapshots carried these names in
        // `config.json -> display_name`, which only the snapshot probe reads;
        // the install probe has a manifest, and a manifest carries an id, not
        // a name. Without these entries repacking them would silently rename
        // them from "Qwen 3.5 9B" to "qwen3.5-9b" in the app and the server's
        // /v1/models listing.
        "qwen3.5-2b": "Qwen 3.5 2B",
        "qwen3.5-4b": "Qwen 3.5 4B",
        "qwen3.5-9b": "Qwen 3.5 9B",
    ]
}
