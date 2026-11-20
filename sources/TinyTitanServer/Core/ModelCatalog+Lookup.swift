import Foundation
import TinyTitanCore
import TinyTitanLib

// The catalog's lookups: a base id, an engine alias of one (`<id>@cpu` /
// `@gpu`), or the path of a directory the catalog holds. Split out of
// `ModelCatalog.swift` (2026-11-20) under the 500-line rule when the engine
// alias resolution pushed that file over; pure code motion.
extension ModelCatalog {
    // MARK: - Lookup

    public func entry(id: String) -> Entry? {
        entries.first { $0.id == id }
    }

    /// `"<id>@cpu"` / `"<id>@gpu"` split into the base id and the engine.
    static func engineAlias(_ value: String) -> (base: String, backend: ModelBackend)? {
        for backend in [ModelBackend.cpu, .gpu] {
            let suffix = "@\(backend.rawValue)"
            if value.hasSuffix(suffix), value.count > suffix.count {
                return (String(value.dropLast(suffix.count)), backend)
            }
        }
        return nil
    }

    /// The entry `--model` names: a catalog id, an engine alias of one, or the
    /// path of a directory the catalog holds.
    ///
    /// The alias has to resolve here, not only in `ModelRouter`: the launcher's
    /// `--engine cpu` starts the server with `--model <id>@cpu`, and before this
    /// the startup lookup refused it with "not a model the catalog can serve"
    /// while the same spelling worked as a request field.
    public func entry(idOrPath value: String) -> Entry? {
        if let byID = entry(id: value) { return byID }
        if let alias = Self.engineAlias(value),
            let base = entry(id: alias.base),
            base.engines.contains(alias.backend)
        {
            // Naming the engine an install already defaults to is the install
            // itself: the router's initial lookup only holds listed ids, and a
            // same-engine alias is deliberately not listed. Naming the *other*
            // engine returns that alias, which is listed.
            return alias.backend == base.backend
                ? base
                : base.served(by: alias.backend, id: value)
        }
        let path = URL(fileURLWithPath: value).standardizedFileURL.resolvingSymlinksInPath().path
        return entries.first { $0.path.resolvingSymlinksInPath().path == path }
    }
}
