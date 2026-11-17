import TinyTitan
import TinyTitanCore

extension CPUEngineFactory {
    static func legacy(snapshot: AffineSnapshot) -> CPUEngineFactory {
        CPUEngineFactory { try CPUQwen35(snapshot: snapshot) }
    }
}
