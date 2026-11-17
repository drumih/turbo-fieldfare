import TinyTitan

package enum ServerModelIdentity {
    package static func base(manifestModelID: String, family: ModelFamily) -> String {
        var id = manifestModelID
        for suffix in ["-4bit", "-8bit", "-6bit"] where id.hasSuffix(suffix) {
            id = String(id.dropLast(suffix.count))
        }
        guard id == "unknown/snapshot" else { return id }
        switch family {
        case .qwen36, .qwen36MTP: return "qwen3.6-35b-a3b"
        case .qwen38flash, .qwen38flashMTP: return "qwen3.8-flash-next"
        case .qwen4flash, .qwen4flashMTP: return "qwen4-omni-120b-think"
        case .qwen35Dense: return "qwen3.5-dense"
        case .k2Horizon: return "k2-horizon"
        case .laguna: return "laguna-s-2-1"
        }
    }

    package static func apiModelID(
        manifestModelID: String, family: ModelFamily, weightBits: Int
    ) -> String {
        base(manifestModelID: manifestModelID, family: family) + "_\(weightBits)-Bit"
    }
}
