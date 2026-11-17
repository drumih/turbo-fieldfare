import TinyTitan

package func defaultPrefillChunkTokens(family: ModelFamily, fallback: Int) -> Int {
    switch family {
    case .qwen36, .qwen38flash, .qwen4flash, .qwen35Dense:
        RuntimeConfiguration.qwenLongPrefillChunkTokens
    default:
        fallback
    }
}
