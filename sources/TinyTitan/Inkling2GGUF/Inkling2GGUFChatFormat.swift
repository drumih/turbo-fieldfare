import TinyTitanCore

/// Inkling 2's chat format now lives in TinyTitanCore as `InklingChatTemplate`,
/// where the shared session's Inkling tokenizer dialect renders with it (TT-122);
/// this route keeps its old name for it until the route is retired (TT-127).
package typealias Inkling2GGUFChatFormat = InklingChatTemplate
