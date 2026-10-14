import Foundation
import TurboFieldfare

/// One app-data store with the identity of the currently selected model.
///
/// Async store work captures the generation and must prove it is still current
/// before changing UI state. A path switch can otherwise let a slow list or
/// open from the old root overwrite the new model's history.
struct ConversationStoreBinding: Sendable {
    let modelDirectory: URL
    let rootURL: URL
    let store: ConversationStore
    var identity: ConversationIdentity?
    let generation: UInt64

    init(modelDirectory: URL, appDataDirectory: URL? = nil, store: ConversationStore? = nil, identity: ConversationIdentity?, generation: UInt64,
         storeProvider: @Sendable (URL) -> ConversationStore = {
             ConversationStore(rootURL: $0)
         }) {
        let modelDirectory = modelDirectory.standardizedFileURL
        self.modelDirectory = modelDirectory
        self.rootURL = appDataDirectory.map { ConversationStoreLocation.url(forAppDataDirectory: $0) }
            ?? ConversationStoreLocation.url(forModelDirectory: modelDirectory)
            .standardizedFileURL
        self.store = store ?? storeProvider(rootURL)
        self.identity = identity
        self.generation = generation
    }
}
