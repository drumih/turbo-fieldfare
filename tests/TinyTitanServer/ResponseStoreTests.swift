import Foundation
import Testing

@testable import TinyTitanLib
@testable import TinyTitanServerCore
@testable import TinyTitanCore

/// The stored-response store behind `previous_response_id`, `GET` and `DELETE`.
/// It is bounded and in memory, and the doc comment states the eviction rule:
/// "the oldest entry goes when the cap is reached".
@Suite struct ResponseStoreTests {
    private func entry(_ text: String) -> ResponseStore.Entry {
        ResponseStore.Entry(
            responseJSON: Data(text.utf8),
            inputItems: [], outputItems: [], created: Date())
    }

    @Test func theCapDropsTheOldestEntry() {
        let store = ResponseStore(capacity: 2)
        store.put(id: "a", entry: entry("a"))
        store.put(id: "b", entry: entry("b"))
        store.put(id: "c", entry: entry("c"))

        #expect(store.get("a") == nil)
        #expect(store.get("b") != nil)
        #expect(store.get("c") != nil)
        #expect(store.count == 2)
    }

    /// A re-put is a write: the response just stored is the newest, so it must
    /// not be the one the cap reaches first. Keeping its original position made
    /// the store evict what the caller had just written and keep the entry it
    /// replaced.
    @Test func aRePutRefreshesTheEvictionOrder() {
        let store = ResponseStore(capacity: 2)
        store.put(id: "a", entry: entry("a1"))
        store.put(id: "b", entry: entry("b1"))
        store.put(id: "a", entry: entry("a2"))
        store.put(id: "c", entry: entry("c"))

        #expect(
            store.get("a")?.responseJSON == Data("a2".utf8),
            "the re-put did not replace the entry")
        #expect(store.get("c") != nil)
        #expect(
            store.get("b") == nil,
            "the cap dropped the re-stored entry instead of the oldest one")
        #expect(store.count == 2)
    }

    @Test func deleteRemovesTheEntryAndItsOrderSlot() {
        let store = ResponseStore(capacity: 3)
        store.put(id: "a", entry: entry("a"))
        store.put(id: "b", entry: entry("b"))
        #expect(store.delete("a"))
        #expect(!store.delete("a"))
        store.put(id: "c", entry: entry("c"))
        store.put(id: "d", entry: entry("d"))
        // Capacity 3 with a removed: b, c, d all fit.
        #expect(store.count == 3)
        #expect(store.get("b") != nil)
    }
}
