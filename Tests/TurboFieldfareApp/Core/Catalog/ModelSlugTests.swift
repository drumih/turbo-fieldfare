import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct ModelSlugTests {
    @Test func derivesStableSlugFromRepositoryID() throws {
        let slug = try ModelSlug.make(repoID: "mlx-community/gemma-4-26b-a4b-it-4bit")
        #expect(slug == "mlx-community--gemma-4-26b-a4b-it-4bit")
    }

    @Test func isDeterministic() throws {
        let first = try ModelSlug.make(repoID: "someone/gemma-4-26b-a4b-uncensored")
        let second = try ModelSlug.make(repoID: "someone/gemma-4-26b-a4b-uncensored")
        #expect(first == second)
    }

    @Test func rejectsParentDirectoryComponents() {
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "../etc")
        }
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "owner/..")
        }
    }

    @Test func rejectsAbsolutePaths() {
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "/etc/passwd")
        }
    }

    @Test func rejectsWrongComponentCount() {
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "no-owner")
        }
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "a/b/c")
        }
    }

    @Test func rejectsEmptyAndSeparatorOnlyInput() {
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "")
        }
        #expect(throws: ModelSlug.InvalidRepositoryID.self) {
            try ModelSlug.make(repoID: "owner/")
        }
    }

    @Test func rejectsShellAndPathMetacharacters() {
        for hostile in ["own er/name", "owner/na\\me", "owner/na:me", "owner/na\0me"] {
            #expect(throws: ModelSlug.InvalidRepositoryID.self) {
                try ModelSlug.make(repoID: hostile)
            }
        }
    }

    @Test func slugStaysInsideParentDirectory() throws {
        let parent = URL(fileURLWithPath: "/tmp/models", isDirectory: true)
        let slug = try ModelSlug.make(repoID: "mlx-community/gemma-4-26b-a4b-it-4bit")
        let resolved = parent.appendingPathComponent(slug, isDirectory: true).standardizedFileURL
        #expect(resolved.path.hasPrefix("/tmp/models/"))
    }
}
