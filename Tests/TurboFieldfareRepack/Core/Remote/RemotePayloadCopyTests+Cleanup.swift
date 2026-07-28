import Foundation
import Testing

@testable import TurboFieldfareRepackCore

extension RemotePayloadCopyTests {
  @Test func discardRemovesOnlyOwnedResumeFiles() throws {
    let root = tmpDirForRemote("discard")
    let output = (root as NSString).appendingPathComponent("model.gturbo")
    let paths = try RemoteInstallPaths(outputDirectory: output)
    let unrelated = (root as NSString).appendingPathComponent("keep.txt")
    defer { cleanUpRemote([root]) }

    try Posix.mkdirP(paths.partialDirectory)
    try Data("{}".utf8).write(to: URL(fileURLWithPath: paths.checkpointFile))
    try Data("keep".utf8).write(to: URL(fileURLWithPath: unrelated))

    try RemoteStreamingRepacker.discardPartial(outputDirectory: output)

    #expect(!FileManager.default.fileExists(atPath: paths.partialDirectory))
    #expect(!FileManager.default.fileExists(atPath: paths.checkpointFile))
    #expect(FileManager.default.fileExists(atPath: unrelated))
  }
}
