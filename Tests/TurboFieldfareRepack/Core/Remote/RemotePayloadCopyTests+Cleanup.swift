import Foundation
import Synchronization
import Testing

@testable import TurboFieldfareRepackCore

extension RemotePayloadCopyTests {
  @Test func stalePartialSweepRemovesDeadPidAndPreservesUnrelatedSiblings() async throws {
    let root = tmpDirForRemote("partial-root")
    let output = (root as NSString).appendingPathComponent("out.gturbo")
    let stale = (root as NSString).appendingPathComponent("out.gturbo.partial.999999")
    let unrelatedA = (root as NSString).appendingPathComponent("out.gturbo.partial")
    let unrelatedB = (root as NSString).appendingPathComponent("outX.gturbo.partial.1")
    defer { cleanUpRemote([root]) }
    try Posix.mkdirP(stale)
    try Data("x".utf8).write(
      to: URL(fileURLWithPath: (stale as NSString).appendingPathComponent("payload")))
    try Posix.mkdirP(unrelatedA)
    try Posix.mkdirP(unrelatedB)
    let audit = RepackAudit()

    try RemoteStreamingRepacker.sweepStalePartials(outputDir: output, audit: audit)

    #expect(!FileManager.default.fileExists(atPath: stale))
    #expect(FileManager.default.fileExists(atPath: unrelatedA))
    #expect(FileManager.default.fileExists(atPath: unrelatedB))
    #expect(audit.stalePartialsRemoved == [stale])
  }

  @Test func stalePartialSweepRefusesLivePid() throws {
    let root = tmpDirForRemote("partial-live-root")
    let output = (root as NSString).appendingPathComponent("out.gturbo")
    let live = (root as NSString).appendingPathComponent("out.gturbo.partial.\(getpid())")
    defer { cleanUpRemote([root]) }
    try Posix.mkdirP(live)

    #expect {
      try RemoteStreamingRepacker.sweepStalePartials(outputDir: output)
    } throws: { error in
      guard case RepackError.configurationInvalid(let detail) = error else {
        return false
      }
      return detail.contains("another repack")
    }
    #expect(FileManager.default.fileExists(atPath: live))
  }

}
