import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite struct DiskSpaceCheckerTests {
    @Test func assessmentDoesNotCreateMissingTargetAndUsesExistingAncestor() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbofieldfare-space-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("nested/model.gturbo", isDirectory: true)

        let result = try DiskSpaceChecker.assess(path: target.path, bytes: 100, reserveBytes: 20)

        #expect(result.path == root.path)
        #expect(result.requiredBytes == 120)
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("nested").path))
    }

    @Test func authoritativeCheckUsesSameRequirement() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbofieldfare-space-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("model.gturbo", isDirectory: true)

        let assessed = try DiskSpaceChecker.assess(path: target.path, bytes: 100, reserveBytes: 20)
        let required = try DiskSpaceChecker.requireAvailable(path: target.path,
                                                             bytes: 100,
                                                             reserveBytes: 20)
        #expect(assessed.requiredBytes == required.requiredBytes)
    }

    @Test func insufficientCheckReportsRequiredAndAvailableBytes() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbofieldfare-space-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("model.gturbo", isDirectory: true)
        let required = UInt64.max

        #expect {
            _ = try DiskSpaceChecker.requireAvailable(path: target.path,
                                                      bytes: required,
                                                      reserveBytes: 0)
        } throws: { error in
            guard case RepackError.diskSpaceInsufficient(_, let reportedRequired, let actual) = error else {
                return false
            }
            return reportedRequired == required && actual < required
        }
    }
}
