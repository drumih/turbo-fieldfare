import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct ModelRevealPolicyTests {
    @Test func installedModelDirectoryIsSelected() {
        let target = ModelRevealPolicy.target(
            forModelPath: "/Users/dev/checkout/scratch/gemma4.gturbo",
            fileExists: { $0 == "/Users/dev/checkout/scratch/gemma4.gturbo" })
        #expect(target == .selectItem(
            URL(fileURLWithPath: "/Users/dev/checkout/scratch/gemma4.gturbo",
                isDirectory: true)))
    }

    @Test func missingModelFallsBackToItsContainer() {
        let target = ModelRevealPolicy.target(
            forModelPath: "/Users/dev/checkout/scratch/gemma4.gturbo",
            fileExists: { $0 == "/Users/dev/checkout/scratch" })
        #expect(target == .openContainer(
            URL(fileURLWithPath: "/Users/dev/checkout/scratch", isDirectory: true)))
    }

    @Test func missingContainerIsUnavailable() {
        let target = ModelRevealPolicy.target(
            forModelPath: "/Users/dev/checkout/scratch/gemma4.gturbo",
            fileExists: { _ in false })
        #expect(target == .unavailable)
    }

    @Test(arguments: ["", "   "])
    func blankPathIsUnavailable(path: String) {
        #expect(ModelRevealPolicy.target(forModelPath: path,
                                         fileExists: { _ in true }) == .unavailable)
    }

    @Test func pathIsStandardizedBeforeReveal() {
        let target = ModelRevealPolicy.target(
            forModelPath: "/Users/dev/checkout/./scratch/../scratch/gemma4.gturbo",
            fileExists: { $0 == "/Users/dev/checkout/scratch/gemma4.gturbo" })
        #expect(target == .selectItem(
            URL(fileURLWithPath: "/Users/dev/checkout/scratch/gemma4.gturbo",
                isDirectory: true)))
    }

    @Test func rootPathDoesNotLoopIntoItself() {
        #expect(ModelRevealPolicy.target(forModelPath: "/",
                                         fileExists: { _ in false }) == .unavailable)
    }

    @Test func existingDirectoryWithTrailingSpaceKeepsItsExactName() {
        // A folder picked from Finder can end with a space. Trimming it would
        // reveal a different model or its parent instead of the selected item.
        let path = "/Volumes/Models/модель.gturbo "
        #expect(ModelRevealPolicy.target(forModelPath: path,
                                         fileExists: { $0 == path }) == .selectItem(
            URL(fileURLWithPath: path, isDirectory: true)))
    }

    @Test func disconnectedVolumeDoesNotRevealUnrelatedVolumesDirectory() {
        #expect(ModelRevealPolicy.target(
            forModelPath: "/Volumes/Disconnected/gemma4.gturbo",
            fileExists: { $0 == "/Volumes" || $0 == "/" }) == .unavailable)
    }
}
