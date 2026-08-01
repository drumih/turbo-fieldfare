import Foundation
import Testing
import TurboFieldfare

@testable import TurboFieldfareAppCore

@Suite struct AppMemoryBudgetTests {
    private let gigabyte: UInt64 = 1_073_741_824

    @Test func eightGigabyteMachineGetsTwoGigabyteCeiling() {
        #expect(AppMemoryBudget.ceilingBytes(installedRAMBytes: 8 * gigabyte) == 2 * gigabyte)
    }

    @Test func sixteenGigabyteMachineGetsTheFourGigabyteCap() {
        #expect(AppMemoryBudget.ceilingBytes(installedRAMBytes: 16 * gigabyte) == 4 * gigabyte)
    }

    @Test func largeMachineIsStillCappedAtFourGigabytes() {
        #expect(AppMemoryBudget.ceilingBytes(installedRAMBytes: 128 * gigabyte) == 4 * gigabyte)
    }

    @Test func tinyMachineGetsTheFloorNotAProportionalShare() {
        let ceiling = AppMemoryBudget.ceilingBytes(installedRAMBytes: 4 * gigabyte)
        #expect(ceiling == UInt64(1.5 * Double(gigabyte)))
    }

    @Test func optionsBelowTheCeilingAreEnabled() {
        let options = AppContextLengthOption.availableOptions(
            architecture: .gemma4_26B_A4B,
            residentWeightBytes: gigabyte,
            installedRAMBytes: 16 * gigabyte)
        let fourK = options.first { $0.tokens == 4_096 }
        #expect(fourK?.isEnabled == true)
    }

    @Test func optionsAboveTheCeilingAreDisabledNotHidden() {
        let options = AppContextLengthOption.availableOptions(
            architecture: .gemma4_26B_A4B,
            residentWeightBytes: 3 * gigabyte,
            installedRAMBytes: 8 * gigabyte)
        let largest = options.last
        #expect(largest?.isEnabled == false)
        #expect(largest?.disabledReason != nil)
        #expect(options.contains { $0.tokens == 65_536 })
    }

    @Test func kvBytesGrowMonotonicallyWithContext() {
        let options = AppContextLengthOption.availableOptions(
            architecture: .gemma4_26B_A4B,
            residentWeightBytes: gigabyte,
            installedRAMBytes: 32 * gigabyte)
        let sorted = options.sorted { $0.tokens < $1.tokens }
        for (smaller, larger) in zip(sorted, sorted.dropFirst()) {
            #expect(larger.kvBytes > smaller.kvBytes)
        }
    }

    @Test func offersContextBeyondSixtyFourKOnLargeMachines() {
        let options = AppContextLengthOption.availableOptions(
            architecture: .gemma4_26B_A4B,
            residentWeightBytes: gigabyte,
            installedRAMBytes: 32 * gigabyte)
        #expect(options.contains { $0.tokens == 131_072 && $0.isEnabled })
    }
}
