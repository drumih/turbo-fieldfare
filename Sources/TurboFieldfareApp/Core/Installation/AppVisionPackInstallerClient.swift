import Foundation

public protocol AppVisionPackInstallerClient: Sendable {
    var descriptor: AppModelInstallDescriptor { get }
    func checkInstallRequirement(textModelDirectory: URL) throws -> AppModelInstallRequirement
    func install(textModelDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error>
    func preparedInstallIsValid(textModelDirectory: URL) -> Bool
    func activatePreparedInstall(
        textModelDirectory: URL,
        onVerifyProgress: (@Sendable (Double) -> Void)?
    ) async throws -> URL
    func discardPartialInstall(textModelDirectory: URL) async throws
    func removeInstalled(textModelDirectory: URL) async throws
    func cancel()
}
