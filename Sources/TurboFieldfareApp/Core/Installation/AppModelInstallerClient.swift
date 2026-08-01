import Foundation

public protocol AppModelInstallerClient: Sendable {
    var descriptor: AppModelInstallDescriptor { get }
    func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement
    func checkInstallRequirement(entry: ModelCatalogEntry,
                                 outputDirectory: URL) throws -> AppModelInstallRequirement
    func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error>
    func install(entry: ModelCatalogEntry,
                 outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error>
    func discardPartialInstall(outputDirectory: URL) async throws
    func cancel()
}

/// Default forwarding so conformers that predate multi-model support — the test
/// doubles in particular — keep compiling without each having to reimplement
/// entry-aware installs they do not model.
public extension AppModelInstallerClient {
    func checkInstallRequirement(entry: ModelCatalogEntry,
                                 outputDirectory: URL) throws -> AppModelInstallRequirement {
        try checkInstallRequirement(outputDirectory: outputDirectory)
    }

    func install(entry: ModelCatalogEntry,
                 outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        installDefaultModel(outputDirectory: outputDirectory)
    }
}
