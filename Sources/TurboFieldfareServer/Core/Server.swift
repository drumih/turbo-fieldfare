import Foundation
import Hummingbird

/// Loads the model, builds the router and runs the HTTP server. The model is
/// fully loaded before the port is bound, so a listening socket always means
/// ready.
public func runServer(args: ServerArgs) async throws {
    let generator = try await RealTokenGenerator.load(modelPath: args.model,
                                                      maxContext: args.maxContext)
    let session = GenerationSession(modelID: args.modelID,
                                    maxContext: args.maxContext,
                                    generator: generator)
    let router = buildServerRouter(session: session)
    let app = Application(
        router: router,
        configuration: ApplicationConfiguration(
            address: .hostname(args.host, port: args.port),
            serverName: "TurboFieldfareServer"))
    if !args.quiet {
        print("TurboFieldfareServer listening on http://\(args.host):\(args.port) (model: \(args.modelID))")
    }
    try await app.runService()
}
