import Foundation
import Testing

@testable import TurboFieldfareAppCore

@Suite struct FakeInferenceClientTests {

  @MainActor
  @Test func installedModelSupportsLoadAndGenerationWithTestClient() async throws {
    let directory = try makeCompleteModelInstall("fake-runtime")
    defer { try? FileManager.default.removeItem(at: directory) }
    let model = AppModel(
      modelDirectory: directory,
      client: FakeInferenceClient(eventDelay: .zero),
      installer: MockModelInstallerClient())

    #expect(model.isModelInstalled)
    #expect(model.loadState == .notLoaded)
    model.loadModel()
    try await waitForFakeClient { model.loadState.isReady }
    model.promptText = "Hello from the test"
    model.run()
    try await waitForFakeClient { model.diagnostics != nil && !model.isRunning }

    #expect(model.isModelInstalled)
    #expect(model.outputText.contains("Simulated response"))
    #expect(model.outputText.contains("Hello from the test"))
    #expect(model.diagnostics?.generatedTokens ?? 0 > 0)
  }

  @Test func fakeInferenceCancellationEmitsCancelledEvent() async throws {
    let client = FakeInferenceClient(eventDelay: .milliseconds(5))
    let request = AppGenerationRequest(
      modelDirectory: URL(fileURLWithPath: "/missing/fake-model"),
      prompt: "Cancel this simulated response")
    try await client.ensureLoaded(
      modelDirectory: request.modelDirectory,
      maxContextTokens: request.maxContextTokens,
      options: request.runtimeOptions,
      forceLogitsHead: !request.isPureGreedy,
      onState: { _ in })
    var sawCancellation = false

    do {
      for try await event in client.generate(request) {
        if case .token = event {
          client.cancel()
        } else if case .cancelled = event {
          sawCancellation = true
        }
      }
    } catch AppInferenceError.cancelled {
    }

    #expect(sawCancellation)
  }

  @Test func fakeInferenceRejectsConcurrentGeneration() async throws {
    let client = FakeInferenceClient(eventDelay: .milliseconds(20))
    let request = AppGenerationRequest(
      modelDirectory: URL(fileURLWithPath: "/missing/fake-model"),
      prompt: "First")
    try await client.ensureLoaded(
      modelDirectory: request.modelDirectory,
      maxContextTokens: request.maxContextTokens,
      options: request.runtimeOptions,
      forceLogitsHead: !request.isPureGreedy,
      onState: { _ in })
    let first = client.generate(request)
    let second = client.generate(request)
    var failure: AppInferenceError?
    do {
      for try await event in second {
        if case .failed(let error, _) = event { failure = error }
      }
    } catch let error as AppInferenceError {
      failure = failure ?? error
    }
    #expect(failure == .generationInFlight)
    client.cancel()
    _ = first
  }

  @MainActor
  private func waitForFakeClient(_ predicate: @escaping @MainActor () -> Bool) async throws {
    for _ in 0..<200 {
      if predicate() { return }
      try await Task.sleep(for: .milliseconds(5))
    }
    Issue.record("timed out waiting for fake client state")
  }

}
