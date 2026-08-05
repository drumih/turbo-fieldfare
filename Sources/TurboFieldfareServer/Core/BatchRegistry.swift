import Foundation

public struct BatchRequest: Sendable {
    public let customID: String
    public let request: ValidatedChatRequest

    public init(customID: String, request: ValidatedChatRequest) {
        self.customID = customID
        self.request = request
    }
}

public actor BatchRegistry {
    public enum Status: String, Codable, Sendable {
        case validating, inProgress = "in_progress", completed, failed, cancelling, cancelled
    }

    public struct Snapshot: Codable, Sendable {
        public let id: String
        public let object = "batch"
        public let status: Status
        public let endpoint: String
        public let inputFileID: String?
        public let completionWindow: String?
        public let createdAt: Int
        public let inProgressAt: Int?
        public let completedAt: Int?
        public let failedAt: Int?
        public let cancellingAt: Int?
        public let cancelledAt: Int?
        public let requestCounts: Counts
        public let outputFileID: String
        public let errorFileID: String?
        public let metadata: [String: String]?

        enum CodingKeys: String, CodingKey {
            case id, object, status, endpoint, metadata
            case inputFileID = "input_file_id"
            case completionWindow = "completion_window"
            case createdAt = "created_at"
            case inProgressAt = "in_progress_at"
            case completedAt = "completed_at"
            case failedAt = "failed_at"
            case cancellingAt = "cancelling_at"
            case cancelledAt = "cancelled_at"
            case requestCounts = "request_counts"
            case outputFileID = "output_file_id"
            case errorFileID = "error_file_id"
        }
    }

    public struct Counts: Codable, Sendable {
        public let total: Int
        public let completed: Int
        public let failed: Int
    }

    public struct List: Codable, Sendable {
        public let object = "list"
        public let data: [Snapshot]
        public let firstID: String?
        public let lastID: String?
        public let hasMore: Bool

        enum CodingKeys: String, CodingKey {
            case object, data
            case firstID = "first_id"
            case lastID = "last_id"
            case hasMore = "has_more"
        }
    }

    private struct Job {
        var status: Status
        let endpoint: String
        let inputFileID: String?
        let completionWindow: String?
        let metadata: [String: String]?
        let createdAt: Int
        let total: Int
        let outputFileID: String
        let outputURL: URL
        var errorFileID: String?
        var errorURL: URL?
        var completed = 0
        var failed = 0
        var task: Task<Void, Never>?
        var inProgressAt: Int?
        var completedAt: Int?
        var failedAt: Int?
        var cancellingAt: Int?
        var cancelledAt: Int?
    }

    private let outputDirectory: URL
    private var jobs: [String: Job] = [:]

    public init(outputDirectory: URL = FileManager.default.temporaryDirectory
        .appendingPathComponent("TurboFieldfare/batches", isDirectory: true)) {
        self.outputDirectory = outputDirectory
    }

    public func create(requests: [BatchRequest],
                       backend: any ServerInferenceBackend,
                       coordinator: ServerCoordinator,
                       modelID: String,
                       inputFileID: String? = nil,
                       completionWindow: String? = nil,
                       metadata: [String: String]? = nil) throws -> Snapshot {
        let id = "batch_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
        let outputFileID = "file_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
        let outputURL = outputDirectory.appendingPathComponent(outputFileID).appendingPathExtension("jsonl")
        try FileManager.default.createDirectory(at: outputDirectory,
                                                withIntermediateDirectories: true)
        try Data().write(to: outputURL, options: .withoutOverwriting)

        let created = Int(Date().timeIntervalSince1970)
        jobs[id] = Job(status: .validating,
                       endpoint: "/v1/chat/completions",
                       inputFileID: inputFileID,
                       completionWindow: completionWindow,
                       metadata: metadata,
                       createdAt: created,
                       total: requests.count,
                       outputFileID: outputFileID,
                       outputURL: outputURL)
        let task = Task { [weak self] in
            guard let self else { return }
            await self.start(id)
            for item in requests {
                if Task.isCancelled {
                    await self.cancelled(id)
                    return
                }
                do {
                    let completion = try await coordinator.run {
                        try await backend.generate(item.request) { _ in }
                    }
                    try await self.succeeded(id,
                                             customID: item.customID,
                                             completion: completion,
                                             modelID: modelID)
                } catch is CancellationError {
                    await self.cancelled(id)
                    return
                } catch {
                    await self.failed(id, customID: item.customID, error: error)
                }
            }
            await self.finish(id)
        }
        jobs[id]?.task = task
        return snapshot(id)!
    }

    public func get(_ id: String) -> Snapshot? { snapshot(id) }

    public func list(limit: Int, after: String?) -> List {
        let ordered = jobs.keys.sorted { lhs, rhs in
            let left = jobs[lhs]!
            let right = jobs[rhs]!
            if left.createdAt != right.createdAt { return left.createdAt > right.createdAt }
            return lhs > rhs
        }
        let start: Int
        if let after, let index = ordered.firstIndex(of: after) {
            start = index + 1
        } else {
            start = 0
        }
        let ids = Array(ordered.dropFirst(start).prefix(limit))
        return List(data: ids.compactMap(snapshot),
                    firstID: ids.first,
                    lastID: ids.last,
                    hasMore: start + ids.count < ordered.count)
    }

    public func cancel(_ id: String) -> Snapshot? {
        guard var job = jobs[id] else { return nil }
        guard job.status == .validating || job.status == .inProgress else { return snapshot(id) }
        job.status = .cancelling
        job.cancellingAt = Int(Date().timeIntervalSince1970)
        job.task?.cancel()
        jobs[id] = job
        return snapshot(id)
    }

    private func start(_ id: String) {
        guard var job = jobs[id], job.status == .validating else { return }
        job.status = .inProgress
        job.inProgressAt = Int(Date().timeIntervalSince1970)
        jobs[id] = job
    }

    private func succeeded(_ id: String,
                           customID: String,
                           completion: ServerCompletion,
                           modelID: String) throws {
        guard var job = jobs[id] else { return }
        try appendSuccess(to: job.outputURL,
                          customID: customID,
                          completion: completion,
                          modelID: modelID)
        job.completed += 1
        jobs[id] = job
    }

    private func failed(_ id: String, customID: String, error: Error) {
        guard var job = jobs[id] else { return }
        do {
            if job.errorFileID == nil {
                let errorFileID = "file_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
                let errorURL = outputDirectory.appendingPathComponent(errorFileID).appendingPathExtension("jsonl")
                try Data().write(to: errorURL, options: .withoutOverwriting)
                job.errorFileID = errorFileID
                job.errorURL = errorURL
            }
            try appendFailure(to: job.errorURL!, customID: customID, error: error)
        } catch {
            // The original inference error remains the result of this item.
        }
        job.failed += 1
        jobs[id] = job
    }

    private func cancelled(_ id: String) {
        guard var job = jobs[id] else { return }
        job.status = .cancelled
        job.cancelledAt = Int(Date().timeIntervalSince1970)
        jobs[id] = job
    }

    private func finish(_ id: String) {
        guard var job = jobs[id], job.status != .cancelled else { return }
        job.status = .completed
        job.completedAt = Int(Date().timeIntervalSince1970)
        jobs[id] = job
    }

    private func snapshot(_ id: String) -> Snapshot? {
        guard let job = jobs[id] else { return nil }
        return Snapshot(id: id,
                        status: job.status,
                        endpoint: job.endpoint,
                        inputFileID: job.inputFileID,
                        completionWindow: job.completionWindow,
                        createdAt: job.createdAt,
                        inProgressAt: job.inProgressAt,
                        completedAt: job.completedAt,
                        failedAt: job.failedAt,
                        cancellingAt: job.cancellingAt,
                        cancelledAt: job.cancelledAt,
                        requestCounts: .init(total: job.total,
                                             completed: job.completed,
                                             failed: job.failed),
                        outputFileID: job.outputFileID,
                        errorFileID: job.errorFileID,
                        metadata: job.metadata)
    }


    private func appendSuccess(to url: URL,
                               customID: String,
                               completion: ServerCompletion,
                               modelID: String) throws {
        let content: Any = completion.content.isEmpty && !completion.toolCalls.isEmpty
            ? NSNull() : completion.content
        var message: [String: Any] = ["role": "assistant", "content": content]
        if !completion.toolCalls.isEmpty {
            message["tool_calls"] = completion.toolCalls.map {
                ["id": $0.id,
                 "type": "function",
                 "function": ["name": $0.name, "arguments": $0.argumentsJSON]]
            }
        }
        let created = Int(Date().timeIntervalSince1970)
        let completionID = "chatcmpl-" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
        let body: [String: Any] = [
            "id": completionID,
            "object": "chat.completion",
            "created": created,
            "model": modelID,
            "choices": [["index": 0, "message": message, "finish_reason": completion.finishReason]],
            "usage": usageObject(completion.usage),
        ]
        try appendLine([
            "id": "batch_req_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: ""),
            "custom_id": customID,
            "response": ["status_code": 200, "request_id": completionID, "body": body],
            "error": NSNull(),
        ], to: url)
    }

    private func appendFailure(to url: URL, customID: String, error: Error) throws {
        let detail: [String: Any]
        if let error = error as? ServerRequestError {
            detail = ["code": error.envelope.error.code, "message": error.envelope.error.message]
        } else {
            detail = ["code": "internal_error", "message": "generation failed"]
        }
        try appendLine([
            "id": "batch_req_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: ""),
            "custom_id": customID,
            "response": NSNull(),
            "error": detail,
        ], to: url)
    }

    private func appendLine(_ object: [String: Any], to url: URL) throws {
        var data = try JSONSerialization.data(withJSONObject: object)
        data.append(0x0A)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: data)
    }

    private func usageObject(_ usage: OpenAIUsage) -> [String: Any] {
        ["prompt_tokens": usage.promptTokens,
         "completion_tokens": usage.completionTokens,
         "total_tokens": usage.totalTokens,
         "prompt_tokens_details": ["cached_tokens": usage.promptTokensDetails.cachedTokens]]
    }
}
