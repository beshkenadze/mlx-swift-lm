import Foundation
@testable import MLXLMServer

/// Zero-dependency `InferenceEngine` implementation used by server smoke
/// tests. Returns a canned model list and a canned single-delta stream.
public struct StubEngine: InferenceEngine {
    public let models: [ModelInfo]
    public let healthReady: Bool
    public let cannedResponse: String
    public let usage: Usage

    public init(
        models: [ModelInfo] = [ModelInfo(id: "stub-model", created: 0, ownedBy: "tests")],
        healthReady: Bool = true,
        cannedResponse: String = "hello from stub",
        usage: Usage = Usage(promptTokens: 0, completionTokens: 1)
    ) {
        self.models = models
        self.healthReady = healthReady
        self.cannedResponse = cannedResponse
        self.usage = usage
    }

    public func availableModels() async -> [ModelInfo] {
        models
    }

    public func health() async -> EngineHealth {
        EngineHealth(
            ready: healthReady,
            modelIDs: models.map(\.id),
            uptimeSeconds: 0
        )
    }

    public func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(
                ChatDelta(
                    textFragments: [cannedResponse],
                    finishReason: .stop,
                    usage: usage
                )
            )
            continuation.finish()
        }
    }
}
