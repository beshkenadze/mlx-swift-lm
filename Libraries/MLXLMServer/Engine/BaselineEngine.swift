import Foundation

public final class BaselineEngine: InferenceEngine, @unchecked Sendable {
    public let configuration: BaselineEngineConfiguration
    private let start: Date
    private var loaded: Bool = false

    public init(configuration: BaselineEngineConfiguration) {
        self.configuration = configuration
        self.start = Date()
    }

    public func availableModels() async -> [ModelInfo] {
        [ModelInfo(
            id: configuration.modelID,
            created: Int(start.timeIntervalSince1970),
            ownedBy: "mlx-swift-lm"
        )]
    }

    public func health() async -> EngineHealth {
        EngineHealth(
            ready: loaded,
            modelIDs: [configuration.modelID],
            uptimeSeconds: Date().timeIntervalSince(start)
        )
    }

    public func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: BaselineEngineError.notYetImplemented)
        }
    }
}

public enum BaselineEngineError: Error {
    case notYetImplemented
    case modelLoadFailed(String)
    case tokenizationFailed(String)
}
