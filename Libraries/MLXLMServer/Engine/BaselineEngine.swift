import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

public final class BaselineEngine: InferenceEngine, @unchecked Sendable {
    public let configuration: BaselineEngineConfiguration
    private let start: Date
    private var loaded: Bool = false

    // Stored after a successful `load()`. Types from MLXLMCommon are not
    // universally `Sendable`; we guard access with `@unchecked Sendable` on
    // this class plus the `loaded` latch.
    private var container: ModelContainer?

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

    /// Load the model + tokenizer referenced by `configuration.modelID`.
    ///
    /// Uses `LLMModelFactory.shared.loadContainer` with a Hugging Face
    /// `HubClient` downloader and the swift-transformers `AutoTokenizer`
    /// loader (both supplied by `MLXHuggingFace` macros). On success the
    /// container is retained and `health().ready` becomes `true`.
    public func load() async throws {
        do {
            let container = try await LLMModelFactory.shared.loadContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: .init(id: configuration.modelID)
            )
            self.container = container
            self.loaded = true
        } catch {
            throw BaselineEngineError.modelLoadFailed(String(describing: error))
        }
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
