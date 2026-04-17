import Foundation
import HuggingFace
import MLX
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

    /// Stream chat-completion deltas autoregressively via MLXLLM.
    ///
    /// The pipeline:
    /// 1. Apply the tokenizer chat template to `request.messages` to get
    ///    prompt token IDs.
    /// 2. Build a greedy `GenerateParameters` (temperature 0 → argmax).
    /// 3. Run `MLXLMCommon.generate(...)` which returns an
    ///    `AsyncStream<Generation>` and forward each `.chunk` as a
    ///    `ChatDelta` text fragment.
    /// 4. On `.info`, emit a final empty-fragments delta carrying the
    ///    finish reason and token-usage accounting.
    public func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                guard let container = self.container else {
                    continuation.finish(
                        throwing: BaselineEngineError.modelLoadFailed("call load() first"))
                    return
                }

                do {
                    try await container.perform { ctx in
                        // 1. Apply chat template — yields [Int] directly (skips
                        //    the separate encode step).
                        let rawMessages: [[String: any Sendable]] = request.messages.map { msg in
                            ["role": msg.role, "content": msg.content]
                        }
                        let promptTokens: [Int]
                        do {
                            promptTokens = try ctx.tokenizer.applyChatTemplate(
                                messages: rawMessages)
                        } catch {
                            throw BaselineEngineError.tokenizationFailed(String(describing: error))
                        }

                        // 2. Cap max tokens at context window minus prompt length.
                        let headroom = max(
                            0, self.configuration.contextWindow - promptTokens.count)
                        let maxTokens = max(1, min(request.maxTokens, headroom))

                        // 3. Greedy sampling: temperature == 0 selects ArgMaxSampler
                        //    inside GenerateParameters.sampler().
                        let parameters = GenerateParameters(
                            maxTokens: maxTokens, temperature: 0
                        )
                        let input = LMInput(tokens: MLXArray(promptTokens))

                        let stream = try MLXLMCommon.generate(
                            input: input, parameters: parameters, context: ctx)

                        let promptTokenCount = promptTokens.count

                        for await item in stream {
                            if Task.isCancelled {
                                continuation.finish(throwing: CancellationError())
                                return
                            }

                            switch item {
                            case .chunk(let text):
                                if !text.isEmpty {
                                    continuation.yield(
                                        ChatDelta(textFragments: [text]))
                                }

                            case .info(let info):
                                let finish: FinishReason
                                switch info.stopReason {
                                case .stop: finish = .stop
                                case .length: finish = .length
                                case .cancelled: finish = .cancelled
                                }
                                continuation.yield(
                                    ChatDelta(
                                        textFragments: [],
                                        finishReason: finish,
                                        usage: Usage(
                                            promptTokens: promptTokenCount,
                                            completionTokens: info.generationTokenCount
                                        )
                                    ))

                            case .toolCall:
                                // Tool calls aren't part of the baseline HTTP
                                // surface yet; drop them.
                                continue
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}

public enum BaselineEngineError: Error {
    case notYetImplemented
    case modelLoadFailed(String)
    case tokenizationFailed(String)
}
