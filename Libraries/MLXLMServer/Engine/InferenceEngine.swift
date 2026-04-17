import Foundation

/// Chat message in the OpenAI `{role, content}` shape.
public struct ChatMessage: Sendable, Codable, Equatable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// Normalized chat request that `InferenceEngine` conformers consume.
/// The HTTP layer handles OpenAI-specific wire format; engines see this.
public struct ChatRequest: Sendable {
    public let modelID: String
    public let messages: [ChatMessage]
    public let maxTokens: Int
    public let stopSequences: Set<String>
    public let stream: Bool

    public init(
        modelID: String,
        messages: [ChatMessage],
        maxTokens: Int,
        stopSequences: Set<String> = [],
        stream: Bool = true
    ) {
        self.modelID = modelID
        self.messages = messages
        self.maxTokens = maxTokens
        self.stopSequences = stopSequences
        self.stream = stream
    }

    /// Returns a copy with a different `modelID`. Used by `EngineRegistry`
    /// after stripping a routing prefix before delegating to the inner engine.
    public func withModelID(_ newModelID: String) -> ChatRequest {
        ChatRequest(
            modelID: newModelID,
            messages: messages,
            maxTokens: maxTokens,
            stopSequences: stopSequences,
            stream: stream
        )
    }
}

/// Why a chat completion stopped.
public enum FinishReason: String, Sendable, Codable {
    case stop
    case length
    case cancelled
}

/// Token-accounting data emitted alongside the final delta.
public struct Usage: Sendable, Codable, Equatable {
    public let promptTokens: Int
    public let completionTokens: Int
    public var totalTokens: Int { promptTokens + completionTokens }

    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }
}

/// One chunk of generated output. An engine may emit one delta per token
/// (AR baseline) or one delta per accepted block (dFlash τ > 1).
public struct ChatDelta: Sendable {
    /// Newly produced tokens, already detokenized to text fragments.
    public let textFragments: [String]
    /// Non-nil on the final delta only.
    public let finishReason: FinishReason?
    /// Non-nil on the final delta only.
    public let usage: Usage?

    public init(textFragments: [String], finishReason: FinishReason? = nil, usage: Usage? = nil) {
        self.textFragments = textFragments
        self.finishReason = finishReason
        self.usage = usage
    }
}

/// Static metadata about a model that an engine exposes.
public struct ModelInfo: Sendable, Codable, Equatable {
    public let id: String
    public let created: Int
    public let ownedBy: String

    public init(id: String, created: Int, ownedBy: String) {
        self.id = id
        self.created = created
        self.ownedBy = ownedBy
    }
}

/// Engine health for the `/health` probe.
public struct EngineHealth: Sendable, Codable, Equatable {
    public let ready: Bool
    public let modelIDs: [String]
    public let uptimeSeconds: Double

    public init(ready: Bool, modelIDs: [String], uptimeSeconds: Double) {
        self.ready = ready
        self.modelIDs = modelIDs
        self.uptimeSeconds = uptimeSeconds
    }
}

/// Minimal contract any MLX-backed engine must satisfy to be served over HTTP.
/// Deliberately narrow: models, a streaming generate, and health. Everything
/// else (chat templating, tokenization, KV-cache lifecycle) is the engine's
/// own responsibility.
public protocol InferenceEngine: Sendable {
    /// Models this engine can serve (usually exactly one).
    func availableModels() async -> [ModelInfo]

    /// Health probe; MUST return within ~10 ms.
    func health() async -> EngineHealth

    /// Stream chat-completion deltas. The stream SHOULD terminate with a
    /// final delta carrying a non-nil `finishReason`. On cancellation
    /// (HTTP connection drop) the server will cancel the `Task` consuming
    /// the stream; engines should honor `Task.isCancelled` in their loops.
    func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error>
}
