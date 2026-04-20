import Foundation

public enum SSEFrame {
    public static let done = "data: [DONE]\n\n"

    /// SSE comment line emitted as a heartbeat during idle streams. Matches
    /// the spec §6.6 wire format (`: keepalive\n\n`). Comments are ignored by
    /// conformant SSE clients but keep intermediate proxies from closing the
    /// connection on idle-timeout.
    public static let keepalive = ": keepalive\n\n"

    public static func data(_ jsonBody: String) -> String {
        "data: \(jsonBody)\n\n"
    }

    /// Produces one `chat.completion.chunk` frame. Pass `contentDelta` nil on the
    /// final frame and `finishReason` nil on intermediate frames.
    public static func chatCompletionChunk(
        id: String,
        model: String,
        created: Int,
        contentDelta: String?,
        finishReason: String?,
        includeAssistantRole: Bool = false,
        usage: Usage? = nil
    ) -> String {
        var delta: [String: Any] = [:]
        if includeAssistantRole { delta["role"] = "assistant" }
        if let c = contentDelta { delta["content"] = c }

        var choice: [String: Any] = [
            "index": 0,
            "delta": delta,
        ]
        choice["finish_reason"] = finishReason as Any? ?? NSNull()

        var payload: [String: Any] = [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [choice],
        ]
        if let usage {
            payload["usage"] = usagePayload(usage)
        }

        let encoded = (try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])) ?? Data()
        let json = String(data: encoded, encoding: .utf8) ?? "{}"
        return data(json)
    }

    private static func usagePayload(_ usage: Usage) -> [String: Any] {
        var payload: [String: Any] = [
            "prompt_tokens": usage.promptTokens,
            "completion_tokens": usage.completionTokens,
            "total_tokens": usage.totalTokens,
        ]
        if let acceptanceRate = usage.acceptanceRate {
            payload["acceptance_rate"] = acceptanceRate
        }
        return payload
    }
}
