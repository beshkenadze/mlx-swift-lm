import Foundation

public enum ChatRequestDecoderError: Error, Equatable {
    case invalidJSON(String)
    case missingField(String)
    case emptyMessages
    case invalidStopField
}

public enum ChatRequestDecoder {
    private struct Payload: Decodable {
        let model: String
        let messages: [ChatMessage]
        let maxTokens: Int?
        let stream: Bool?
        let stop: StopField?

        enum CodingKeys: String, CodingKey {
            case model, messages, stream, stop
            case maxTokens = "max_tokens"
        }
    }

    private enum StopField: Decodable {
        case single(String)
        case list([String])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let one = try? container.decode(String.self) {
                self = .single(one)
            } else if let many = try? container.decode([String].self) {
                self = .list(many)
            } else {
                throw ChatRequestDecoderError.invalidStopField
            }
        }

        var values: Set<String> {
            switch self {
            case .single(let s): return [s]
            case .list(let xs): return Set(xs)
            }
        }
    }

    public static func decode(_ data: Data, defaultMaxTokens: Int = 256) throws -> ChatRequest {
        let payload: Payload
        do {
            payload = try JSONDecoder().decode(Payload.self, from: data)
        } catch {
            throw ChatRequestDecoderError.invalidJSON(String(describing: error))
        }
        guard !payload.messages.isEmpty else {
            throw ChatRequestDecoderError.emptyMessages
        }
        return ChatRequest(
            modelID: payload.model,
            messages: payload.messages,
            maxTokens: payload.maxTokens ?? defaultMaxTokens,
            stopSequences: payload.stop?.values ?? [],
            stream: payload.stream ?? false
        )
    }
}
