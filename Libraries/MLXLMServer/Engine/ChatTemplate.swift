import Foundation
import Jinja

/// Minimal wrapper over `Jinja.Template` for rendering chat messages into
/// a prompt string. `BaselineEngine` consumes this so it never touches
/// Jinja or tokenizer internals directly.
public struct ChatTemplate: Sendable {
    public let source: String

    public init(source: String) {
        self.source = source
    }

    /// Factory for a literal Jinja template string (useful in tests and
    /// for model-specific templates read from `tokenizer_config.json`).
    public static func literal(_ template: String) -> ChatTemplate {
        ChatTemplate(source: template)
    }

    /// Render `messages` through the template. `addGenerationPrompt`
    /// mirrors the HuggingFace convention (`add_generation_prompt`).
    public func render(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = false
    ) throws -> String {
        let messageValues: [Value] = messages.map { message in
            .object([
                "role": .string(message.role),
                "content": .string(message.content),
            ])
        }
        let context: [String: Value] = [
            "messages": .array(messageValues),
            "add_generation_prompt": .boolean(addGenerationPrompt),
        ]
        let template = try Template(source)
        return try template.render(context)
    }
}
