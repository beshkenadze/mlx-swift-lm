import XCTest

@testable import MLXLMServer

final class ChatTemplateTests: XCTestCase {
    func testRendersMessagesToPromptString() throws {
        let template = ChatTemplate.literal(
            "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"
        )
        let rendered = try template.render(messages: [
            ChatMessage(role: "system", content: "sys"),
            ChatMessage(role: "user", content: "hi"),
        ])
        XCTAssertEqual(rendered, "system:sys\nuser:hi\n")
    }
}
