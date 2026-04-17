import XCTest

@testable import MLXLMServer

final class BaselineEngineGenerateTests: XCTestCase {
    func testGenerateEmitsAssistantTextAndFinalDelta() async throws {
        try XCTSkipUnless(runtimeMetallibAvailable(), "MLX metallib unavailable")

        // Small quantized Qwen2.5 checkpoint (~400 MB) — fast HF download.
        // Requires network on first run to populate the HF cache.
        let engine = BaselineEngine(
            configuration: BaselineEngineConfiguration(
                modelID: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                defaultMaxTokens: 8
            )
        )
        try await engine.load()

        let request = ChatRequest(
            modelID: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            messages: [ChatMessage(role: "user", content: "Hi.")],
            maxTokens: 8
        )

        var textFragments: [String] = []
        var finishReason: FinishReason?
        var usage: Usage?
        for try await delta in engine.generate(request) {
            textFragments.append(contentsOf: delta.textFragments)
            if let reason = delta.finishReason { finishReason = reason }
            if let u = delta.usage { usage = u }
        }

        XCTAssertFalse(
            textFragments.joined().isEmpty,
            "expected at least one non-empty text fragment"
        )
        XCTAssertNotNil(finishReason, "expected a finish reason on the final delta")
        XCTAssertNotNil(usage, "expected usage on the final delta")
        XCTAssertGreaterThan(usage?.promptTokens ?? 0, 0)
        XCTAssertGreaterThan(usage?.completionTokens ?? 0, 0)
    }
}
