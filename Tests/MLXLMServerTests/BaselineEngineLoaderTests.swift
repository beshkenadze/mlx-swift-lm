import XCTest

@testable import MLXLMServer

final class BaselineEngineLoaderTests: XCTestCase {
    func testLoadMarksHealthReady() async throws {
        try XCTSkipUnless(runtimeMetallibAvailable(), "MLX metallib unavailable")

        // Small quantized Qwen2.5 checkpoint (~400 MB) — fast HF download.
        // Note: this test requires network on first run to populate the HF cache.
        let engine = BaselineEngine(
            configuration: BaselineEngineConfiguration(
                modelID: "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
            )
        )

        let preLoadHealth = await engine.health()
        XCTAssertFalse(preLoadHealth.ready, "engine must be not-ready before load()")

        try await engine.load()

        let postLoadHealth = await engine.health()
        XCTAssertTrue(postLoadHealth.ready, "engine must be ready after successful load()")
        XCTAssertEqual(postLoadHealth.modelIDs, ["mlx-community/Qwen2.5-0.5B-Instruct-4bit"])
    }
}
