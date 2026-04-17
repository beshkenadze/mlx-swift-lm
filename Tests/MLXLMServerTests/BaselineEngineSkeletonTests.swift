import XCTest
@testable import MLXLMServer

final class BaselineEngineSkeletonTests: XCTestCase {
    func testAvailableModelsReflectConfig() async {
        let engine = BaselineEngine(configuration: BaselineEngineConfiguration(modelID: "foo/bar"))
        let models = await engine.availableModels()
        XCTAssertEqual(models.first?.id, "foo/bar")
    }

    func testHealthNotReadyBeforeLoad() async {
        let engine = BaselineEngine(configuration: BaselineEngineConfiguration(modelID: "foo/bar"))
        let health = await engine.health()
        XCTAssertFalse(health.ready)
    }
}
