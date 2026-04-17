import XCTest
@testable import MLXLMServer

final class BaselineEngineConfigurationTests: XCTestCase {
    func testDefaults() {
        let config = BaselineEngineConfiguration(modelID: "foo/bar")
        XCTAssertEqual(config.modelID, "foo/bar")
        XCTAssertEqual(config.defaultMaxTokens, 256)
        XCTAssertEqual(config.contextWindow, 4096)
    }
}
