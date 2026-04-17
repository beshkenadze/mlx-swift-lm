import Foundation
import XCTest
@testable import MLXLMServer

final class ModelsEndpointTests: XCTestCase {
    func testListModelsReturnsOpenAIShape() async throws {
        let engine = StubEngine(
            models: [
                ModelInfo(id: "foo-model", created: 1, ownedBy: "tests"),
                ModelInfo(id: "bar-model", created: 2, ownedBy: "tests"),
            ]
        )
        let server = MLXLMHTTPServer(engine: engine, host: "127.0.0.1", port: 0)
        let (_, port) = try server.bindAndRun()
        defer { try? server.stop() }

        let url = URL(string: "http://127.0.0.1:\(port)/v1/models")!
        let (data, response) = try await URLSession.shared.data(from: url)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200)

        let payload = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        XCTAssertEqual(payload["object"] as? String, "list")
        let items = try XCTUnwrap(payload["data"] as? [[String: Any]])
        XCTAssertEqual(items.count, 2)
        XCTAssertEqual(items[0]["id"] as? String, "foo-model")
        XCTAssertEqual(items[0]["object"] as? String, "model")
        XCTAssertEqual(items[0]["owned_by"] as? String, "tests")
        XCTAssertEqual(items[0]["created"] as? Int, 1)
    }
}
