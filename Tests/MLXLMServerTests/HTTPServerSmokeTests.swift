import Foundation
import XCTest
@testable import MLXLMServer

final class HTTPServerSmokeTests: XCTestCase {
    func testHealthEndpointReturnsReadyJSON() async throws {
        let server = MLXLMHTTPServer(
            engine: StubEngine(),
            host: "127.0.0.1",
            port: 0   // ephemeral
        )

        let bindResult = try server.bindAndRun()
        let port = bindResult.1
        defer { try? server.stop() }

        let url = URL(string: "http://127.0.0.1:\(port)/health")!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"

        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200)

        let payload = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        XCTAssertEqual(payload["status"] as? String, "ready")
        XCTAssertEqual(payload["model_ids"] as? [String], ["stub-model"])
    }

    func testUnknownRouteReturns404WithOpenAIErrorShape() async throws {
        let server = MLXLMHTTPServer(
            engine: StubEngine(),
            host: "127.0.0.1",
            port: 0
        )
        let bindResult = try server.bindAndRun()
        let port = bindResult.1
        defer { try? server.stop() }

        let url = URL(string: "http://127.0.0.1:\(port)/does-not-exist")!
        let (data, response) = try await URLSession.shared.data(from: url)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 404)

        let payload = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let errorObject = try XCTUnwrap(payload["error"] as? [String: Any])
        XCTAssertEqual(errorObject["type"] as? String, "invalid_request_error")
    }
}
