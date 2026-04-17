import Foundation
import XCTest
@testable import MLXLMServer

final class ChatCompletionsNonStreamingTests: XCTestCase {
    func testNonStreamingCompletion() async throws {
        let engine = StubEngine(cannedResponse: "stub says hi")
        let server = MLXLMHTTPServer(engine: engine, host: "127.0.0.1", port: 0)
        let (_, port) = try server.bindAndRun()
        defer { try? server.stop() }

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"""
        {"model":"stub-model","messages":[{"role":"user","content":"hi"}],"stream":false}
        """#.utf8)

        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200)

        let payload = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        XCTAssertEqual(payload["object"] as? String, "chat.completion")
        XCTAssertEqual(payload["model"] as? String, "stub-model")

        let choices = try XCTUnwrap(payload["choices"] as? [[String: Any]])
        XCTAssertEqual(choices.count, 1)
        let message = try XCTUnwrap(choices[0]["message"] as? [String: Any])
        XCTAssertEqual(message["role"] as? String, "assistant")
        XCTAssertEqual(message["content"] as? String, "stub says hi")
        XCTAssertEqual(choices[0]["finish_reason"] as? String, "stop")

        let usage = try XCTUnwrap(payload["usage"] as? [String: Any])
        XCTAssertEqual(usage["completion_tokens"] as? Int, 1)
    }

    func testMalformedBodyReturns400() async throws {
        let server = MLXLMHTTPServer(engine: StubEngine(), host: "127.0.0.1", port: 0)
        let (_, port) = try server.bindAndRun()
        defer { try? server.stop() }

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("not json".utf8)

        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 400)
        let payload = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let err = try XCTUnwrap(payload["error"] as? [String: Any])
        XCTAssertEqual(err["type"] as? String, "invalid_request_error")
    }
}
