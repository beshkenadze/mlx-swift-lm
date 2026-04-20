import Foundation
import XCTest
@testable import MLXLMServer

final class ChatCompletionsStreamingTests: XCTestCase {
    func testStreamingReturnsSSEChunks() async throws {
        // Stub emits: "hello", " ", "world", then a final-empty-delta with finishReason.
        let engine = MultiDeltaStubEngine(
            deltas: [
                ChatDelta(textFragments: ["hello"]),
                ChatDelta(textFragments: [" "]),
                ChatDelta(textFragments: ["world"]),
                ChatDelta(
                    textFragments: [],
                    finishReason: .stop,
                    usage: Usage(promptTokens: 1, completionTokens: 3, acceptanceRate: 0.75)
                ),
            ]
        )
        let server = MLXLMHTTPServer(engine: engine, host: "127.0.0.1", port: 0)
        let (_, port) = try server.bindAndRun()
        defer { try? server.stop() }

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"""
        {"model":"stub","messages":[{"role":"user","content":"x"}],"stream":true}
        """#.utf8)

        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200)
        XCTAssertTrue(
            http.value(forHTTPHeaderField: "Content-Type")?.contains("text/event-stream") ?? false
        )

        var collectedLines: [String] = []
        for try await line in bytes.lines {
            collectedLines.append(line)
        }

        let dataLines = collectedLines.filter { $0.hasPrefix("data: ") }
        XCTAssertGreaterThanOrEqual(dataLines.count, 4)  // 3 content + 1 final + [DONE]
        XCTAssertEqual(dataLines.last, "data: [DONE]")

        let contentChunks = dataLines.dropLast()    // drop [DONE]
        var finalUsage: [String: Any]?
        let reconstructed = contentChunks.compactMap { line -> String? in
            let jsonStart = line.index(line.startIndex, offsetBy: "data: ".count)
            let data = Data(line[jsonStart...].utf8)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = obj["choices"] as? [[String: Any]],
                  let delta = choices.first?["delta"] as? [String: Any]
            else { return nil }
            if let usage = obj["usage"] as? [String: Any] {
                finalUsage = usage
            }
            return delta["content"] as? String
        }.joined()
        XCTAssertEqual(reconstructed, "hello world")
        XCTAssertEqual(finalUsage?["acceptance_rate"] as? Double, 0.75)
    }
}

/// Test-only engine that replays a fixed sequence of deltas.
struct MultiDeltaStubEngine: InferenceEngine {
    let deltas: [ChatDelta]

    func availableModels() async -> [ModelInfo] {
        [ModelInfo(id: "stub", created: 0, ownedBy: "tests")]
    }

    func health() async -> EngineHealth {
        EngineHealth(ready: true, modelIDs: ["stub"], uptimeSeconds: 0)
    }

    func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error> {
        AsyncThrowingStream { continuation in
            for delta in deltas {
                continuation.yield(delta)
            }
            continuation.finish()
        }
    }
}
