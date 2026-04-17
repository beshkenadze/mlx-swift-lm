import Foundation
import XCTest
@testable import MLXLMServer

final class SingleFlightConcurrencyTests: XCTestCase {
    func testSecondConcurrentRequestReturns409() async throws {
        // Stub that blocks until we release a continuation — lets the first
        // request stay in-flight while the second fires.
        let engine = BlockingStubEngine()
        let server = MLXLMHTTPServer(engine: engine, host: "127.0.0.1", port: 0)
        let (_, port) = try server.bindAndRun()
        defer { try? server.stop() }

        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"""
        {"model":"stub","messages":[{"role":"user","content":"x"}]}
        """#.utf8)

        let firstRequest = request
        let secondRequest = request
        async let firstTask: (Data, URLResponse) = URLSession.shared.data(for: firstRequest)
        // give the first request a moment to land and acquire the gate
        try await Task.sleep(nanoseconds: 100_000_000)
        async let secondTask: (Data, URLResponse) = URLSession.shared.data(for: secondRequest)

        let second = try await secondTask
        let secondHTTP = try XCTUnwrap(second.1 as? HTTPURLResponse)
        XCTAssertEqual(secondHTTP.statusCode, 409)

        // unblock the first request so the server drains
        await engine.unblock()
        let first = try await firstTask
        let firstHTTP = try XCTUnwrap(first.1 as? HTTPURLResponse)
        XCTAssertEqual(firstHTTP.statusCode, 200)
    }
}

actor BlockingStubEngine: InferenceEngine {
    private var continuation: AsyncThrowingStream<ChatDelta, Error>.Continuation?
    private var unblocked = false

    func availableModels() async -> [ModelInfo] {
        [ModelInfo(id: "stub", created: 0, ownedBy: "tests")]
    }

    func health() async -> EngineHealth {
        EngineHealth(ready: true, modelIDs: ["stub"], uptimeSeconds: 0)
    }

    nonisolated func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error> {
        AsyncThrowingStream { continuation in
            Task { await self.register(continuation: continuation) }
        }
    }

    private func register(continuation: AsyncThrowingStream<ChatDelta, Error>.Continuation) {
        if unblocked {
            continuation.yield(ChatDelta(textFragments: ["ok"], finishReason: .stop, usage: Usage(promptTokens: 1, completionTokens: 1)))
            continuation.finish()
        } else {
            self.continuation = continuation
        }
    }

    func unblock() {
        unblocked = true
        continuation?.yield(ChatDelta(textFragments: ["ok"], finishReason: .stop, usage: Usage(promptTokens: 1, completionTokens: 1)))
        continuation?.finish()
        continuation = nil
    }
}
