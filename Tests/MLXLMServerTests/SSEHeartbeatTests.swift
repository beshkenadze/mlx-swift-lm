import Foundation
import XCTest
@testable import MLXLMServer

/// Covers the `heartbeatInterval:` option on `MLXLMHTTPServer`. When a stream
/// idles past the configured interval the handler must emit `: keepalive\n\n`
/// SSE comments so intermediate L7 proxies do not close the connection.
/// Spec: dflash-mlx §6.6 (R-301, T-301c).
final class SSEHeartbeatTests: XCTestCase {
    func testSSEHeartbeatFiresOnSlowStream() async throws {
        // Engine emits one fragment after ~2.5s, then finishes. With a 1s
        // heartbeat we expect at least two `: keepalive` comments before the
        // delta and none after (stream closes promptly).
        let engine = SlowEmitterEngine(
            fragments: ["hi"],
            initialDelayMs: 2_500,
            interFragmentDelayMs: 0
        )
        let server = MLXLMHTTPServer(
            engine: engine,
            host: "127.0.0.1",
            port: 0,
            heartbeatInterval: 1
        )
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

        var allLines: [String] = []
        for try await line in bytes.lines {
            allLines.append(line)
        }

        // SSE comment lines start with `:` — URLSession.bytes.lines strips
        // the trailing `\n\n` so each keepalive appears as ": keepalive".
        let keepalives = allLines.filter { $0.hasPrefix(": keepalive") }
        XCTAssertGreaterThanOrEqual(
            keepalives.count, 2,
            "expected >=2 `: keepalive` comments in idle stream, got \(keepalives.count): \(allLines)"
        )

        // Regression guard: real deltas + [DONE] still flow through.
        let dataLines = allLines.filter { $0.hasPrefix("data: ") }
        XCTAssertEqual(dataLines.last, "data: [DONE]")
        XCTAssertGreaterThanOrEqual(dataLines.count, 2)  // 1 content + final + [DONE]
    }

    func testHeartbeatDisabledWhenNil() async throws {
        // Fast stream + no heartbeat configured → no comment lines.
        let engine = SlowEmitterEngine(
            fragments: ["a", "b"],
            initialDelayMs: 0,
            interFragmentDelayMs: 0
        )
        let server = MLXLMHTTPServer(
            engine: engine,
            host: "127.0.0.1",
            port: 0,
            heartbeatInterval: nil
        )
        let (_, port) = try server.bindAndRun()
        defer { try? server.stop() }

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"""
        {"model":"stub","messages":[{"role":"user","content":"x"}],"stream":true}
        """#.utf8)

        let (bytes, _) = try await URLSession.shared.bytes(for: request)
        var allLines: [String] = []
        for try await line in bytes.lines {
            allLines.append(line)
        }
        let keepalives = allLines.filter { $0.hasPrefix(": keepalive") }
        XCTAssertEqual(keepalives.count, 0)
    }
}

/// Test-only engine with configurable latency between fragments. Used to
/// exercise the SSE heartbeat path without needing MLX runtime.
struct SlowEmitterEngine: InferenceEngine {
    let fragments: [String]
    let initialDelayMs: Int
    let interFragmentDelayMs: Int

    func availableModels() async -> [ModelInfo] {
        [ModelInfo(id: "stub", created: 0, ownedBy: "tests")]
    }

    func health() async -> EngineHealth {
        EngineHealth(ready: true, modelIDs: ["stub"], uptimeSeconds: 0)
    }

    func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error> {
        let fragments = self.fragments
        let initial = self.initialDelayMs
        let inter = self.interFragmentDelayMs
        return AsyncThrowingStream<ChatDelta, Error>(bufferingPolicy: .unbounded) { (continuation: AsyncThrowingStream<ChatDelta, Error>.Continuation) in
            let task = Task {
                if initial > 0 {
                    try? await Task.sleep(nanoseconds: UInt64(initial) * 1_000_000)
                }
                for (i, fragment) in fragments.enumerated() {
                    if i > 0, inter > 0 {
                        try? await Task.sleep(nanoseconds: UInt64(inter) * 1_000_000)
                    }
                    continuation.yield(ChatDelta(textFragments: [fragment]))
                }
                continuation.yield(
                    ChatDelta(
                        textFragments: [],
                        finishReason: .stop,
                        usage: Usage(
                            promptTokens: 1,
                            completionTokens: fragments.count
                        )
                    )
                )
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
