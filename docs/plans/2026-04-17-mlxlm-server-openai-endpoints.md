# MLXLMServer OpenAI Endpoints Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the `MLXLMServer` target from the current `/health`-only skeleton into a production-useful OpenAI-compatible HTTP front-end (models + non-streaming chat + SSE + single-flight guard) and add a first real engine (`BaselineEngine`) that wraps `MLXLLM.generate`.

**Architecture:** Incremental feature layers on top of existing `MLXLMHTTPServer` / `MLXLMHTTPHandler`. Each new endpoint lands as (1) a small pure-logic unit (request decoder, SSE writer, single-flight actor) with its own unit tests, then (2) is wired into the handler and verified with a URLSession integration test against an ephemeral-port server driven by `StubEngine`. `BaselineEngine` lives in its own subdirectory and depends on `MLXLLM` + `MLXLMCommon` + `swift-transformers` (tokenizer) + `swift-jinja` (chat template).

**Tech Stack:** Swift 6.1, SwiftPM, swift-nio 2.75+, NIOHTTP1, NIOPosix, XCTest, URLSession (test client), MLX-Swift 0.31.3, MLXLLM/MLXLMCommon (existing targets), swift-transformers, swift-jinja.

**Baseline:** commit `fd68eb2` on branch `feat/mlxlm-server`. `swift test --filter MLXLMServerTests` green: 2 tests (/health + 404).

**Testing conventions:**
- Run a single suite: `swift test --filter MLXLMServerTests.<SuiteName>`
- Run a single test: `swift test --filter MLXLMServerTests.<SuiteName>/<methodName>`
- Full run: `swift test`
- All tests MUST pass non-interactively. No input prompts.

**Commit conventions:** Conventional Commits (`feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`). Single-purpose atomic commits. No co-author/claude footers. All commits signed (SSH via 1Password) — if signing fails, STOP and report; do not bypass.

**MLX `eval` note:** references to `MLX.eval` throughout this plan mean the MLX-Swift top-level synchronization function (unrelated to any dynamic-code-execution primitive). Always write it fully qualified.

**Scope note:** All five phases target the server infrastructure. No dFlash / TriAttention / TurboQuant specifics — those land in separate follow-up branches that CONSUME this server. Stay focused.

---

## Phase 1 — `GET /v1/models`

### Task 1.1: Add `/v1/models` endpoint

**Files:**
- Modify: `Libraries/MLXLMServer/HTTP/MLXLMHTTPHandler.swift`
- Create: `Tests/MLXLMServerTests/ModelsEndpointTests.swift`

**Step 1: Write the failing test**

Create `Tests/MLXLMServerTests/ModelsEndpointTests.swift`:

```swift
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
```

**Step 2: Run test, verify fail**

```
swift test --filter MLXLMServerTests.ModelsEndpointTests/testListModelsReturnsOpenAIShape
```

Expected: FAIL — currently routes to 404.

**Step 3: Wire the route**

In `Libraries/MLXLMServer/HTTP/MLXLMHTTPHandler.swift`, locate the `route(context:head:)` method and add a case before `default:`:

```swift
case (.GET, "/v1/models"):
    handleListModels(context: context, head: head)
```

Below `handleHealth`, add:

```swift
private func handleListModels(context: ChannelHandlerContext, head: HTTPRequestHead) {
    let eventLoop = context.eventLoop
    let contextBox = NIOLoopBound(context, eventLoop: eventLoop)
    let headBox = NIOLoopBound(head, eventLoop: eventLoop)
    let engine = self.engine

    Task {
        let models = await engine.availableModels()
        let data: [[String: Any]] = models.map { model in
            [
                "id": model.id,
                "object": "model",
                "created": model.created,
                "owned_by": model.ownedBy,
            ]
        }
        let payload: [String: Any] = [
            "object": "list",
            "data": data,
        ]
        let encoded = (try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])) ?? Data()
        let json = String(data: encoded, encoding: .utf8) ?? "{\"object\":\"list\",\"data\":[]}"

        eventLoop.execute {
            self.respondJSON(
                context: contextBox.value,
                head: headBox.value,
                status: .ok,
                body: json
            )
        }
    }
}
```

**Step 4: Run test, verify pass**

```
swift test --filter MLXLMServerTests.ModelsEndpointTests
```

Expected: PASS, 1 test.

**Step 5: Run the broader suite**

```
swift test --filter MLXLMServerTests
```

Expected: PASS, 3 tests total (health + 404 + models).

**Step 6: Commit**

```
cd /Volumes/DATA/mlx-swift-lm
git add Libraries/MLXLMServer/HTTP/MLXLMHTTPHandler.swift \
        Tests/MLXLMServerTests/ModelsEndpointTests.swift
git commit -S -m "feat(server): GET /v1/models endpoint"
```

---

## Phase 2 — `POST /v1/chat/completions` (non-streaming)

### Task 2.1: Request decoder (pure logic, unit-tested in isolation)

**Files:**
- Create: `Libraries/MLXLMServer/HTTP/ChatRequestDecoder.swift`
- Create: `Tests/MLXLMServerTests/ChatRequestDecoderTests.swift`

**Step 1: Write the failing test**

Create `Tests/MLXLMServerTests/ChatRequestDecoderTests.swift`:

```swift
import Foundation
import XCTest
@testable import MLXLMServer

final class ChatRequestDecoderTests: XCTestCase {
    func testDecodeMinimalRequest() throws {
        let json = #"""
        {
          "model": "stub-model",
          "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "say hi"}
          ]
        }
        """#
        let request = try ChatRequestDecoder.decode(Data(json.utf8))
        XCTAssertEqual(request.modelID, "stub-model")
        XCTAssertEqual(request.messages.count, 2)
        XCTAssertEqual(request.messages[0].role, "system")
        XCTAssertEqual(request.messages[1].content, "say hi")
        XCTAssertEqual(request.maxTokens, 256)     // default
        XCTAssertEqual(request.stream, false)      // default
        XCTAssertEqual(request.stopSequences, [])
    }

    func testDecodeWithOverrides() throws {
        let json = #"""
        {
          "model": "m",
          "messages": [{"role":"user","content":"hi"}],
          "max_tokens": 10,
          "stream": true,
          "stop": ["\n\n", "<|end|>"]
        }
        """#
        let request = try ChatRequestDecoder.decode(Data(json.utf8))
        XCTAssertEqual(request.maxTokens, 10)
        XCTAssertEqual(request.stream, true)
        XCTAssertEqual(request.stopSequences, ["\n\n", "<|end|>"])
    }

    func testDecodeWithStopAsSingleString() throws {
        let json = #"""
        { "model": "m",
          "messages": [{"role":"user","content":"x"}],
          "stop": "DONE" }
        """#
        let request = try ChatRequestDecoder.decode(Data(json.utf8))
        XCTAssertEqual(request.stopSequences, ["DONE"])
    }

    func testDecodeRejectsEmptyMessages() {
        let json = #"""
        { "model": "m", "messages": [] }
        """#
        XCTAssertThrowsError(try ChatRequestDecoder.decode(Data(json.utf8))) { error in
            guard let e = error as? ChatRequestDecoderError else {
                return XCTFail("wrong error type")
            }
            XCTAssertEqual(e, .emptyMessages)
        }
    }

    func testDecodeRejectsMissingModel() {
        let json = #"""
        { "messages": [{"role":"user","content":"x"}] }
        """#
        XCTAssertThrowsError(try ChatRequestDecoder.decode(Data(json.utf8)))
    }
}
```

**Step 2: Run, verify fail**

```
swift test --filter MLXLMServerTests.ChatRequestDecoderTests
```

Expected: FAIL — `ChatRequestDecoder` does not exist.

**Step 3: Implement the decoder**

Create `Libraries/MLXLMServer/HTTP/ChatRequestDecoder.swift`:

```swift
import Foundation

public enum ChatRequestDecoderError: Error, Equatable {
    case invalidJSON(String)
    case missingField(String)
    case emptyMessages
    case invalidStopField
}

public enum ChatRequestDecoder {
    private struct Payload: Decodable {
        let model: String
        let messages: [ChatMessage]
        let maxTokens: Int?
        let stream: Bool?
        let stop: StopField?

        enum CodingKeys: String, CodingKey {
            case model, messages, stream, stop
            case maxTokens = "max_tokens"
        }
    }

    private enum StopField: Decodable {
        case single(String)
        case list([String])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let one = try? container.decode(String.self) {
                self = .single(one)
            } else if let many = try? container.decode([String].self) {
                self = .list(many)
            } else {
                throw ChatRequestDecoderError.invalidStopField
            }
        }

        var values: Set<String> {
            switch self {
            case .single(let s): return [s]
            case .list(let xs): return Set(xs)
            }
        }
    }

    public static func decode(_ data: Data, defaultMaxTokens: Int = 256) throws -> ChatRequest {
        let payload: Payload
        do {
            payload = try JSONDecoder().decode(Payload.self, from: data)
        } catch {
            throw ChatRequestDecoderError.invalidJSON(String(describing: error))
        }
        guard !payload.messages.isEmpty else {
            throw ChatRequestDecoderError.emptyMessages
        }
        return ChatRequest(
            modelID: payload.model,
            messages: payload.messages,
            maxTokens: payload.maxTokens ?? defaultMaxTokens,
            stopSequences: payload.stop?.values ?? [],
            stream: payload.stream ?? false
        )
    }
}
```

**Step 4: Run test, verify pass**

```
swift test --filter MLXLMServerTests.ChatRequestDecoderTests
```

Expected: PASS, 5 tests.

**Step 5: Commit**

```
git add Libraries/MLXLMServer/HTTP/ChatRequestDecoder.swift \
        Tests/MLXLMServerTests/ChatRequestDecoderTests.swift
git commit -S -m "feat(server): ChatRequestDecoder for OpenAI wire format"
```

---

### Task 2.2: Non-streaming `POST /v1/chat/completions`

**Files:**
- Modify: `Libraries/MLXLMServer/HTTP/MLXLMHTTPHandler.swift`
- Create: `Tests/MLXLMServerTests/ChatCompletionsNonStreamingTests.swift`

**Step 1: Write the failing test**

Create `Tests/MLXLMServerTests/ChatCompletionsNonStreamingTests.swift`:

```swift
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
```

**Step 2: Run, verify fail**

```
swift test --filter MLXLMServerTests.ChatCompletionsNonStreamingTests
```

Expected: FAIL — routes to 404.

**Step 3: Wire the handler**

In `Libraries/MLXLMServer/HTTP/MLXLMHTTPHandler.swift`:

- In `route(context:head:)`, add before `default`:
  ```swift
  case (.POST, "/v1/chat/completions"):
      handleChatCompletions(context: context, head: head)
  ```

- Add new method after `handleListModels`:

  ```swift
  private func handleChatCompletions(context: ChannelHandlerContext, head: HTTPRequestHead) {
      let eventLoop = context.eventLoop
      let contextBox = NIOLoopBound(context, eventLoop: eventLoop)
      let headBox = NIOLoopBound(head, eventLoop: eventLoop)
      let engine = self.engine

      let bodyBytes = Data(requestBody.readableBytesView)

      Task {
          let chatRequest: ChatRequest
          do {
              chatRequest = try ChatRequestDecoder.decode(bodyBytes)
          } catch {
              let errorBody = #"{"error":{"message":"\#(String(describing: error))","type":"invalid_request_error"}}"#
              eventLoop.execute {
                  self.respondJSON(
                      context: contextBox.value,
                      head: headBox.value,
                      status: .badRequest,
                      body: errorBody
                  )
              }
              return
          }

          // consume stream, aggregate all fragments
          var text = ""
          var finishReason: FinishReason?
          var usage: Usage?
          do {
              for try await delta in engine.generate(chatRequest) {
                  text += delta.textFragments.joined()
                  if let reason = delta.finishReason { finishReason = reason }
                  if let u = delta.usage { usage = u }
              }
          } catch {
              let errorBody = #"{"error":{"message":"generation failed","type":"server_error"}}"#
              eventLoop.execute {
                  self.respondJSON(
                      context: contextBox.value,
                      head: headBox.value,
                      status: .internalServerError,
                      body: errorBody
                  )
              }
              return
          }

          let choice: [String: Any] = [
              "index": 0,
              "message": [
                  "role": "assistant",
                  "content": text,
              ],
              "finish_reason": (finishReason ?? .stop).rawValue,
          ]
          var payload: [String: Any] = [
              "id": "chatcmpl-\(UUID().uuidString)",
              "object": "chat.completion",
              "created": Int(Date().timeIntervalSince1970),
              "model": chatRequest.modelID,
              "choices": [choice],
          ]
          if let usage {
              payload["usage"] = [
                  "prompt_tokens": usage.promptTokens,
                  "completion_tokens": usage.completionTokens,
                  "total_tokens": usage.totalTokens,
              ]
          }
          let encoded = (try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])) ?? Data()
          let json = String(data: encoded, encoding: .utf8) ?? "{}"

          eventLoop.execute {
              self.respondJSON(
                  context: contextBox.value,
                  head: headBox.value,
                  status: .ok,
                  body: json
              )
          }
      }
  }
  ```

**Step 4: Run test, verify pass**

```
swift test --filter MLXLMServerTests.ChatCompletionsNonStreamingTests
```

Expected: PASS, 2 tests.

**Step 5: Full suite check**

```
swift test --filter MLXLMServerTests
```

Expected: all green (health + 404 + models + 2 decoder tests + 2 non-stream = 7+).

**Step 6: Commit**

```
git add Libraries/MLXLMServer/HTTP/MLXLMHTTPHandler.swift \
        Tests/MLXLMServerTests/ChatCompletionsNonStreamingTests.swift
git commit -S -m "feat(server): POST /v1/chat/completions non-streaming"
```

---

## Phase 3 — SSE streaming

### Task 3.1: `SSEFrame` writer (pure logic, unit tested)

**Files:**
- Create: `Libraries/MLXLMServer/HTTP/SSEFrame.swift`
- Create: `Tests/MLXLMServerTests/SSEFrameTests.swift`

**Step 1: Write the failing test**

`Tests/MLXLMServerTests/SSEFrameTests.swift`:

```swift
import XCTest
@testable import MLXLMServer

final class SSEFrameTests: XCTestCase {
    func testFrameWrapsJSONInDataPrefix() {
        let frame = SSEFrame.data(#"{"a":1}"#)
        XCTAssertEqual(frame, #"data: {"a":1}\#n\#n"#)
    }

    func testDoneMarker() {
        XCTAssertEqual(SSEFrame.done, "data: [DONE]\n\n")
    }

    func testChatCompletionChunkShape() {
        let chunk = SSEFrame.chatCompletionChunk(
            id: "chatcmpl-x",
            model: "m",
            created: 42,
            contentDelta: "hello",
            finishReason: nil
        )
        XCTAssertTrue(chunk.hasPrefix("data: "))
        XCTAssertTrue(chunk.hasSuffix("\n\n"))
        let jsonStart = chunk.index(chunk.startIndex, offsetBy: "data: ".count)
        let jsonEnd = chunk.index(chunk.endIndex, offsetBy: -2)
        let jsonString = String(chunk[jsonStart..<jsonEnd])
        let obj = try! JSONSerialization.jsonObject(with: Data(jsonString.utf8)) as! [String: Any]
        XCTAssertEqual(obj["object"] as? String, "chat.completion.chunk")
        XCTAssertEqual(obj["model"] as? String, "m")
        let choices = obj["choices"] as! [[String: Any]]
        let delta = choices[0]["delta"] as! [String: Any]
        XCTAssertEqual(delta["content"] as? String, "hello")
    }

    func testFinalChunkCarriesFinishReasonAndEmptyDelta() {
        let chunk = SSEFrame.chatCompletionChunk(
            id: "chatcmpl-x",
            model: "m",
            created: 42,
            contentDelta: nil,
            finishReason: "stop"
        )
        let jsonStart = chunk.index(chunk.startIndex, offsetBy: "data: ".count)
        let jsonEnd = chunk.index(chunk.endIndex, offsetBy: -2)
        let obj = try! JSONSerialization.jsonObject(
            with: Data(String(chunk[jsonStart..<jsonEnd]).utf8)
        ) as! [String: Any]
        let choices = obj["choices"] as! [[String: Any]]
        XCTAssertEqual(choices[0]["finish_reason"] as? String, "stop")
        let delta = choices[0]["delta"] as! [String: Any]
        XCTAssertTrue(delta.isEmpty)
    }
}
```

**Step 2: Run, verify fail**

```
swift test --filter MLXLMServerTests.SSEFrameTests
```

Expected: FAIL — type missing.

**Step 3: Implement `SSEFrame`**

`Libraries/MLXLMServer/HTTP/SSEFrame.swift`:

```swift
import Foundation

public enum SSEFrame {
    public static let done = "data: [DONE]\n\n"

    public static func data(_ jsonBody: String) -> String {
        "data: \(jsonBody)\n\n"
    }

    /// Produces one `chat.completion.chunk` frame. Pass `contentDelta` nil on the
    /// final frame and `finishReason` nil on intermediate frames.
    public static func chatCompletionChunk(
        id: String,
        model: String,
        created: Int,
        contentDelta: String?,
        finishReason: String?,
        includeAssistantRole: Bool = false
    ) -> String {
        var delta: [String: Any] = [:]
        if includeAssistantRole { delta["role"] = "assistant" }
        if let c = contentDelta { delta["content"] = c }

        var choice: [String: Any] = [
            "index": 0,
            "delta": delta,
        ]
        choice["finish_reason"] = finishReason as Any? ?? NSNull()

        let payload: [String: Any] = [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [choice],
        ]

        let encoded = (try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])) ?? Data()
        let json = String(data: encoded, encoding: .utf8) ?? "{}"
        return data(json)
    }
}
```

**Step 4: Run, verify pass**

```
swift test --filter MLXLMServerTests.SSEFrameTests
```

Expected: PASS, 4 tests.

**Step 5: Commit**

```
git add Libraries/MLXLMServer/HTTP/SSEFrame.swift \
        Tests/MLXLMServerTests/SSEFrameTests.swift
git commit -S -m "feat(server): SSEFrame writer for OpenAI chunks"
```

---

### Task 3.2: Streaming `POST /v1/chat/completions`

**Files:**
- Modify: `Libraries/MLXLMServer/HTTP/MLXLMHTTPHandler.swift`
- Create: `Tests/MLXLMServerTests/ChatCompletionsStreamingTests.swift`

**Step 1: Write the failing test**

`Tests/MLXLMServerTests/ChatCompletionsStreamingTests.swift`:

```swift
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
                ChatDelta(textFragments: [], finishReason: .stop, usage: Usage(promptTokens: 1, completionTokens: 3)),
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
        let reconstructed = contentChunks.compactMap { line -> String? in
            let jsonStart = line.index(line.startIndex, offsetBy: "data: ".count)
            let data = Data(line[jsonStart...].utf8)
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = obj["choices"] as? [[String: Any]],
                  let delta = choices.first?["delta"] as? [String: Any]
            else { return nil }
            return delta["content"] as? String
        }.joined()
        XCTAssertEqual(reconstructed, "hello world")
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
```

**Step 2: Run, verify fail**

```
swift test --filter MLXLMServerTests.ChatCompletionsStreamingTests
```

Expected: FAIL — current handler always batches.

**Step 3: Branch the handler on `chatRequest.stream`**

In `MLXLMHTTPHandler.swift`, change `handleChatCompletions` so that when `chatRequest.stream == true` it takes the SSE path. Replace the "consume stream, aggregate" block with:

```swift
if chatRequest.stream {
    // emit SSE headers, then stream frames
    let chatID = "chatcmpl-\(UUID().uuidString)"
    let createdTs = Int(Date().timeIntervalSince1970)
    let model = chatRequest.modelID

    eventLoop.execute {
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "connection", value: "keep-alive")
        contextBox.value.writeAndFlush(
            self.wrapOutboundOut(.head(HTTPResponseHead(version: headBox.value.version, status: .ok, headers: headers))),
            promise: nil
        )
    }

    var firstChunkEmitted = false
    var finishReason: FinishReason?
    do {
        for try await delta in engine.generate(chatRequest) {
            for fragment in delta.textFragments where !fragment.isEmpty {
                let frame = SSEFrame.chatCompletionChunk(
                    id: chatID,
                    model: model,
                    created: createdTs,
                    contentDelta: fragment,
                    finishReason: nil,
                    includeAssistantRole: !firstChunkEmitted
                )
                firstChunkEmitted = true
                eventLoop.execute {
                    self.writeSSEFrame(context: contextBox.value, frame: frame)
                }
            }
            if let reason = delta.finishReason { finishReason = reason }
        }
    } catch {
        // best-effort: emit finishReason=stop anyway
    }

    let finalFrame = SSEFrame.chatCompletionChunk(
        id: chatID,
        model: model,
        created: createdTs,
        contentDelta: nil,
        finishReason: (finishReason ?? .stop).rawValue,
        includeAssistantRole: false
    )
    eventLoop.execute {
        self.writeSSEFrame(context: contextBox.value, frame: finalFrame)
        self.writeSSEFrame(context: contextBox.value, frame: SSEFrame.done)
        _ = contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)))
        contextBox.value.close(promise: nil)
    }
    return
}

// non-stream path: existing code that aggregates text and calls respondJSON
```

Add helper near `respondJSON`:

```swift
private func writeSSEFrame(context: ChannelHandlerContext, frame: String) {
    let bytes = Data(frame.utf8)
    var buffer = context.channel.allocator.buffer(capacity: bytes.count)
    buffer.writeBytes(bytes)
    context.writeAndFlush(wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
}
```

**Step 4: Run, verify pass**

```
swift test --filter MLXLMServerTests.ChatCompletionsStreamingTests
```

Expected: PASS.

**Step 5: Full suite**

```
swift test --filter MLXLMServerTests
```

Expected: 10+ tests green, no regressions.

**Step 6: Commit**

```
git add Libraries/MLXLMServer/HTTP/MLXLMHTTPHandler.swift \
        Tests/MLXLMServerTests/ChatCompletionsStreamingTests.swift
git commit -S -m "feat(server): SSE streaming for chat completions"
```

---

## Phase 4 — Single-flight 409 queue

### Task 4.1: `SingleFlightGate` actor

**Files:**
- Create: `Libraries/MLXLMServer/HTTP/SingleFlightGate.swift`
- Create: `Tests/MLXLMServerTests/SingleFlightGateTests.swift`

**Step 1: Write the failing test**

`Tests/MLXLMServerTests/SingleFlightGateTests.swift`:

```swift
import XCTest
@testable import MLXLMServer

final class SingleFlightGateTests: XCTestCase {
    func testAcquireReleaseAllowsSequential() async throws {
        let gate = SingleFlightGate()
        try await gate.acquire()
        await gate.release()
        try await gate.acquire()
        await gate.release()
    }

    func testAcquireWhileHeldThrows() async throws {
        let gate = SingleFlightGate()
        try await gate.acquire()
        do {
            try await gate.acquire()
            XCTFail("second acquire should have thrown")
        } catch SingleFlightError.busy {
            // expected
        }
        await gate.release()
    }

    func testReleaseTwiceIsIdempotent() async throws {
        let gate = SingleFlightGate()
        try await gate.acquire()
        await gate.release()
        await gate.release()   // no crash
    }
}
```

**Step 2: Run, verify fail**

```
swift test --filter MLXLMServerTests.SingleFlightGateTests
```

Expected: FAIL — type missing.

**Step 3: Implement**

`Libraries/MLXLMServer/HTTP/SingleFlightGate.swift`:

```swift
import Foundation

public enum SingleFlightError: Error, Equatable {
    case busy
}

public actor SingleFlightGate {
    private var held: Bool = false

    public init() {}

    public func acquire() async throws {
        guard !held else { throw SingleFlightError.busy }
        held = true
    }

    public func release() {
        held = false
    }
}
```

**Step 4: Run, verify pass**

```
swift test --filter MLXLMServerTests.SingleFlightGateTests
```

Expected: PASS, 3 tests.

**Step 5: Commit**

```
git add Libraries/MLXLMServer/HTTP/SingleFlightGate.swift \
        Tests/MLXLMServerTests/SingleFlightGateTests.swift
git commit -S -m "feat(server): SingleFlightGate actor"
```

---

### Task 4.2: Wire the gate into `/v1/chat/completions`

**Files:**
- Modify: `Libraries/MLXLMServer/HTTP/MLXLMHTTPServer.swift` (one shared gate per server)
- Modify: `Libraries/MLXLMServer/HTTP/MLXLMHTTPHandler.swift`
- Create: `Tests/MLXLMServerTests/SingleFlightConcurrencyTests.swift`

**Step 1: Write the failing test**

`Tests/MLXLMServerTests/SingleFlightConcurrencyTests.swift`:

```swift
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

        async let firstTask: (Data, URLResponse) = URLSession.shared.data(for: request)
        // give the first request a moment to land and acquire the gate
        try await Task.sleep(nanoseconds: 100_000_000)
        async let secondTask: (Data, URLResponse) = URLSession.shared.data(for: request)

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
```

**Step 2: Run, verify fail**

```
swift test --filter MLXLMServerTests.SingleFlightConcurrencyTests
```

Expected: FAIL — both requests return 200.

**Step 3: Add shared gate to the server**

In `MLXLMHTTPServer.swift`, add an immutable property:

```swift
private let gate = SingleFlightGate()
```

Pass it into the handler init:

```swift
channel.pipeline.addHandler(MLXLMHTTPHandler(engine: engine, gate: gate))
```

In `MLXLMHTTPHandler.swift`, accept the gate:

```swift
private let gate: SingleFlightGate
init(engine: InferenceEngine, gate: SingleFlightGate) {
    self.engine = engine
    self.gate = gate
}
```

In `handleChatCompletions`, acquire at the start of the `Task` block. On failure, respond 409:

```swift
Task {
    do {
        try await gate.acquire()
    } catch {
        let body = #"{"error":{"message":"another request is in flight","type":"rate_limit_error"}}"#
        eventLoop.execute {
            self.respondJSON(
                context: contextBox.value, head: headBox.value,
                status: .conflict, body: body
            )
        }
        return
    }
    defer { Task { await gate.release() } }

    // existing decoder + stream-vs-batch logic …
}
```

**Step 4: Run, verify pass**

```
swift test --filter MLXLMServerTests.SingleFlightConcurrencyTests
```

Expected: PASS.

**Step 5: Full suite regression check**

```
swift test --filter MLXLMServerTests
```

Expected: all green (single-flight must not break earlier tests — the gate is per-server-instance and each test gets a fresh server).

**Step 6: Commit**

```
git add Libraries/MLXLMServer/HTTP/MLXLMHTTPServer.swift \
        Libraries/MLXLMServer/HTTP/MLXLMHTTPHandler.swift \
        Tests/MLXLMServerTests/SingleFlightConcurrencyTests.swift
git commit -S -m "feat(server): single-flight 409 on concurrent chat completions"
```

---

## Phase 5 — `BaselineEngine` (real AR inference via MLXLLM)

**Note:** This phase has higher integration risk. Actual model loading and MLX forward passes require metallib at runtime. On a build-only host, tests will SKIP (follow the `runtimeMetallibAvailable()` pattern from dflash-mlx, or the equivalent in this repo). Identify that helper first — search for `runtimeMetallibAvailable` across `Tests/`. If not present, add a minimal `MLXRuntimeGuard.swift` utility in `Tests/MLXLMServerTests/` that returns `false` unless `MTLCreateSystemDefaultDevice()` succeeds AND a tiny MLX op round-trips to CPU.

### Task 5.1: `BaselineEngineConfiguration`

**Files:**
- Create: `Libraries/MLXLMServer/Engine/BaselineEngineConfiguration.swift`
- Create: `Tests/MLXLMServerTests/BaselineEngineConfigurationTests.swift`

**Step 1: Write the failing test**

```swift
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
```

**Step 2: Run, verify fail** (`swift test --filter MLXLMServerTests.BaselineEngineConfigurationTests`)

**Step 3: Implement**

```swift
import Foundation

public struct BaselineEngineConfiguration: Sendable {
    public let modelID: String
    public let defaultMaxTokens: Int
    public let contextWindow: Int

    public init(
        modelID: String,
        defaultMaxTokens: Int = 256,
        contextWindow: Int = 4096
    ) {
        self.modelID = modelID
        self.defaultMaxTokens = defaultMaxTokens
        self.contextWindow = contextWindow
    }
}
```

**Step 4: Run, verify pass**

**Step 5: Commit**

```
git add Libraries/MLXLMServer/Engine/BaselineEngineConfiguration.swift \
        Tests/MLXLMServerTests/BaselineEngineConfigurationTests.swift
git commit -S -m "feat(server): BaselineEngineConfiguration"
```

---

### Task 5.2: Chat-template renderer wrapper

**Files:**
- Create: `Libraries/MLXLMServer/Engine/ChatTemplate.swift`
- Create: `Tests/MLXLMServerTests/ChatTemplateTests.swift`

This task wraps `Tokenizers.applyChatTemplate` so `BaselineEngine` doesn't touch tokenizer internals directly.

**Step 1: Investigate** — before writing, grep `Libraries/MLXLMCommon/` and `Libraries/MLXLLM/` for `applyChatTemplate` usages. The wrapper must forward to whichever public API already exists. If `swift-transformers` is not a direct dependency of `MLXLMServer`, add it: Package.swift target `MLXLMServer` → add `.product(name: "Tokenizers", package: "swift-transformers")`.

**Step 2: Failing test** (skips if tokenizer unavailable):

```swift
import XCTest
@testable import MLXLMServer

final class ChatTemplateTests: XCTestCase {
    func testRendersMessagesToPromptString() throws {
        // Use a tokenizer stub with a trivial template — not a real Qwen load.
        let template = ChatTemplate.literal(
            "{% for m in messages %}{{ m.role }}:{{ m.content }}\n{% endfor %}"
        )
        let rendered = try template.render(messages: [
            ChatMessage(role: "system", content: "sys"),
            ChatMessage(role: "user", content: "hi"),
        ])
        XCTAssertEqual(rendered, "system:sys\nuser:hi\n")
    }
}
```

**Step 3: Implement**

```swift
import Foundation
import Jinja   // from swift-jinja

public struct ChatTemplate: Sendable {
    public let source: String

    public static func literal(_ template: String) -> ChatTemplate {
        ChatTemplate(source: template)
    }

    public func render(messages: [ChatMessage], addGenerationPrompt: Bool = false) throws -> String {
        let context: [String: Any] = [
            "messages": messages.map { ["role": $0.role, "content": $0.content] },
            "add_generation_prompt": addGenerationPrompt,
        ]
        let jinjaTemplate = try Template(source)
        return try jinjaTemplate.render(context)
    }
}
```

Add `Jinja` product to `MLXLMServer` dependencies in Package.swift:
`.product(name: "Jinja", package: "swift-jinja"),`

And declare the `swift-jinja` dependency at top:
`.package(url: "https://github.com/johnmai-dev/swift-jinja", from: "1.0.0")` — match the version already pinned by swift-transformers via `Package.resolved` (check it first).

**Step 4: Run, verify pass**

**Step 5: Commit**

```
git add Package.swift Libraries/MLXLMServer/Engine/ChatTemplate.swift \
        Tests/MLXLMServerTests/ChatTemplateTests.swift
git commit -S -m "feat(server): ChatTemplate wrapper over swift-jinja"
```

---

### Task 5.3: `BaselineEngine` skeleton (conformance + availableModels + health only)

**Files:**
- Create: `Libraries/MLXLMServer/Engine/BaselineEngine.swift`
- Create: `Tests/MLXLMServerTests/BaselineEngineSkeletonTests.swift`

**Step 1: Failing test**

```swift
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
```

**Step 2: Run, verify fail**

**Step 3: Implement**

```swift
import Foundation

public final class BaselineEngine: InferenceEngine, @unchecked Sendable {
    public let configuration: BaselineEngineConfiguration
    private let start: Date
    private var loaded: Bool = false

    public init(configuration: BaselineEngineConfiguration) {
        self.configuration = configuration
        self.start = Date()
    }

    public func availableModels() async -> [ModelInfo] {
        [ModelInfo(
            id: configuration.modelID,
            created: Int(start.timeIntervalSince1970),
            ownedBy: "mlx-swift-lm"
        )]
    }

    public func health() async -> EngineHealth {
        EngineHealth(
            ready: loaded,
            modelIDs: [configuration.modelID],
            uptimeSeconds: Date().timeIntervalSince(start)
        )
    }

    public func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: BaselineEngineError.notYetImplemented)
        }
    }
}

public enum BaselineEngineError: Error {
    case notYetImplemented
    case modelLoadFailed(String)
    case tokenizationFailed(String)
}
```

**Step 4: Run, verify pass**

**Step 5: Commit**

```
git add Libraries/MLXLMServer/Engine/BaselineEngine.swift \
        Tests/MLXLMServerTests/BaselineEngineSkeletonTests.swift
git commit -S -m "feat(server): BaselineEngine skeleton (protocol conformance only)"
```

---

### Task 5.4: Model + tokenizer loader (metallib-gated)

**Files:**
- Modify: `Libraries/MLXLMServer/Engine/BaselineEngine.swift`
- Create: `Tests/MLXLMServerTests/BaselineEngineLoaderTests.swift`

**Step 1: Investigate** — identify the existing model-loading entry point in this repo. Grep:
```
grep -RE "loadContainer|loadModel|ModelFactory" Libraries/MLXLMCommon Libraries/MLXLLM
```

Record the exact public API surface in your report. The loader signature typically takes a `ModelConfiguration` (or a HuggingFace model ID string) and returns `ModelContainer` / `(ModelContext, Tokenizer)`. Reuse what exists; do NOT re-implement.

**Step 2: Failing test** (metallib-gated):

```swift
import XCTest
@testable import MLXLMServer

final class BaselineEngineLoaderTests: XCTestCase {
    func testLoadMarksHealthReady() async throws {
        try XCTSkipUnless(runtimeMetallibAvailable(), "MLX metallib unavailable")
        let engine = BaselineEngine(configuration: BaselineEngineConfiguration(
            modelID: "mlx-community/Qwen2.5-0.5B-Instruct-4bit"   // small, fast download
        ))
        try await engine.load()
        let health = await engine.health()
        XCTAssertTrue(health.ready)
    }
}
```

Add the `runtimeMetallibAvailable()` helper (or reuse the existing one) in a new file `Tests/MLXLMServerTests/MLXRuntimeGuard.swift`.

**Step 3: Implement `load()`**

In `BaselineEngine.swift`:

```swift
public func load() async throws {
    // Use the established ModelContainer loader from MLXLMCommon here.
    // Subagent: fill with concrete call discovered in Task 5.4 Step 1.
    // After successful load, store the container and tokenizer in `self`.
    self.loaded = true
}
```

Also add stored properties for `container` and `tokenizer` (concrete types from the investigation step).

**Step 4: Run (likely SKIPS locally)**

```
swift test --filter MLXLMServerTests.BaselineEngineLoaderTests
```

On a non-Metal host: SKIP. On a Metal host: PASS (may take ~30s first run due to model download).

**Step 5: Commit**

```
git add Libraries/MLXLMServer/Engine/BaselineEngine.swift \
        Tests/MLXLMServerTests/BaselineEngineLoaderTests.swift \
        Tests/MLXLMServerTests/MLXRuntimeGuard.swift
git commit -S -m "feat(server): BaselineEngine model loader via MLXLMCommon"
```

---

### Task 5.5: `BaselineEngine.generate` — tokenize, run MLXLLM, stream deltas

**Files:**
- Modify: `Libraries/MLXLMServer/Engine/BaselineEngine.swift`
- Create: `Tests/MLXLMServerTests/BaselineEngineGenerateTests.swift`

**Step 1: Investigate** — find the streaming generate API in `MLXLLM`. Grep:
```
grep -RE "public func generate|TokenIterator|GenerationStream" Libraries/MLXLLM
```

Write down the signature: whether it takes an `AsyncStream`, a closure callback, or returns a `TokenIterator`. Plan the bridge to `AsyncThrowingStream<ChatDelta, Error>` accordingly.

**Step 2: Failing test** (metallib-gated, end-to-end tiny generation):

```swift
import XCTest
@testable import MLXLMServer

final class BaselineEngineGenerateTests: XCTestCase {
    func testGenerateEmitsAssistantTextAndFinalDelta() async throws {
        try XCTSkipUnless(runtimeMetallibAvailable(), "MLX metallib unavailable")
        let engine = BaselineEngine(configuration: BaselineEngineConfiguration(
            modelID: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            defaultMaxTokens: 8
        ))
        try await engine.load()

        let request = ChatRequest(
            modelID: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            messages: [ChatMessage(role: "user", content: "Hi.")],
            maxTokens: 8
        )
        var textFragments: [String] = []
        var finishReason: FinishReason?
        for try await delta in engine.generate(request) {
            textFragments.append(contentsOf: delta.textFragments)
            if let reason = delta.finishReason { finishReason = reason }
        }
        XCTAssertFalse(textFragments.joined().isEmpty)
        XCTAssertNotNil(finishReason)
    }
}
```

**Step 3: Implement**

Replace the stub `generate` with (pseudocode — subagent fills concrete calls from Step 1):

```swift
public func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error> {
    AsyncThrowingStream { continuation in
        Task {
            do {
                guard loaded, let tokenizer = self.tokenizer else {
                    throw BaselineEngineError.modelLoadFailed("call load() first")
                }
                // 1. Apply chat template via tokenizer (Qwen3.5 template is in tokenizer_config.json)
                // 2. Encode to [Int] token IDs
                // 3. Call MLXLLM streaming generate with:
                //    - maxTokens: request.maxTokens (capped at config.contextWindow - prompt.count)
                //    - stopSequences: request.stopSequences ∪ model EOS
                // 4. For each emitted token (or batch), decode incrementally and yield a ChatDelta
                // 5. On finish, yield a final empty-fragments delta with finishReason + usage
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }
}
```

Use the existing MLXLLM streaming API (discovered in Step 1). Handle:
- BPE-safe incremental detokenization (keep a pending byte buffer)
- Task cancellation: check `Task.isCancelled` in the loop and `continuation.finish(throwing: CancellationError())` cleanly

**Step 4: Run** — SKIPS locally, PASS on Metal host

**Step 5: Commit**

```
git add Libraries/MLXLMServer/Engine/BaselineEngine.swift \
        Tests/MLXLMServerTests/BaselineEngineGenerateTests.swift
git commit -S -m "feat(server): BaselineEngine autoregressive generate via MLXLLM"
```

---

### Task 5.6: Final green-bar

**Step 1:** Full suite

```
swift test --filter MLXLMServerTests
```

Expected: 20+ tests; metallib-gated ones SKIP on non-Metal host; no failures.

**Step 2:** Full repo suite (regression check — nothing else should be affected)

```
swift test
```

Expected: all existing MLXLMTests etc. stay green.

**Step 3:** Commit milestone marker (empty, signed)

```
git commit -S --allow-empty -m "chore(server): MLXLMServer OpenAI endpoints milestone complete"
```

---

## Verification Checklist (post-plan)

- [ ] `swift test --filter MLXLMServerTests` green on non-Metal host (metallib-gated tests SKIP)
- [ ] `swift test` green across the repo
- [ ] `/v1/models` returns `{"object":"list","data":[…]}` with at least one entry
- [ ] `/v1/chat/completions` non-stream returns full `chat.completion` object
- [ ] `/v1/chat/completions` stream emits `text/event-stream`, one chunk per fragment, terminates with `data: [DONE]`
- [ ] Concurrent chat request returns HTTP 409
- [ ] `BaselineEngine` conforms to `InferenceEngine` and compiles
- [ ] `BaselineEngine.load()` + `.generate()` path works with a tiny real model on a Metal host (verified manually with the user's setup)
- [ ] All commits signed (SSH); no `-c commit.gpgsign=false` used

---

## Total

**5 phases, 12 tasks, ~14 atomic commits.** Expected execution time: 2–3 focused sessions. Highest integration risk is Phase 5 (real model loading) — isolate that from Phases 1–4 so a green bar is provable before touching MLX.
