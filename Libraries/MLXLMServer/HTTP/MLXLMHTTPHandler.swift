import Foundation
import NIO
import NIOCore
import NIOHTTP1

/// Monotonic "last touched" timestamp used by the heartbeat race to decide
/// whether enough idle time has elapsed to emit a keepalive. Kept as an
/// actor so touches and reads from the delta-consumer and heartbeat tasks
/// never interleave; monotonic nanoseconds via `DispatchTime.now()`.
actor AtomicInstant {
    private var lastNs: UInt64

    init() { self.lastNs = DispatchTime.now().uptimeNanoseconds }

    func touch() { lastNs = DispatchTime.now().uptimeNanoseconds }

    func elapsedNanoseconds() -> UInt64 {
        let now = DispatchTime.now().uptimeNanoseconds
        return now >= lastNs ? now - lastNs : 0
    }
}

/// Per-connection HTTP handler. Accumulates a request, routes it, writes
/// a single response. Streaming endpoints will live here later.
final class MLXLMHTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let engine: InferenceEngine
    private let gate: SingleFlightGate
    private let healthResponder: MLXLMHTTPServer.HealthResponder?
    private let heartbeatInterval: Int?
    private var requestHead: HTTPRequestHead?
    private var requestBody = ByteBuffer()

    init(
        engine: InferenceEngine,
        gate: SingleFlightGate,
        healthResponder: MLXLMHTTPServer.HealthResponder? = nil,
        heartbeatInterval: Int? = nil
    ) {
        self.engine = engine
        self.gate = gate
        self.healthResponder = healthResponder
        self.heartbeatInterval = heartbeatInterval
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let part = unwrapInboundIn(data)
        switch part {
        case .head(let head):
            requestHead = head
            requestBody.clear()
        case .body(var buffer):
            requestBody.writeBuffer(&buffer)
        case .end:
            guard let head = requestHead else { return }
            route(context: context, head: head)
            requestHead = nil
            requestBody.clear()
        }
    }

    private func route(context: ChannelHandlerContext, head: HTTPRequestHead) {
        let path = head.uri.split(separator: "?", maxSplits: 1).first.map(String.init) ?? head.uri
        switch (head.method, path) {
        case (.GET, "/health"):
            handleHealth(context: context, head: head)
        case (.GET, "/v1/models"):
            handleListModels(context: context, head: head)
        case (.POST, "/v1/chat/completions"):
            handleChatCompletions(context: context, head: head)
        default:
            respondJSON(
                context: context,
                head: head,
                status: .notFound,
                body: #"{"error":{"message":"not found","type":"invalid_request_error"}}"#
            )
        }
    }

    private func handleHealth(context: ChannelHandlerContext, head: HTTPRequestHead) {
        let eventLoop = context.eventLoop
        let contextBox = NIOLoopBound(context, eventLoop: eventLoop)
        let headBox = NIOLoopBound(head, eventLoop: eventLoop)

        if let responder = self.healthResponder {
            Task {
                let response = await responder()
                let status = HTTPResponseStatus(statusCode: response.status)
                let json = String(data: response.body, encoding: .utf8) ?? "{}"
                eventLoop.execute {
                    self.respondJSON(
                        context: contextBox.value,
                        head: headBox.value,
                        status: status,
                        body: json
                    )
                }
            }
            return
        }

        let engine = self.engine

        Task {
            let health = await engine.health()
            let payload: [String: Any] = [
                "status": health.ready ? "ready" : "not_ready",
                "model_ids": health.modelIDs,
                "uptime_s": health.uptimeSeconds,
            ]
            let data = (try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])) ?? Data()
            let json = String(data: data, encoding: .utf8) ?? "{}"
            let status: HTTPResponseStatus = health.ready ? .ok : .serviceUnavailable

            eventLoop.execute {
                self.respondJSON(
                    context: contextBox.value,
                    head: headBox.value,
                    status: status,
                    body: json
                )
            }
        }
    }

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

    private func handleChatCompletions(context: ChannelHandlerContext, head: HTTPRequestHead) {
        let eventLoop = context.eventLoop
        let contextBox = NIOLoopBound(context, eventLoop: eventLoop)
        let headBox = NIOLoopBound(head, eventLoop: eventLoop)
        let engine = self.engine
        let gate = self.gate

        let bodyBytes = Data(requestBody.readableBytesView)

        Task {
            do {
                try await gate.acquire()
            } catch SingleFlightError.busy {
                let busyPayload: [String: Any] = [
                    "error": [
                        "message": "another request is in flight",
                        "type": "rate_limit_error",
                    ],
                ]
                let encodedBusy = (try? JSONSerialization.data(withJSONObject: busyPayload, options: [.sortedKeys])) ?? Data()
                let busyBody = String(data: encodedBusy, encoding: .utf8)
                    ?? #"{"error":{"message":"another request is in flight","type":"rate_limit_error"}}"#
                eventLoop.execute {
                    self.respondJSON(
                        context: contextBox.value,
                        head: headBox.value,
                        status: .conflict,
                        body: busyBody
                    )
                }
                return
            } catch {
                let errorPayload: [String: Any] = [
                    "error": [
                        "message": String(describing: error),
                        "type": "server_error",
                    ],
                ]
                let encodedError = (try? JSONSerialization.data(withJSONObject: errorPayload, options: [.sortedKeys])) ?? Data()
                let errorBody = String(data: encodedError, encoding: .utf8)
                    ?? #"{"error":{"message":"internal error","type":"server_error"}}"#
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
            defer { Task { await gate.release() } }

            let chatRequest: ChatRequest
            do {
                chatRequest = try ChatRequestDecoder.decode(bodyBytes)
            } catch {
                let errorPayload: [String: Any] = [
                    "error": [
                        "message": String(describing: error),
                        "type": "invalid_request_error",
                    ],
                ]
                let encodedError = (try? JSONSerialization.data(withJSONObject: errorPayload, options: [.sortedKeys])) ?? Data()
                let errorBody = String(data: encodedError, encoding: .utf8)
                    ?? #"{"error":{"message":"invalid request","type":"invalid_request_error"}}"#
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

            if chatRequest.stream {
                let chatID = "chatcmpl-\(UUID().uuidString)"
                let createdTs = Int(Date().timeIntervalSince1970)
                let model = chatRequest.modelID

                eventLoop.execute {
                    var headers = HTTPHeaders()
                    headers.add(name: "content-type", value: "text/event-stream")
                    headers.add(name: "cache-control", value: "no-cache")
                    headers.add(name: "connection", value: "keep-alive")
                    contextBox.value.writeAndFlush(
                        self.wrapOutboundOut(
                            .head(HTTPResponseHead(
                                version: headBox.value.version,
                                status: .ok,
                                headers: headers
                            ))
                        ),
                        promise: nil
                    )
                }

                var firstChunkEmitted = false
                var streamFinishReason: FinishReason?
                let heartbeatIntervalSeconds = self.heartbeatInterval
                do {
                    let events = Self.mergeDeltasWithHeartbeat(
                        stream: engine.generate(chatRequest),
                        heartbeatInterval: heartbeatIntervalSeconds
                    )
                    for try await event in events {
                        switch event {
                        case .heartbeat:
                            eventLoop.execute {
                                self.writeSSEFrame(
                                    context: contextBox.value,
                                    frame: SSEFrame.keepalive
                                )
                            }
                        case .delta(let delta):
                            for fragment in delta.textFragments where !fragment.isEmpty {
                                let includeRole = !firstChunkEmitted
                                let frame = SSEFrame.chatCompletionChunk(
                                    id: chatID,
                                    model: model,
                                    created: createdTs,
                                    contentDelta: fragment,
                                    finishReason: nil,
                                    includeAssistantRole: includeRole
                                )
                                firstChunkEmitted = true
                                eventLoop.execute {
                                    self.writeSSEFrame(context: contextBox.value, frame: frame)
                                }
                            }
                            if let reason = delta.finishReason { streamFinishReason = reason }
                        }
                    }
                } catch {
                    // best-effort: emit finishReason=stop anyway
                }

                let finalFrame = SSEFrame.chatCompletionChunk(
                    id: chatID,
                    model: model,
                    created: createdTs,
                    contentDelta: nil,
                    finishReason: (streamFinishReason ?? .stop).rawValue,
                    includeAssistantRole: false
                )
                eventLoop.execute {
                    self.writeSSEFrame(context: contextBox.value, frame: finalFrame)
                    self.writeSSEFrame(context: contextBox.value, frame: SSEFrame.done)
                    contextBox.value.writeAndFlush(
                        self.wrapOutboundOut(.end(nil)),
                        promise: nil
                    )
                    contextBox.value.close(promise: nil)
                }
                return
            }

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

    /// Event emitted by `mergeDeltasWithHeartbeat`: either a `ChatDelta` from
    /// the engine or a heartbeat tick requesting a `: keepalive\n\n` SSE
    /// comment on the wire. Heartbeats reset each time a delta is observed.
    enum StreamEvent {
        case delta(ChatDelta)
        case heartbeat
    }

    /// Race the engine's delta stream against a periodic heartbeat timer.
    ///
    /// When `heartbeatInterval` is `nil` or `<= 0` this degrades to forwarding
    /// deltas unchanged (no heartbeat task spawned). Otherwise a sibling task
    /// yields `.heartbeat` events whenever `heartbeatInterval` seconds elapse
    /// without a delta. The timer resets on every observed delta. Errors from
    /// the inner stream propagate as `AsyncThrowingStream` failures.
    static func mergeDeltasWithHeartbeat(
        stream: AsyncThrowingStream<ChatDelta, Error>,
        heartbeatInterval: Int?
    ) -> AsyncThrowingStream<StreamEvent, Error> {
        guard let interval = heartbeatInterval, interval > 0 else {
            return AsyncThrowingStream { continuation in
                let task = Task {
                    do {
                        for try await delta in stream {
                            continuation.yield(.delta(delta))
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { _ in task.cancel() }
            }
        }

        let nanos = UInt64(interval) * 1_000_000_000
        // Actor is @unchecked Sendable so it can be shared with tasks that
        // outlive the call boundary without requiring `let` capture heroics.
        let lastDelta = AtomicInstant()
        return AsyncThrowingStream { continuation in
            let deltaTask = Task {
                do {
                    for try await delta in stream {
                        await lastDelta.touch()
                        continuation.yield(.delta(delta))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            let heartbeatTask = Task {
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: nanos)
                    if Task.isCancelled { break }
                    let idleNs = await lastDelta.elapsedNanoseconds()
                    if idleNs >= nanos {
                        continuation.yield(.heartbeat)
                        await lastDelta.touch()
                    }
                }
            }
            continuation.onTermination = { _ in
                deltaTask.cancel()
                heartbeatTask.cancel()
            }
        }
    }

    private func writeSSEFrame(context: ChannelHandlerContext, frame: String) {
        let bytes = Data(frame.utf8)
        var buffer = context.channel.allocator.buffer(capacity: bytes.count)
        buffer.writeBytes(bytes)
        context.writeAndFlush(wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
    }

    private func respondJSON(
        context: ChannelHandlerContext,
        head: HTTPRequestHead,
        status: HTTPResponseStatus,
        body: String
    ) {
        let data = Data(body.utf8)
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "application/json; charset=utf-8")
        headers.add(name: "content-length", value: "\(data.count)")
        if !head.isKeepAlive {
            headers.add(name: "connection", value: "close")
        }

        context.write(
            wrapOutboundOut(.head(HTTPResponseHead(version: head.version, status: status, headers: headers))),
            promise: nil
        )
        var buffer = context.channel.allocator.buffer(capacity: data.count)
        buffer.writeBytes(data)
        context.write(wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)

        let endPromise = context.eventLoop.makePromise(of: Void.self)
        context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: endPromise)
        if !head.isKeepAlive {
            endPromise.futureResult.whenComplete { _ in
                context.close(promise: nil)
            }
        }
    }
}
