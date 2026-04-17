import Foundation
import NIO
import NIOCore
import NIOHTTP1

/// Per-connection HTTP handler. Accumulates a request, routes it, writes
/// a single response. Streaming endpoints will live here later.
final class MLXLMHTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let engine: InferenceEngine
    private var requestHead: HTTPRequestHead?
    private var requestBody = ByteBuffer()

    init(engine: InferenceEngine) {
        self.engine = engine
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

        let bodyBytes = Data(requestBody.readableBytesView)

        Task {
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
