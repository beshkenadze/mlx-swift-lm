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
