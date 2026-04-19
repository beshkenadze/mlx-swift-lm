import Foundation
import NIO
import NIOHTTP1
import NIOPosix

/// HTTP front-end for an `InferenceEngine`. Binds a single `host:port`, speaks
/// OpenAI-compatible HTTP/1.1, and is intended for localhost-only use
/// (no auth, no TLS).
///
/// Currently implements `GET /health` only. Other endpoints respond 501.
/// Additional handlers (`/v1/models`, `/v1/chat/completions`) land in
/// follow-up commits.
public final class MLXLMHTTPServer: @unchecked Sendable {
    /// Response to `GET /health`. Implementations MUST return within ~10 ms.
    /// `status` is the HTTP status to emit (typically 200 for ready, 503 for
    /// not-ready); `body` is the raw JSON bytes written verbatim with
    /// `content-type: application/json; charset=utf-8`.
    public struct HealthResponse: Sendable {
        public let status: Int
        public let body: Data

        public init(status: Int, body: Data) {
            self.status = status
            self.body = body
        }
    }

    public typealias HealthResponder = @Sendable () async -> HealthResponse

    public let host: String
    public let port: Int
    private let engine: InferenceEngine
    private let healthResponder: HealthResponder?
    private let gate = SingleFlightGate()
    private let eventLoopGroup: MultiThreadedEventLoopGroup
    private var boundChannel: Channel?

    public init(
        engine: InferenceEngine,
        host: String = "127.0.0.1",
        port: Int = 8080,
        numberOfThreads: Int = 1,
        healthResponder: HealthResponder? = nil
    ) {
        self.engine = engine
        self.host = host
        self.port = port
        self.healthResponder = healthResponder
        self.eventLoopGroup = MultiThreadedEventLoopGroup(numberOfThreads: numberOfThreads)
    }

    /// Bind and block until the server is stopped. Call `stop()` from
    /// another task to terminate.
    public func run() throws {
        let engine = self.engine
        let gate = self.gate
        let healthResponder = self.healthResponder
        let bootstrap = ServerBootstrap(group: eventLoopGroup)
            .serverChannelOption(ChannelOptions.backlog, value: 8)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                channel.pipeline.configureHTTPServerPipeline(
                    withPipeliningAssistance: true,
                    withErrorHandling: true
                ).flatMap {
                    channel.pipeline.addHandler(
                        MLXLMHTTPHandler(
                            engine: engine,
                            gate: gate,
                            healthResponder: healthResponder
                        )
                    )
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 1)

        let channel = try bootstrap.bind(host: host, port: port).wait()
        self.boundChannel = channel
        try channel.closeFuture.wait()
    }

    /// Bind and return a port assigned by the OS. Useful when `port: 0` is
    /// passed and the caller needs the actual port for test clients.
    /// The returned future resolves once the server is bound; pair with
    /// `stop()` to tear down.
    public func bindAndRun() throws -> (Channel, Int) {
        let engine = self.engine
        let gate = self.gate
        let healthResponder = self.healthResponder
        let bootstrap = ServerBootstrap(group: eventLoopGroup)
            .serverChannelOption(ChannelOptions.backlog, value: 8)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                channel.pipeline.configureHTTPServerPipeline(
                    withPipeliningAssistance: true,
                    withErrorHandling: true
                ).flatMap {
                    channel.pipeline.addHandler(
                        MLXLMHTTPHandler(
                            engine: engine,
                            gate: gate,
                            healthResponder: healthResponder
                        )
                    )
                }
            }

        let channel = try bootstrap.bind(host: host, port: port).wait()
        self.boundChannel = channel
        let actualPort = channel.localAddress?.port ?? port
        return (channel, actualPort)
    }

    public func stop() throws {
        try boundChannel?.close().wait()
        try eventLoopGroup.syncShutdownGracefully()
    }
}
