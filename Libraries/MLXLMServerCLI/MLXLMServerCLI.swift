import Foundation
import MLXLMServer

@main
struct MLXLMServerCLI {
    static func main() async throws {
        var model = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        var port = 8080
        var host = "127.0.0.1"
        var maxTokens = 256
        var showHelp = false

        let args = CommandLine.arguments.dropFirst()
        var iterator = args.makeIterator()
        while let arg = iterator.next() {
            switch arg {
            case "--model":
                guard let next = iterator.next() else { fatal("--model requires a value") }
                model = next
            case "--port":
                guard let next = iterator.next(), let p = Int(next) else { fatal("--port requires an integer") }
                port = p
            case "--host":
                guard let next = iterator.next() else { fatal("--host requires a value") }
                host = next
            case "--max-tokens":
                guard let next = iterator.next(), let n = Int(next) else { fatal("--max-tokens requires an integer") }
                maxTokens = n
            case "-h", "--help":
                showHelp = true
            default:
                FileHandle.standardError.write(Data("unknown argument: \(arg)\n".utf8))
                exit(2)
            }
        }

        if showHelp {
            printUsage()
            return
        }

        FileHandle.standardError.write(Data("loading model: \(model)\n".utf8))
        let engine = BaselineEngine(
            configuration: BaselineEngineConfiguration(
                modelID: model,
                defaultMaxTokens: maxTokens
            )
        )
        try await engine.load()
        FileHandle.standardError.write(Data("model ready\n".utf8))

        let server = MLXLMHTTPServer(engine: engine, host: host, port: port)
        FileHandle.standardError.write(
            Data("listening on http://\(host):\(port)  (Ctrl-C to stop)\n".utf8)
        )
        try server.run()
    }

    static func printUsage() {
        let text = """
        mlx-lm-server — minimal OpenAI-compatible HTTP server for MLXLMServer engines.

        Usage:
          swift run -c release mlx-lm-server [options]

        Options:
          --model <id>          HuggingFace model id (default: mlx-community/Qwen2.5-0.5B-Instruct-4bit)
          --port <n>            TCP port (default: 8080)
          --host <addr>         bind address (default: 127.0.0.1)
          --max-tokens <n>      default generation cap when the client omits max_tokens (default: 256)
          -h, --help            show this help

        Endpoints:
          GET  /health
          GET  /v1/models
          POST /v1/chat/completions   (streaming and non-streaming)
        """
        print(text)
    }

    static func fatal(_ message: String) -> Never {
        FileHandle.standardError.write(Data("error: \(message)\n".utf8))
        exit(2)
    }
}
