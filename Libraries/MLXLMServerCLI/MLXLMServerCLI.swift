import Foundation
import MLXLMServer

@main
struct MLXLMServerCLI {
    static func main() async throws {
        var model = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        var dflashTarget = "mlx-community/Qwen3-4B-bf16"
        var dflashDraft = "z-lab/Qwen3-4B-DFlash-b16"
        var dflashAlias = "qwen3-4b"
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
            case "--dflash-target":
                guard let next = iterator.next() else { fatal("--dflash-target requires a value") }
                dflashTarget = next
            case "--dflash-draft":
                guard let next = iterator.next() else { fatal("--dflash-draft requires a value") }
                dflashDraft = next
            case "--dflash-alias":
                guard let next = iterator.next() else { fatal("--dflash-alias requires a value") }
                dflashAlias = next
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

        FileHandle.standardError.write(Data("loading baseline model: \(model)\n".utf8))
        let baseline = BaselineEngine(
            configuration: BaselineEngineConfiguration(
                modelID: model,
                defaultMaxTokens: maxTokens
            )
        )
        let dflash = DFlashEngine(
            configuration: DFlashEngineConfiguration(
                targetModelID: dflashTarget,
                draftRepositoryID: dflashDraft,
                modelAlias: dflashAlias,
                defaultMaxTokens: maxTokens
            )
        )

        try await baseline.load()
        FileHandle.standardError.write(Data("baseline model ready\n".utf8))

        let registry = try EngineRegistry([
            .init(prefix: "baseline", engine: baseline, isDefault: true),
            .init(prefix: "dflash", engine: dflash),
        ])

        let server = MLXLMHTTPServer(engine: registry, host: host, port: port)
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
          --dflash-target <id>  DFlash target HuggingFace model id (default: mlx-community/Qwen3-4B-bf16)
          --dflash-draft <id>   DFlash draft repository id (default: z-lab/Qwen3-4B-DFlash-b16)
          --dflash-alias <id>   DFlash model alias exposed as dflash:<alias> (default: qwen3-4b)
          --port <n>            TCP port (default: 8080)
          --host <addr>         bind address (default: 127.0.0.1)
          --max-tokens <n>      default generation cap when the client omits max_tokens (default: 256)
          -h, --help            show this help

        Endpoints:
          GET  /health
          GET  /v1/models
          POST /v1/chat/completions   (streaming and non-streaming)

        Models:
          <baseline-model-id>         default route via baseline engine
          baseline:<baseline-model-id>
          dflash:<alias>
        """
        print(text)
    }

    static func fatal(_ message: String) -> Never {
        FileHandle.standardError.write(Data("error: \(message)\n".utf8))
        exit(2)
    }
}
