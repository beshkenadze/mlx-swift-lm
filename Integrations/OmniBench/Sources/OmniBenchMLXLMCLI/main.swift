import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXLMOmniBench
import OmniBench
import Tokenizers

private enum QualificationError: Error, LocalizedError {
    case usage(String)

    var errorDescription: String? {
        switch self {
        case .usage(let detail): detail
        }
    }
}

private enum DeliveryMode: String {
    case batch
    case streaming

    var measurementProfileId: String {
        switch self {
        case .batch: "text_generation.batch_single.v1"
        case .streaming: "text_generation.streaming_single.v1"
        }
    }
}

private struct Arguments {
    let modelDirectory: URL
    let modelId: String
    let modelArtifactSHA256: String
    let quantization: String?
    let manifest: URL
    let registryBundle: URL
    let output: URL
    let mode: DeliveryMode
    let backendVersion: String
    let implementation: String
    let environmentLabel: String?
    let minOutputTokens: Int
    let maxOutputTokens: Int
    let chunkMS: Int
    let warmupSamples: Int

    init(_ raw: [String]) throws {
        var values: [String: String] = [:]
        var index = 0
        while index < raw.count {
            let key = raw[index]
            guard key.hasPrefix("--"), index + 1 < raw.count else {
                throw QualificationError.usage("expected --name value arguments")
            }
            guard values[key] == nil else {
                throw QualificationError.usage("duplicate argument: \(key)")
            }
            values[key] = raw[index + 1]
            index += 2
        }
        let allowed: Set<String> = [
            "--model-directory", "--model-id", "--model-artifact-sha256",
            "--quantization", "--manifest", "--registry-bundle", "--out", "--mode",
            "--backend-version", "--implementation", "--environment-label",
            "--output-tokens", "--min-output-tokens", "--max-output-tokens",
            "--chunk-ms", "--warmup-samples",
        ]
        let unknown = Set(values.keys).subtracting(allowed)
        guard unknown.isEmpty else {
            throw QualificationError.usage("unknown arguments: \(unknown.sorted())")
        }

        func required(_ key: String) throws -> String {
            guard let value = values[key], !value.isEmpty else {
                throw QualificationError.usage("missing \(key)")
            }
            return value
        }
        func integer(_ key: String, default fallback: Int, minimum: Int) throws -> Int {
            let value = values[key].flatMap(Int.init) ?? fallback
            guard value >= minimum else {
                throw QualificationError.usage("\(key) must be >= \(minimum)")
            }
            return value
        }

        modelDirectory = URL(fileURLWithPath: try required("--model-directory"))
            .standardizedFileURL
        modelId = try required("--model-id")
        modelArtifactSHA256 = try required("--model-artifact-sha256")
        quantization = values["--quantization"].flatMap { $0 == "null" ? nil : $0 }
        manifest = URL(fileURLWithPath: try required("--manifest")).standardizedFileURL
        registryBundle = URL(fileURLWithPath: try required("--registry-bundle"))
            .standardizedFileURL
        output = URL(fileURLWithPath: try required("--out")).standardizedFileURL
        guard let parsedMode = DeliveryMode(rawValue: try required("--mode")) else {
            throw QualificationError.usage("--mode must be batch or streaming")
        }
        mode = parsedMode
        backendVersion = try required("--backend-version")
        implementation = try required("--implementation")
        environmentLabel = values["--environment-label"]
        if values["--output-tokens"] != nil
            && (values["--min-output-tokens"] != nil || values["--max-output-tokens"] != nil)
        {
            throw QualificationError.usage(
                "--output-tokens cannot be combined with --min-output-tokens or --max-output-tokens")
        }
        if let exact = values["--output-tokens"] {
            guard let parsed = Int(exact), parsed >= 1 else {
                throw QualificationError.usage("--output-tokens must be >= 1")
            }
            minOutputTokens = parsed
            maxOutputTokens = parsed
        } else {
            minOutputTokens = try integer("--min-output-tokens", default: 0, minimum: 0)
            maxOutputTokens = try integer("--max-output-tokens", default: 256, minimum: 1)
            guard maxOutputTokens >= minOutputTokens else {
                throw QualificationError.usage(
                    "--max-output-tokens must be >= --min-output-tokens")
            }
        }
        chunkMS = try integer("--chunk-ms", default: 100, minimum: 1)
        warmupSamples = try integer("--warmup-samples", default: 0, minimum: 0)

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: modelDirectory.path, isDirectory: &isDirectory), isDirectory.boolValue
        else {
            throw QualificationError.usage("model directory does not exist")
        }
    }

    var runProfile: [String: Any] {
        [
            "delivery": mode.rawValue,
            "chunk_ms": mode == .streaming ? chunkMS : NSNull(),
            "warmup_samples": warmupSamples,
            "concurrency": 1,
            "family_parameters": [
                "min_output_tokens": minOutputTokens,
                "max_output_tokens": maxOutputTokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 0,
            ],
        ]
    }

    var modelIdentity: [String: Any] {
        [
            "base_model_id": modelId,
            "artifact_sha256": modelArtifactSHA256,
            "quantization": quantization.map { $0 as Any } ?? NSNull(),
        ]
    }
}

@main
private struct OmniBenchMLXLMCommand {
    static func main() async {
        do {
            let arguments = try Arguments(Array(CommandLine.arguments.dropFirst()))
            let run = try TextGenerationApplication.resolveRun(
                manifestURL: arguments.manifest,
                registryBundleURL: arguments.registryBundle,
                model: arguments.modelIdentity,
                backend: ["id": "mlx_swift_lm", "version": arguments.backendVersion],
                measurementProfileId: arguments.mode.measurementProfileId,
                runProfile: arguments.runProfile,
                implementation: arguments.implementation,
                environmentLabel: arguments.environmentLabel)

            let container = try await LLMModelFactory.shared.loadContainer(
                from: arguments.modelDirectory,
                using: #huggingFaceTokenizerLoader())
            let adapter = MLXLMGenerator(container: container)
            let producer = TextGenerationProducer(run: run)
            let result: TextGenerationProductionResult
            switch arguments.mode {
            case .batch:
                result = try await producer.runBatch(
                    adapter: adapter, outputURL: arguments.output)
            case .streaming:
                result = try await producer.runStreaming(
                    adapter: adapter, outputURL: arguments.output)
            }
            print("artifact=\(result.artifactURL.path)")
            print("identity_key=\(result.identityKey)")
            print("artifact_sha256=\(result.artifactSha256)")
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
            exit(2)
        }
    }
}
