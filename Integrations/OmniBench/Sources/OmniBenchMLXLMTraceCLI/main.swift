import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLX
import Tokenizers

private enum TraceError: Error, LocalizedError {
    case usage(String)

    var errorDescription: String? {
        switch self {
        case .usage(let detail): detail
        }
    }
}

private struct Arguments {
    let modelDirectory: URL
    let manifest: URL
    let sampleId: String
    let maxTokens: Int
    let output: URL

    init(_ raw: [String]) throws {
        var values: [String: String] = [:]
        var index = 0
        while index < raw.count {
            let key = raw[index]
            guard key.hasPrefix("--"), index + 1 < raw.count else {
                throw TraceError.usage("expected --name value arguments")
            }
            guard values[key] == nil else {
                throw TraceError.usage("duplicate argument: \(key)")
            }
            values[key] = raw[index + 1]
            index += 2
        }

        let allowed: Set<String> = [
            "--model-directory", "--manifest", "--sample-id", "--max-tokens", "--out",
        ]
        let unknown = Set(values.keys).subtracting(allowed)
        guard unknown.isEmpty else {
            throw TraceError.usage("unknown arguments: \(unknown.sorted())")
        }

        func required(_ key: String) throws -> String {
            guard let value = values[key], !value.isEmpty else {
                throw TraceError.usage("missing \(key)")
            }
            return value
        }

        modelDirectory = URL(fileURLWithPath: try required("--model-directory"))
            .standardizedFileURL
        manifest = URL(fileURLWithPath: try required("--manifest")).standardizedFileURL
        sampleId = try required("--sample-id")
        output = URL(fileURLWithPath: try required("--out")).standardizedFileURL
        maxTokens = try values["--max-tokens"].map {
            guard let value = Int($0), value > 0 else {
                throw TraceError.usage("--max-tokens must be a positive integer")
            }
            return value
        } ?? 32
    }
}

private struct TraceRequest: Sendable {
    let prompt: String
    let maxTokens: Int
}

private struct TokenTrace: Sendable {
    let promptTokenIds: [Int]
    let generatedTokenIds: [Int]
}

@main
private struct OmniBenchMLXLMTraceCommand {
    static func main() async {
        do {
            let arguments = try Arguments(Array(CommandLine.arguments.dropFirst()))
            let prompt = try loadPrompt(
                manifest: arguments.manifest, sampleId: arguments.sampleId)
            let container = try await LLMModelFactory.shared.loadContainer(
                from: arguments.modelDirectory,
                using: #huggingFaceTokenizerLoader())
            let request = TraceRequest(prompt: prompt, maxTokens: arguments.maxTokens)
            let trace = try await container.perform(nonSendable: request) { context, request in
                let promptTokenIds = context.tokenizer.encode(text: request.prompt)
                let input = LMInput(tokens: MLXArray(promptTokenIds))
                let parameters = GenerateParameters(
                    maxTokens: request.maxTokens, temperature: 0, topP: 1)
                let iterator = try TokenIterator(
                    input: input,
                    model: context.model,
                    processor: parameters.processor(),
                    sampler: parameters.sampler(),
                    prefillStepSize: parameters.prefillStepSize,
                    maxTokens: request.maxTokens)
                let (stream, generationTask) = generateTokenTask(
                    promptTokenCount: input.text.tokens.size,
                    modelConfiguration: context.configuration,
                    tokenizer: context.tokenizer,
                    iterator: iterator,
                    includeStopToken: false)

                var generatedTokenIds: [Int] = []
                for await event in stream {
                    if case .token(let token) = event { generatedTokenIds.append(token) }
                }
                await generationTask.value
                return TokenTrace(
                    promptTokenIds: promptTokenIds,
                    generatedTokenIds: generatedTokenIds)
            }

            let document: [String: Any] = [
                "diagnostic_type": "omni_bench_text_generation_token_trace",
                "sample_id": arguments.sampleId,
                "prompt_token_ids": trace.promptTokenIds,
                "generated_token_ids": trace.generatedTokenIds,
            ]
            let data = try JSONSerialization.data(
                withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: arguments.output, options: .atomic)
            print("trace=\(arguments.output.path)")
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
            exit(2)
        }
    }

    private static func loadPrompt(manifest: URL, sampleId: String) throws -> String {
        let data = try Data(contentsOf: manifest)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let samples = root["samples"] as? [[String: Any]],
            let sample = samples.first(where: { $0["sample_id"] as? String == sampleId }),
            let payload = sample["payload"] as? [String: Any],
            let prompt = payload["prompt"] as? String
        else {
            throw TraceError.usage("sample prompt not found in manifest")
        }
        return prompt
    }
}
