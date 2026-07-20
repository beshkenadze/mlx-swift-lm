import CoreFoundation
import Foundation
import MLX

import struct MLXLMCommon.GenerateParameters
import protocol MLXLMCommon.LogitProcessor
import class MLXLMCommon.ModelContainer
import struct MLXLMCommon.NaiveStreamingDetokenizer
import enum MLXLMCommon.TokenGeneration
import struct MLXLMCommon.TokenIterator
import struct MLXLMCommon.UserInput
import func MLXLMCommon.generateTokenTask
import struct OmniBench.Capabilities
import struct OmniBench.Generation
import protocol OmniBench.Generator
import struct OmniBench.PromptInput
import protocol OmniBench.StreamingGenerator
import struct OmniBench.TaskContext
import struct OmniBench.TokenEvent

public struct MLXLMAdapterError: Error, Equatable, LocalizedError {
    public let code: String
    public let detail: String

    public init(code: String, detail: String) {
        self.code = code
        self.detail = detail
    }

    public var errorDescription: String? { "\(code): \(detail)" }
}

public struct GenerationControls: Equatable, Sendable {
    public let minOutputTokens: Int
    public let maxOutputTokens: Int
    public let temperature: Double
    public let topP: Double
    public let seed: Int?

    public init(
        minOutputTokens: Int,
        maxOutputTokens: Int,
        temperature: Double,
        topP: Double,
        seed: Int?
    ) throws {
        guard minOutputTokens >= 0, maxOutputTokens >= 1, maxOutputTokens >= minOutputTokens else {
            throw MLXLMAdapterError(
                code: "run_profile.invalid_output_token_bounds",
                detail: "expected 0 <= min_output_tokens <= max_output_tokens"
            )
        }
        guard temperature.isFinite, temperature >= 0 else {
            throw MLXLMAdapterError(
                code: "run_profile.invalid_temperature",
                detail: "temperature must be finite and non-negative"
            )
        }
        guard topP.isFinite, topP > 0, topP <= 1 else {
            throw MLXLMAdapterError(
                code: "run_profile.invalid_top_p",
                detail: "top_p must be finite and in (0, 1]"
            )
        }
        self.minOutputTokens = minOutputTokens
        self.maxOutputTokens = maxOutputTokens
        self.temperature = temperature
        self.topP = topP
        self.seed = seed
    }

    public static func resolve(task: TaskContext) throws -> GenerationControls {
        guard let parameters = task.runProfile["family_parameters"] as? [String: Any] else {
            throw MLXLMAdapterError(
                code: "run_profile.missing_family_parameters",
                detail: "text_generation.v1 requires family_parameters"
            )
        }
        let required = Set([
            "min_output_tokens", "max_output_tokens", "temperature", "top_p", "seed",
        ])
        guard Set(parameters.keys) == required else {
            throw MLXLMAdapterError(
                code: "run_profile.invalid_family_parameters",
                detail: "text-generation parameters must use the closed 0.6.0 shape"
            )
        }
        guard let minOutputTokens = integer(parameters["min_output_tokens"]),
            let maxOutputTokens = integer(parameters["max_output_tokens"]),
            let temperature = number(parameters["temperature"]),
            let topP = number(parameters["top_p"])
        else {
            throw MLXLMAdapterError(
                code: "run_profile.invalid_family_parameters",
                detail: "text-generation controls have invalid JSON types"
            )
        }

        let seed: Int?
        if parameters["seed"] is NSNull {
            seed = nil
        } else if let value = integer(parameters["seed"]) {
            seed = value
        } else {
            throw MLXLMAdapterError(
                code: "run_profile.invalid_seed",
                detail: "seed must be an integer or null"
            )
        }
        return try GenerationControls(
            minOutputTokens: minOutputTokens,
            maxOutputTokens: maxOutputTokens,
            temperature: temperature,
            topP: topP,
            seed: seed
        )
    }

    private static func integer(_ value: Any?) -> Int? {
        guard let value, !(value is Bool) else { return nil }
        if let value = value as? Int { return value }
        guard let value = value as? NSNumber,
            CFGetTypeID(value) != CFBooleanGetTypeID()
        else { return nil }
        let number = value.doubleValue
        guard number.isFinite, number.rounded() == number,
            number >= Double(Int.min), number <= Double(Int.max)
        else { return nil }
        return Int(number)
    }

    private static func number(_ value: Any?) -> Double? {
        guard let value, !(value is Bool) else { return nil }
        if let value = value as? Double { return value }
        if let value = value as? Int { return Double(value) }
        guard let value = value as? NSNumber,
            CFGetTypeID(value) != CFBooleanGetTypeID()
        else { return nil }
        return value.doubleValue
    }
}

struct MinimumOutputGate {
    let minimum: Int
    let hasStopTokens: Bool
    private(set) var sampled = 0

    var shouldMaskStopTokens: Bool {
        sampled < minimum && hasStopTokens
    }

    mutating func didSample() {
        sampled += 1
    }
}

struct MinimumOutputProcessor: LogitProcessor {
    private var upstream: (any LogitProcessor)?
    private let stopTokenIds: [Int]
    private var gate: MinimumOutputGate

    init(upstream: (any LogitProcessor)?, stopTokenIds: Set<Int>, minimum: Int) {
        self.upstream = upstream
        self.stopTokenIds = stopTokenIds.sorted()
        self.gate = MinimumOutputGate(
            minimum: minimum,
            hasStopTokens: !stopTokenIds.isEmpty
        )
    }

    mutating func prompt(_ prompt: MLXArray) {
        upstream?.prompt(prompt)
    }

    func process(logits: MLXArray) -> MLXArray {
        let logits = upstream?.process(logits: logits) ?? logits
        if gate.shouldMaskStopTokens {
            logits[0..., MLXArray(stopTokenIds)] = MLXArray(-Float.infinity)
        }
        return logits
    }

    mutating func didSample(token: MLXArray) {
        upstream?.didSample(token: token)
        gate.didSample()
    }
}

struct RunOutput: Sendable {
    let text: String
    let generatedTokens: Int
}

typealias EventSink = (String) throws -> Void
typealias Runner = (String, GenerationControls, EventSink?) async throws -> RunOutput

public final class MLXLMGenerator: Generator, StreamingGenerator {
    private let runner: Runner

    public init(container: ModelContainer) {
        runner = { prompt, controls, emit in
            try await Self.run(container: container, prompt: prompt, controls: controls, emit: emit)
        }
    }

    init(runner: @escaping Runner) {
        self.runner = runner
    }

    public func capabilities() -> Capabilities {
        // The first qualification is deliberately single-stream. Shared-model
        // concurrent load needs its own reviewed evidence before this is raised.
        Capabilities(supportsStreaming: true, maxConcurrency: 1)
    }

    public func resetCache() {
        // Every request below creates a fresh TokenIterator and KV cache.
    }

    public func generate(_ prompt: PromptInput, task: TaskContext) async throws -> Generation {
        let controls = try GenerationControls.resolve(task: task)
        let output = try await runner(prompt.prompt, controls, nil)
        return Generation(text: output.text, generatedTokens: output.generatedTokens)
    }

    public func generateStream(
        _ prompt: PromptInput,
        task: TaskContext,
        emit: @escaping (TokenEvent) throws -> Void
    ) async throws -> Generation {
        let controls = try GenerationControls.resolve(task: task)
        let output = try await runner(prompt.prompt, controls) { delta in
            try emit(TokenEvent(textDelta: delta, tokenCount: 1))
        }
        return Generation(text: output.text, generatedTokens: output.generatedTokens)
    }

    private struct Request {
        let prompt: String
        let controls: GenerationControls
        let emit: EventSink?
    }

    private static func run(
        container: ModelContainer,
        prompt: String,
        controls: GenerationControls,
        emit: EventSink?
    ) async throws -> RunOutput {
        let request = Request(prompt: prompt, controls: controls, emit: emit)
        return try await container.perform(nonSendable: request) { context, request in
            if let seed = request.controls.seed {
                MLXRandom.seed(UInt64(bitPattern: Int64(seed)))
            }

            let input = try await context.processor.prepare(
                input: UserInput(prompt: .text(request.prompt))
            )
            let parameters = GenerateParameters(
                maxTokens: request.controls.maxOutputTokens,
                temperature: Float(request.controls.temperature),
                topP: Float(request.controls.topP)
            )
            var stopTokenIds = context.configuration.eosTokenIds
            if let eos = context.tokenizer.eosTokenId { stopTokenIds.insert(eos) }
            if let unknown = context.tokenizer.unknownTokenId { stopTokenIds.insert(unknown) }
            for token in context.configuration.extraEOSTokens {
                if let id = context.tokenizer.convertTokenToId(token) {
                    stopTokenIds.insert(id)
                }
            }
            let processor = MinimumOutputProcessor(
                upstream: parameters.processor(),
                stopTokenIds: stopTokenIds,
                minimum: request.controls.minOutputTokens
            )
            let iterator = try TokenIterator(
                input: input,
                model: context.model,
                processor: processor,
                sampler: parameters.sampler(),
                prefillStepSize: parameters.prefillStepSize,
                maxTokens: request.controls.maxOutputTokens
            )
            let (stream, generationTask) = generateTokenTask(
                promptTokenCount: input.text.tokens.size,
                modelConfiguration: context.configuration,
                tokenizer: context.tokenizer,
                iterator: iterator,
                includeStopToken: false
            )

            var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
            var pieces: [String] = []
            var generatedTokens = 0
            var completionCount: Int?
            do {
                for await event in stream {
                    switch event {
                    case .token(let token):
                        detokenizer.append(token: token)
                        let delta = detokenizer.next() ?? ""
                        pieces.append(delta)
                        generatedTokens += 1
                        try request.emit?(delta)
                    case .info(let info):
                        completionCount = info.generationTokenCount
                    }
                }
                await generationTask.value
            } catch {
                generationTask.cancel()
                await generationTask.value
                throw error
            }

            guard completionCount == generatedTokens else {
                throw MLXLMAdapterError(
                    code: "generation.token_count_mismatch",
                    detail: "MLX completion count does not match raw token events"
                )
            }
            guard generatedTokens >= request.controls.minOutputTokens,
                generatedTokens <= request.controls.maxOutputTokens
            else {
                throw MLXLMAdapterError(
                    code: "generation.output_token_bounds",
                    detail: "generated token count is outside the requested bounds"
                )
            }
            return RunOutput(text: pieces.joined(), generatedTokens: generatedTokens)
        }
    }
}
