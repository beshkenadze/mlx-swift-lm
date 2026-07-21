import Foundation
import XCTest

import struct OmniBench.PromptInput
import struct OmniBench.TaskContext
import struct OmniBench.TokenEvent

@testable import MLXLMOmniBench

final class MLXLMGeneratorTests: XCTestCase {
    private func task(_ overrides: [String: Any] = [:]) -> TaskContext {
        var parameters: [String: Any] = [
            "min_output_tokens": 2,
            "max_output_tokens": 3,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 7,
        ]
        parameters.merge(overrides) { _, replacement in replacement }
        return TaskContext(
            definitionId: "textgen.performance.standard.v1",
            interfaceFamilyId: "text_generation.v1",
            language: "en",
            runProfile: ["family_parameters": parameters]
        )
    }

    func testResolvesClosedIdentityBearingControlsIncludingNullSeed() throws {
        let controls = try GenerationControls.resolve(task: task(["seed": NSNull()]))
        XCTAssertEqual(controls.minOutputTokens, 2)
        XCTAssertEqual(controls.maxOutputTokens, 3)
        XCTAssertEqual(controls.temperature, 0)
        XCTAssertEqual(controls.topP, 1)
        XCTAssertNil(controls.seed)
    }

    func testRejectsInvalidControlsAndBooleanNumbers() {
        XCTAssertThrowsError(
            try GenerationControls.resolve(
                task: task([
                    "min_output_tokens": 4,
                    "max_output_tokens": 3,
                ])))
        XCTAssertThrowsError(
            try GenerationControls.resolve(
                task: task([
                    "temperature": true
                ])))
        XCTAssertThrowsError(
            try GenerationControls.resolve(
                task: task([
                    "top_p": 0
                ])))
        XCTAssertThrowsError(
            try GenerationControls.resolve(
                task: task([
                    "seed": "7"
                ])))
    }

    func testMinimumOutputGateMasksStopTokensUntilRequestedCount() {
        var gate = MinimumOutputGate(minimum: 2, hasStopTokens: true)
        XCTAssertTrue(gate.shouldMaskStopTokens)
        gate.didSample()
        XCTAssertTrue(gate.shouldMaskStopTokens)
        gate.didSample()
        XCTAssertFalse(gate.shouldMaskStopTokens)

        let noStops = MinimumOutputGate(minimum: 2, hasStopTokens: false)
        XCTAssertFalse(noStops.shouldMaskStopTokens)
    }

    func testGeneratedTextNormalizerTrimsOnlyTheInitialSpace() {
        var normalizer = GeneratedTextNormalizer()
        XCTAssertEqual(normalizer.append(""), "")
        XCTAssertEqual(normalizer.append(" "), "")
        XCTAssertEqual(normalizer.append("hello"), "hello")
        XCTAssertEqual(normalizer.append(" world"), " world")
    }

    func testStreamingEmitsOneEventPerRawTokenIncludingEmptyText() async throws {
        let generator = MLXLMGenerator { prompt, controls, emit in
            XCTAssertEqual(prompt, "hello")
            XCTAssertEqual(controls.seed, 7)
            let deltas = ["A", "", "B"]
            for delta in deltas { try emit?(delta) }
            return RunOutput(text: deltas.joined(), generatedTokens: deltas.count)
        }
        var events: [TokenEvent] = []
        let generation = try await generator.generateStream(
            PromptInput(id: "sample-1", prompt: "hello"),
            task: task(),
            emit: { events.append($0) }
        )

        XCTAssertEqual(events.map(\.textDelta), ["A", "", "B"])
        XCTAssertEqual(events.map(\.tokenCount), [1, 1, 1])
        XCTAssertEqual(events.map(\.textDelta).joined(), generation.text)
        XCTAssertEqual(generation.generatedTokens, events.count)
    }

    func testBatchDoesNotSynthesizeStreamingEvents() async throws {
        let generator = MLXLMGenerator { _, _, emit in
            XCTAssertNil(emit)
            return RunOutput(text: "answer", generatedTokens: 2)
        }
        let generation = try await generator.generate(
            PromptInput(id: "sample-1", prompt: "hello"),
            task: task()
        )
        XCTAssertEqual(generation.text, "answer")
        XCTAssertEqual(generation.generatedTokens, 2)
    }

    func testDeclaresOnlyQualifiedSingleStreamConcurrency() {
        let generator = MLXLMGenerator { _, _, _ in
            RunOutput(text: "", generatedTokens: 0)
        }
        let capabilities = generator.capabilities()
        XCTAssertTrue(capabilities.supportsStreaming)
        XCTAssertEqual(capabilities.maxConcurrency, 1)
    }
}
