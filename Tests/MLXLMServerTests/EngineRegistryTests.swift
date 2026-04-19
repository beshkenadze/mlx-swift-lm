import Foundation
import XCTest
@testable import MLXLMServer

final class EngineRegistryTests: XCTestCase {
    // MARK: - Init validation

    func testRejectsMultipleDefaults() {
        let a = NamedStubEngine(models: [.init(id: "m", created: 0, ownedBy: "t")])
        let b = NamedStubEngine(models: [.init(id: "n", created: 0, ownedBy: "t")])
        XCTAssertThrowsError(
            try EngineRegistry([
                .init(prefix: "a", engine: a, isDefault: true),
                .init(prefix: "b", engine: b, isDefault: true),
            ])
        ) { error in
            XCTAssertEqual(error as? EngineRegistryError, .multipleDefaults)
        }
    }

    func testRejectsInvalidPrefix() {
        let engine = NamedStubEngine(models: [.init(id: "m", created: 0, ownedBy: "t")])
        for bad in ["", "Bad", "a:b", "-nope", "1abc", "with space"] {
            XCTAssertThrowsError(
                try EngineRegistry([.init(prefix: bad, engine: engine)])
            ) { error in
                XCTAssertEqual(error as? EngineRegistryError, .invalidPrefix(bad))
            }
        }
    }

    func testRejectsDuplicatePrefix() {
        let engine = NamedStubEngine(models: [.init(id: "m", created: 0, ownedBy: "t")])
        XCTAssertThrowsError(
            try EngineRegistry([
                .init(prefix: "foo", engine: engine),
                .init(prefix: "foo", engine: engine),
            ])
        ) { error in
            XCTAssertEqual(error as? EngineRegistryError, .duplicatePrefix("foo"))
        }
    }

    // MARK: - availableModels aggregation

    func testAggregatesPrefixedAndBaseAliasForDefault() async throws {
        let baseline = NamedStubEngine(models: [
            .init(id: "qwen2.5-0.5b-4bit", created: 1, ownedBy: "tests"),
        ])
        let dflash = NamedStubEngine(models: [
            .init(id: "qwen2.5-0.5b-4bit", created: 2, ownedBy: "tests"),
        ])
        let registry = try EngineRegistry([
            .init(prefix: "baseline", engine: baseline, isDefault: true),
            .init(prefix: "dflash", engine: dflash),
        ])

        let models = await registry.availableModels()
        let ids = models.map(\.id).sorted()
        XCTAssertEqual(ids, [
            "baseline:qwen2.5-0.5b-4bit",
            "dflash:qwen2.5-0.5b-4bit",
            "qwen2.5-0.5b-4bit",        // bare alias from default engine
        ])
    }

    func testNotReadyEngineIsFilteredOutOfAvailableModels() async throws {
        let baseline = NamedStubEngine(models: [
            .init(id: "foo", created: 1, ownedBy: "tests"),
        ])
        let dflashNotLoaded = NamedStubEngine(models: [])  // empty = not ready
        let registry = try EngineRegistry([
            .init(prefix: "baseline", engine: baseline, isDefault: true),
            .init(prefix: "dflash", engine: dflashNotLoaded),
        ])

        let models = await registry.availableModels()
        let ids = Set(models.map(\.id))
        XCTAssertTrue(ids.contains("baseline:foo"))
        XCTAssertTrue(ids.contains("foo"))
        XCTAssertFalse(ids.contains { $0.hasPrefix("dflash:") })
    }

    func testRegistryWithoutDefaultEmitsOnlyPrefixed() async throws {
        let baseline = NamedStubEngine(models: [.init(id: "x", created: 0, ownedBy: "t")])
        let dflash = NamedStubEngine(models: [.init(id: "x", created: 0, ownedBy: "t")])
        let registry = try EngineRegistry([
            .init(prefix: "baseline", engine: baseline),
            .init(prefix: "dflash", engine: dflash),
        ])
        let ids = await registry.availableModels().map(\.id).sorted()
        XCTAssertEqual(ids, ["baseline:x", "dflash:x"])
    }

    // MARK: - Routing

    func testPrefixedRoutingDispatchesToMatchingEngine() async throws {
        let baseline = NamedStubEngine(
            models: [.init(id: "foo", created: 0, ownedBy: "t")],
            tag: "baseline"
        )
        let dflash = NamedStubEngine(
            models: [.init(id: "foo", created: 0, ownedBy: "t")],
            tag: "dflash"
        )
        let registry = try EngineRegistry([
            .init(prefix: "baseline", engine: baseline, isDefault: true),
            .init(prefix: "dflash", engine: dflash),
        ])

        let request = ChatRequest(
            modelID: "dflash:foo",
            messages: [ChatMessage(role: "user", content: "x")],
            maxTokens: 4,
            stream: false
        )
        var fragments: [String] = []
        for try await delta in registry.generate(request) {
            fragments.append(contentsOf: delta.textFragments)
        }
        XCTAssertEqual(fragments.joined(), "dflash")
    }

    func testBareModelIDGoesToDefaultEngine() async throws {
        let baseline = NamedStubEngine(
            models: [.init(id: "foo", created: 0, ownedBy: "t")],
            tag: "baseline"
        )
        let dflash = NamedStubEngine(
            models: [.init(id: "foo", created: 0, ownedBy: "t")],
            tag: "dflash"
        )
        let registry = try EngineRegistry([
            .init(prefix: "baseline", engine: baseline, isDefault: true),
            .init(prefix: "dflash", engine: dflash),
        ])

        let request = ChatRequest(
            modelID: "foo",
            messages: [ChatMessage(role: "user", content: "x")],
            maxTokens: 4,
            stream: false
        )
        var fragments: [String] = []
        for try await delta in registry.generate(request) {
            fragments.append(contentsOf: delta.textFragments)
        }
        XCTAssertEqual(fragments.joined(), "baseline")
    }

    func testUnknownPrefixFailsStreamWithUnknownModel() async throws {
        let baseline = NamedStubEngine(
            models: [.init(id: "foo", created: 0, ownedBy: "t")],
            tag: "baseline"
        )
        let registry = try EngineRegistry([
            .init(prefix: "baseline", engine: baseline, isDefault: true),
        ])

        // "bogus:foo" looks like a prefix split but no engine registered.
        // Falls through to default, which doesn't have "bogus:foo".
        let request = ChatRequest(
            modelID: "bogus:foo",
            messages: [ChatMessage(role: "user", content: "x")],
            maxTokens: 4
        )

        do {
            for try await _ in registry.generate(request) { /* drain */ }
            XCTFail("expected unknownModel error")
        } catch EngineRegistryError.unknownModel(let id) {
            XCTAssertEqual(id, "bogus:foo")
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testBareModelIDWithoutDefaultFails() async throws {
        let baseline = NamedStubEngine(models: [.init(id: "foo", created: 0, ownedBy: "t")])
        let registry = try EngineRegistry([
            .init(prefix: "baseline", engine: baseline),   // no default
        ])
        let request = ChatRequest(
            modelID: "foo",
            messages: [ChatMessage(role: "user", content: "x")],
            maxTokens: 4
        )
        do {
            for try await _ in registry.generate(request) {}
            XCTFail("expected noEngineForBareModel")
        } catch EngineRegistryError.noEngineForBareModel(let id) {
            XCTAssertEqual(id, "foo")
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    // MARK: - Per-engine health breakdown

    func testEngineHealthBreakdown() async throws {
        let baseline = NamedStubEngine(
            models: [.init(id: "X", created: 0, ownedBy: "t")],
            tag: "baseline"
        )
        let dflash = NamedStubEngine(
            models: [.init(id: "Y", created: 0, ownedBy: "t")],
            tag: "dflash"
        )
        let registry = try EngineRegistry([
            .init(prefix: "baseline", engine: baseline, isDefault: true),
            .init(prefix: "dflash", engine: dflash),
        ])
        let breakdown = await registry.engineHealth()
        XCTAssertEqual(Set(breakdown.keys), ["baseline", "dflash"])
        XCTAssertTrue(breakdown["baseline"]!.ready)
        XCTAssertTrue(breakdown["dflash"]!.ready)
        XCTAssertEqual(breakdown["baseline"]!.modelIDs, ["X"])
        XCTAssertEqual(breakdown["dflash"]!.modelIDs, ["Y"])
    }

    func testEngineHealthBreakdownReflectsUnreadyEngine() async throws {
        let baseline = NamedStubEngine(
            models: [.init(id: "X", created: 0, ownedBy: "t")],
            tag: "baseline"
        )
        let dflashNotReady = NamedStubEngine(models: [], tag: "dflash")
        let registry = try EngineRegistry([
            .init(prefix: "baseline", engine: baseline, isDefault: true),
            .init(prefix: "dflash", engine: dflashNotReady),
        ])
        let breakdown = await registry.engineHealth()
        XCTAssertTrue(breakdown["baseline"]!.ready)
        XCTAssertFalse(breakdown["dflash"]!.ready)
    }

    func testInnerEngineSeesStrippedModelID() async throws {
        let baseline = InspectingStubEngine(
            models: [.init(id: "inner", created: 0, ownedBy: "t")]
        )
        let registry = try EngineRegistry([
            .init(prefix: "baseline", engine: baseline, isDefault: true),
        ])
        let request = ChatRequest(
            modelID: "baseline:inner",
            messages: [ChatMessage(role: "user", content: "x")],
            maxTokens: 4
        )
        for try await _ in registry.generate(request) {}
        let captured = await baseline.lastModelID()
        XCTAssertEqual(captured, "inner")
    }
}

// MARK: - Test helpers

/// Stub engine that returns `tag` as the sole delta text fragment. Lets tests
/// verify which engine handled a given request by inspecting the output.
private struct NamedStubEngine: InferenceEngine {
    let models: [ModelInfo]
    let tag: String

    init(models: [ModelInfo], tag: String = "stub") {
        self.models = models
        self.tag = tag
    }

    func availableModels() async -> [ModelInfo] { models }

    func health() async -> EngineHealth {
        EngineHealth(
            ready: !models.isEmpty,
            modelIDs: models.map(\.id),
            uptimeSeconds: 0
        )
    }

    func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error> {
        let tag = self.tag
        return AsyncThrowingStream { continuation in
            continuation.yield(
                ChatDelta(
                    textFragments: [tag],
                    finishReason: .stop,
                    usage: Usage(promptTokens: 0, completionTokens: 1)
                )
            )
            continuation.finish()
        }
    }
}

/// Engine that records the modelID it received, for verifying that
/// `EngineRegistry` strips the routing prefix before delegating.
private actor InspectingStubEngine: InferenceEngine {
    private let models: [ModelInfo]
    private var capturedModelID: String?

    init(models: [ModelInfo]) {
        self.models = models
    }

    func lastModelID() -> String? { capturedModelID }

    nonisolated func availableModels() async -> [ModelInfo] {
        await modelsList()
    }

    private func modelsList() -> [ModelInfo] { models }

    nonisolated func health() async -> EngineHealth {
        await healthValue()
    }

    private func healthValue() -> EngineHealth {
        EngineHealth(ready: true, modelIDs: models.map(\.id), uptimeSeconds: 0)
    }

    nonisolated func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error> {
        AsyncThrowingStream { continuation in
            Task { await self.capture(request.modelID) }
            continuation.yield(
                ChatDelta(
                    textFragments: ["ok"],
                    finishReason: .stop,
                    usage: Usage(promptTokens: 0, completionTokens: 1)
                )
            )
            continuation.finish()
        }
    }

    private func capture(_ id: String) {
        capturedModelID = id
    }
}
