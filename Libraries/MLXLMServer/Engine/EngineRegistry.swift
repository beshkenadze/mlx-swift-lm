import Foundation

/// One engine registration inside a `EngineRegistry`.
public struct EngineRegistryEntry: Sendable {
    /// Routing prefix (e.g. "baseline", "dflash"). Must match
    /// `^[a-z][a-z0-9-]*$`.
    public let prefix: String
    public let engine: any InferenceEngine
    /// If true, this engine is invoked when a client requests a model ID
    /// without any routing prefix. At most one entry per registry may be
    /// marked `isDefault`.
    public let isDefault: Bool

    public init(prefix: String, engine: any InferenceEngine, isDefault: Bool = false) {
        self.prefix = prefix
        self.engine = engine
        self.isDefault = isDefault
    }
}

public enum EngineRegistryError: Error, Equatable {
    case multipleDefaults
    case invalidPrefix(String)
    case duplicatePrefix(String)
    case unknownModel(String)
    case noEngineForBareModel(String)
}

/// Composes multiple `InferenceEngine`s behind a single `InferenceEngine`
/// surface. Routes generate requests by model-ID prefix.
///
/// Example:
/// ```
/// let registry = try EngineRegistry([
///     .init(prefix: "baseline", engine: base, isDefault: true),
///     .init(prefix: "dflash", engine: dflash),
/// ])
/// try MLXLMHTTPServer(engine: registry).run()
/// ```
/// Clients see:
/// - `foo`                      → baseline (default alias)
/// - `baseline:foo`             → baseline
/// - `dflash:foo`               → dflash
/// Engines whose `availableModels()` returns `[]` are excluded from the
/// aggregated list (they stay invisible until ready).
public final class EngineRegistry: InferenceEngine, @unchecked Sendable {
    public let entries: [EngineRegistryEntry]
    private let defaultEntry: EngineRegistryEntry?

    public init(_ entries: [EngineRegistryEntry]) throws {
        // Validate prefixes.
        var seenPrefixes = Set<String>()
        for entry in entries {
            guard Self.isValidPrefix(entry.prefix) else {
                throw EngineRegistryError.invalidPrefix(entry.prefix)
            }
            if !seenPrefixes.insert(entry.prefix).inserted {
                throw EngineRegistryError.duplicatePrefix(entry.prefix)
            }
        }
        let defaults = entries.filter(\.isDefault)
        guard defaults.count <= 1 else {
            throw EngineRegistryError.multipleDefaults
        }
        self.entries = entries
        self.defaultEntry = defaults.first
    }

    public func availableModels() async -> [ModelInfo] {
        var result: [ModelInfo] = []
        var baseEmitted = Set<String>()

        for entry in entries {
            let models = await entry.engine.availableModels()
            for model in models {
                // Always emit prefixed form while the engine actually lists it.
                result.append(ModelInfo(
                    id: "\(entry.prefix):\(model.id)",
                    created: model.created,
                    ownedBy: model.ownedBy
                ))
            }
        }

        // Emit bare alias only for the default engine's models, once per ID.
        if let defaultEntry {
            for model in await defaultEntry.engine.availableModels() where !baseEmitted.contains(model.id) {
                baseEmitted.insert(model.id)
                result.append(model)
            }
        }

        return result
    }

    public func health() async -> EngineHealth {
        var anyReady = false
        var ids: [String] = []
        var uptime: Double = 0
        for entry in entries {
            let h = await entry.engine.health()
            anyReady = anyReady || h.ready
            uptime = max(uptime, h.uptimeSeconds)
            for id in h.modelIDs {
                ids.append("\(entry.prefix):\(id)")
            }
        }
        if let defaultEntry {
            let defaultHealth = await defaultEntry.engine.health()
            ids.append(contentsOf: defaultHealth.modelIDs)
        }
        return EngineHealth(ready: anyReady, modelIDs: ids, uptimeSeconds: uptime)
    }

    public func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error> {
        let (chosenEntry, innerID): (EngineRegistryEntry?, String) = {
            if let (prefix, remainder) = Self.splitPrefix(request.modelID),
               let match = entries.first(where: { $0.prefix == prefix })
            {
                return (match, remainder)
            }
            if let defaultEntry {
                return (defaultEntry, request.modelID)
            }
            return (nil, request.modelID)
        }()

        guard let chosenEntry else {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: EngineRegistryError.noEngineForBareModel(request.modelID))
            }
        }

        let innerRequest = request.withModelID(innerID)
        let engine = chosenEntry.engine

        // Validate the inner engine actually serves this model. If not, fail
        // synchronously so clients get a clean 404-style error rather than
        // whatever the engine decides to do on an unknown ID.
        return AsyncThrowingStream { continuation in
            Task {
                let models = await engine.availableModels()
                guard models.contains(where: { $0.id == innerID }) else {
                    continuation.finish(throwing: EngineRegistryError.unknownModel(request.modelID))
                    return
                }
                do {
                    for try await delta in engine.generate(innerRequest) {
                        continuation.yield(delta)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Helpers

    static func isValidPrefix(_ prefix: String) -> Bool {
        guard !prefix.isEmpty, prefix.count <= 32 else { return false }
        var isFirst = true
        for scalar in prefix.unicodeScalars {
            if isFirst {
                guard CharacterSet.lowercaseLetters.contains(scalar) else { return false }
                isFirst = false
            } else {
                let allowed = CharacterSet.lowercaseLetters
                    .union(.decimalDigits)
                    .union(CharacterSet(charactersIn: "-"))
                guard allowed.contains(scalar) else { return false }
            }
        }
        return true
    }

    static func splitPrefix(_ modelID: String) -> (prefix: String, remainder: String)? {
        guard let colonIdx = modelID.firstIndex(of: ":") else { return nil }
        let prefix = String(modelID[..<colonIdx])
        let remainder = String(modelID[modelID.index(after: colonIdx)...])
        guard isValidPrefix(prefix), !remainder.isEmpty else { return nil }
        return (prefix, remainder)
    }
}
