// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Minimal draft-model configuration surface needed by `DFlashIterator`.
///
/// This lives in `MLXLMCommon` so the iterator does not need to depend on
/// concrete `MLXLLM` draft-model types.
public protocol DFlashDraftConfiguration {
    var blockSize: Int { get }
    var maskTokenId: Int { get }
    var targetLayerIds: [Int] { get }
}

/// Minimal draft-model runtime surface needed by `DFlashIterator`.
///
/// This keeps the iterator generic over draft implementations without creating
/// an `MLXLMCommon -> MLXLLM` target dependency.
public protocol DFlashDraftingModel {
    var numDraftLayers: Int { get }

    func callAsFunction(
        noiseEmbedding: MLXArray,
        targetHidden: MLXArray,
        caches: [KVCache?]
    ) -> MLXArray
}

public struct DFlashIterator: TokenIteratorProtocol {
    public typealias Element = Int

    let promptTokens: MLXArray
    let target: any DFlashTargetModel
    let drafter: any DFlashDraftingModel
    let draftConfig: any DFlashDraftConfiguration
    let blockSize: Int
    let maskTokenId: Int
    let targetLayerIds: [Int]
    let stopTokenIds: Set<Int>
    public let maxTokens: Int?

    var targetCache: [KVCache]
    var draftCache: [KVCache]
    var committedLength: Int
    var lastTargetHidden: MLXArray?
    var pendingTokens: [Int]
    var pendingIndex: Int
    var emittedCount: Int
    var firstStepDone: Bool

    public private(set) var totalProposed: Int = 0
    public private(set) var totalAccepted: Int = 0
    public private(set) var promptPrefillTime: TimeInterval = 0.0

    public var tokenCount: Int { emittedCount }

    public var acceptanceRate: Double {
        totalProposed == 0 ? 0 : Double(totalAccepted) / Double(totalProposed)
    }

    public init(
        promptTokens: MLXArray,
        target: any DFlashTargetModel,
        drafter: any DFlashDraftingModel,
        draftConfig: any DFlashDraftConfiguration,
        stopTokenIds: Set<Int> = [],
        maxTokens: Int? = nil
    ) throws {
        self.promptTokens = promptTokens
        self.target = target
        self.drafter = drafter
        self.draftConfig = draftConfig
        self.blockSize = draftConfig.blockSize
        self.maskTokenId = draftConfig.maskTokenId
        self.targetLayerIds = draftConfig.targetLayerIds
        self.stopTokenIds = stopTokenIds
        self.maxTokens = maxTokens

        self.targetCache = target.newCache(parameters: nil)
        self.draftCache = (0 ..< drafter.numDraftLayers).map { _ in KVCacheSimple() as KVCache }

        guard canTrimPromptCache(self.targetCache), canTrimPromptCache(self.draftCache) else {
            throw KVCacheError(message: "Speculative decoding requires trimmable KV caches")
        }

        self.committedLength = 0
        self.lastTargetHidden = nil
        self.pendingTokens = []
        self.pendingIndex = 0
        self.emittedCount = 0
        self.firstStepDone = false
    }

    public mutating func next() -> Int? {
        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            emittedCount += 1
            return token
        }

        if let maxTokens, emittedCount >= maxTokens {
            return nil
        }

        if !firstStepDone {
            let start = Date.timeIntervalSinceReferenceDate
            prefill()
            promptPrefillTime = Date.timeIntervalSinceReferenceDate - start
            firstStepDone = true

            if pendingIndex < pendingTokens.count {
                let token = pendingTokens[pendingIndex]
                pendingIndex += 1
                emittedCount += 1
                return token
            }
        }

        // B1.2 will implement speculation loop here.
        return nil
    }

    private mutating func prefill() {
        let result = target.captureHiddenStatesAndLogits(
            inputs: promptTokens,
            layerIndices: targetLayerIds,
            cache: targetCache
        )

        let combinedHidden =
            if result.hiddenStates.count == 1 {
                result.hiddenStates[0]
            } else {
                concatenated(result.hiddenStates, axis: -1)
            }
        eval(result.logits, combinedHidden)

        let firstToken = argMax(result.logits[0..., -1, 0...], axis: -1).item(Int.self)
        pendingTokens = [firstToken]
        pendingIndex = 0
        committedLength = promptTokens.dim(1)
        lastTargetHidden = combinedHidden
    }
}
