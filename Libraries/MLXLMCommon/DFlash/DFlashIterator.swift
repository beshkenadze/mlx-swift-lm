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
    var isTerminated: Bool

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
        self.isTerminated = false
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

        if isTerminated {
            return nil
        }

        guard firstStepDone else {
            return nil
        }

        runOneSpeculationRound()
        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            emittedCount += 1
            return token
        }
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
        if stopTokenIds.contains(firstToken) {
            isTerminated = true
        }
    }

    private mutating func runOneSpeculationRound() {
        // INVARIANTS after every round:
        //   committedLength = prompt_len + total_emitted_decode_tokens
        //   target_cache.offset == committedLength
        //   committedCount = pendingTokens.count
        //   draft_cache.offset == committedLength - committedCount
        //                      == committedLength - lastTargetHidden.shape[1]
        //                      == previous round's committedLength
        //   lastTargetHidden.shape == [1, committedCount, sum_of_target_layer_hidden_sizes]
        //
        // The draft cache intentionally lags behind committedLength by (acceptanceLen+1).
        // DFlash's drafter re-receives accepted context each round via `targetHidden` (a slice
        // of the verify hidden states), not via its own KV cache. Mirrors HF `crop(start)` semantics.
        //
        // OFF-BY-ONE SPEC:
        //   block[0] = last_committed_token (already in target_cache from prev round)
        //   block[1..<16] = mask OR draft proposals
        //   draft forward uses positions [committedLength..<committedLength+16]
        //   target verify forward uses positions [committedLength..<committedLength+16]
        //   BUT: target_cache already has block[0] from previous verify's last position.
        //        So verify runs block as a fresh 16-token forward; we trim blockSize-(acc_len+1) at end.
        //   posterior[i] = target's prediction for position (committedLength + i + 1) given [..., block[i]]
        //   block[1..<16][i] = draft_tokens[i]  (15 proposals)
        //   Greedy match:
        //     for i in 0..<15:
        //       if block[i+1] == posterior[i]: accept += 1; else: break
        //   Commit: block[1..<1+acc_len] + [posterior[acc_len]]  (bonus)
        guard firstStepDone,
            let lastHidden = lastTargetHidden,
            let lastCommitted = pendingTokens.last
        else {
            return
        }

        var blockIds: [Int32] = [Int32(lastCommitted)]
        blockIds.append(contentsOf: repeatElement(Int32(maskTokenId), count: blockSize - 1))
        let initialBlock = MLXArray(blockIds).reshaped([1, blockSize])

        let noiseEmbedding = target.embedTokens(initialBlock)
        let draftHidden = drafter(
            noiseEmbedding: noiseEmbedding,
            targetHidden: lastHidden,
            caches: draftCache.map { Optional($0) }
        )
        let draftLogits = target.applyLMHead(draftHidden[0..., 1..., 0...])
        let draftTokensArray = argMax(draftLogits, axis: -1).asType(.int32)
        eval(draftTokensArray)

        let draftToTrim = Swift.max(
            0,
            (draftCache.first?.offset ?? committedLength) - committedLength
        )
        if draftToTrim > 0 {
            for cache in draftCache {
                _ = cache.trim(draftToTrim)
            }
        }

        var filled: [Int32] = [Int32(lastCommitted)]
        for i in 0..<(blockSize - 1) {
            filled.append(draftTokensArray[0, i].item(Int32.self))
        }
        let filledBlock = MLXArray(filled).reshaped([1, blockSize])

        let verifyResult = target.captureHiddenStatesAndLogits(
            inputs: filledBlock,
            layerIndices: targetLayerIds,
            cache: targetCache
        )
        let posterior = argMax(verifyResult.logits, axis: -1).asType(.int32)
        eval(posterior)

        var acceptanceLen = 0
        for i in 0..<(blockSize - 1) {
            let draftProposal = filled[i + 1]
            let targetPrediction = posterior[0, i].item(Int32.self)
            if draftProposal == targetPrediction {
                acceptanceLen += 1
            } else {
                break
            }
        }
        totalProposed += blockSize - 1

        var newTokens: [Int] = []
        newTokens.reserveCapacity(acceptanceLen + 1)
        for i in 0..<acceptanceLen {
            newTokens.append(Int(filled[i + 1]))
        }
        let bonus = Int(posterior[0, acceptanceLen].item(Int32.self))
        newTokens.append(bonus)
        if let stopIndex = newTokens.firstIndex(where: { stopTokenIds.contains($0) }) {
            newTokens.removeSubrange((stopIndex + 1)...)
            isTerminated = true
        }
        if let maxTokens {
            let remainingBudget = maxTokens - emittedCount
            if newTokens.count > remainingBudget {
                newTokens.removeSubrange(remainingBudget...)
            }
        }
        let committedCount = newTokens.count
        let committedAccepted = Swift.min(acceptanceLen, committedCount)
        totalAccepted += committedAccepted
        pendingTokens = newTokens
        pendingIndex = 0

        let targetExtra = blockSize - committedCount
        if targetExtra > 0 {
            for cache in targetCache {
                _ = cache.trim(targetExtra)
            }
        }
        committedLength += committedCount

        let combinedVerifyHidden =
            if verifyResult.hiddenStates.count == 1 {
                verifyResult.hiddenStates[0]
            } else {
                concatenated(verifyResult.hiddenStates, axis: -1)
            }
        let nextHidden = combinedVerifyHidden[0..., 0..<committedCount, 0...]
        eval(nextHidden)
        lastTargetHidden = nextHidden
    }
}
