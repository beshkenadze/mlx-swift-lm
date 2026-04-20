// Copyright © 2026 Apple Inc.

import MLX
import MLXNN
@testable import MLXLMCommon
import Testing

struct DFlashIteratorSpecRoundTests {

    private let blockSize = 16
    private let promptLength = 3
    private let hiddenSize = 4
    private let targetLayerIds = [0, 1]

    private struct SyntheticDraftConfig: DFlashDraftConfiguration {
        let blockSize: Int
        let maskTokenId: Int
        let targetLayerIds: [Int]
    }

    private final class SyntheticTargetModel: Module, LanguageModel, KVCacheDimensionProvider,
        DFlashTargetModel
    {
        let kvHeads: [Int] = [1]

        private let hiddenSize: Int
        private let vocabSize: Int
        private let prefillToken: Int32
        private let verifyPosterior: [Int32]

        init(hiddenSize: Int, vocabSize: Int, prefillToken: Int32, verifyPosterior: [Int32]) {
            self.hiddenSize = hiddenSize
            self.vocabSize = vocabSize
            self.prefillToken = prefillToken
            self.verifyPosterior = verifyPosterior
        }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }

        func sanitize(weights: [String : MLXArray]) -> [String : MLXArray] { weights }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            logits(for: MLXArray.zeros([inputs.dim(0), inputs.dim(1)], dtype: .int32))
        }

        func embedTokens(_ inputs: MLXArray) -> MLXArray {
            MLXArray.zeros([inputs.dim(0), inputs.dim(1), hiddenSize], dtype: .float32)
        }

        func applyLMHead(_ hidden: MLXArray) -> MLXArray {
            logits(for: hidden[0..., 0..., 0].asType(.int32))
        }

        func captureHiddenStatesAndLogits(
            inputs: MLXArray,
            layerIndices: [Int],
            cache: [KVCache]?
        ) -> (hiddenStates: [MLXArray], logits: MLXArray) {
            advance(cache: cache, by: inputs.dim(1))

            var hiddenStates: [MLXArray] = []
            hiddenStates.reserveCapacity(layerIndices.count)
            let elementCount = inputs.dim(0) * inputs.dim(1) * hiddenSize
            for (offset, _) in layerIndices.enumerated() {
                let values = Array(repeating: Float32(offset + 1), count: elementCount)
                hiddenStates.append(MLXArray(values).reshaped([inputs.dim(0), inputs.dim(1), hiddenSize]))
            }

            let tokenIds: MLXArray
            if inputs.dim(1) == verifyPosterior.count {
                tokenIds = MLXArray(verifyPosterior).reshaped([inputs.dim(0), inputs.dim(1)])
            } else {
                var prefillPosterior = Array(repeating: Int32(0), count: inputs.dim(1))
                if !prefillPosterior.isEmpty {
                    prefillPosterior[prefillPosterior.count - 1] = prefillToken
                }
                tokenIds = MLXArray(prefillPosterior).reshaped([inputs.dim(0), inputs.dim(1)])
            }

            return (hiddenStates, logits(for: tokenIds))
        }

        private func logits(for tokenIds: MLXArray) -> MLXArray {
            let vocab = MLXArray(0 ..< Int32(vocabSize)).reshaped([1, 1, vocabSize])
            return (expandedDimensions(tokenIds, axis: -1) .== vocab).asType(.float32)
        }

        private func advance(cache: [KVCache]?, by length: Int) {
            guard let cache, length > 0 else { return }
            let keys = MLXArray.zeros([1, 1, length, 1], dtype: .float32)
            let values = MLXArray.zeros([1, 1, length, 1], dtype: .float32)
            for entry in cache {
                _ = entry.update(keys: keys, values: values)
            }
        }
    }

    private final class SyntheticDrafter: Module, DFlashDraftingModel {
        let numDraftLayers: Int

        private let hiddenSize: Int
        private let proposalTokens: [Int32]

        init(hiddenSize: Int, proposalTokens: [Int32], numDraftLayers: Int = 1) {
            self.hiddenSize = hiddenSize
            self.proposalTokens = proposalTokens
            self.numDraftLayers = numDraftLayers
        }

        func callAsFunction(
            noiseEmbedding: MLXArray,
            targetHidden: MLXArray,
            caches: [KVCache?]
        ) -> MLXArray {
            let appendedLength = targetHidden.dim(1) + noiseEmbedding.dim(1)
            if appendedLength > 0 {
                let keys = MLXArray.zeros([1, 1, appendedLength, 1], dtype: .float32)
                let values = MLXArray.zeros([1, 1, appendedLength, 1], dtype: .float32)
                for cache in caches {
                    guard let cache else { continue }
                    _ = cache.update(keys: keys, values: values)
                }
            }

            let batch = noiseEmbedding.dim(0)
            let seqLen = noiseEmbedding.dim(1)
            var hidden = Array(repeating: Float32(0), count: batch * seqLen * hiddenSize)
            for batchIndex in 0..<batch {
                for position in 1..<seqLen {
                    let index = batchIndex * seqLen * hiddenSize + position * hiddenSize
                    hidden[index] = Float32(proposalTokens[position - 1])
                }
            }
            return MLXArray(hidden).reshaped([batch, seqLen, hiddenSize])
        }
    }

    func makeIterator(
        proposals: [Int32],
        posterior: [Int32],
        firstPrefillToken: Int32 = 3,
        stopTokenIds: Set<Int> = [],
        maxTokens: Int? = nil
    ) throws -> DFlashIterator {
        let target = SyntheticTargetModel(
            hiddenSize: hiddenSize,
            vocabSize: 64,
            prefillToken: firstPrefillToken,
            verifyPosterior: posterior
        )
        let drafter = SyntheticDrafter(hiddenSize: hiddenSize, proposalTokens: proposals)
        let draftConfig = SyntheticDraftConfig(
            blockSize: blockSize,
            maskTokenId: 63,
            targetLayerIds: targetLayerIds
        )
        let promptTokens = MLXArray([Int32(11), 12, 13]).reshaped([1, promptLength])
        return try DFlashIterator(
            promptTokens: promptTokens,
            target: target,
            drafter: drafter,
            draftConfig: draftConfig,
            stopTokenIds: stopTokenIds,
            maxTokens: maxTokens
        )
    }

    @Test(
        "DFlashIterator speculation round fully accepts a matching draft block",
        .disabled(if: !dflashMetallibAvailable, "MLX metallib unavailable")
    )
    func testSpeculationRoundPerfectMatch() throws {
        let acceptedToken = Int32(7)
        var iterator = try makeIterator(
            proposals: Array(repeating: acceptedToken, count: blockSize - 1),
            posterior: Array(repeating: acceptedToken, count: blockSize)
        )

        #expect(iterator.next() == 3)
        #expect(iterator.next() == Int(acceptedToken))

        #expect(iterator.totalProposed == blockSize - 1)
        #expect(iterator.totalAccepted == blockSize - 1)
        #expect(iterator.pendingTokens.count == blockSize)
        #expect(iterator.pendingIndex == 1)
        #expect(iterator.committedLength == promptLength + blockSize)
        #expect(iterator.targetCache.first?.offset == iterator.committedLength)
        #expect(iterator.draftCache.first?.offset == promptLength)

        let lastTargetHidden = try #require(iterator.lastTargetHidden)
        #expect(lastTargetHidden.shape == [1, blockSize, targetLayerIds.count * hiddenSize])
    }

    @Test(
        "DFlashIterator speculation round commits only the bonus token on immediate mismatch",
        .disabled(if: !dflashMetallibAvailable, "MLX metallib unavailable")
    )
    func testSpeculationRoundZeroMatch() throws {
        let bonusToken = Int32(5)
        var proposals = Array(repeating: Int32(9), count: blockSize - 1)
        proposals[0] = 8
        var posterior = Array(repeating: Int32(4), count: blockSize)
        posterior[0] = bonusToken

        var iterator = try makeIterator(proposals: proposals, posterior: posterior)

        #expect(iterator.next() == 3)
        #expect(iterator.next() == Int(bonusToken))

        #expect(iterator.totalProposed == blockSize - 1)
        #expect(iterator.totalAccepted == 0)
        #expect(iterator.pendingTokens.count == 1)
        #expect(iterator.pendingIndex == 1)
        #expect(iterator.committedLength == promptLength + 1)
        #expect(iterator.targetCache.first?.offset == iterator.committedLength)
        #expect(iterator.draftCache.first?.offset == promptLength)

        let lastTargetHidden = try #require(iterator.lastTargetHidden)
        #expect(lastTargetHidden.shape == [1, 1, targetLayerIds.count * hiddenSize])
    }
}
