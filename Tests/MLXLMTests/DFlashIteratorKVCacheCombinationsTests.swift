// Copyright © 2026 Apple Inc.

import MLX
import MLXNN
@testable import MLXLMCommon
import Testing

/// Structural compatibility gate for `DFlashIterator` across target-cache backends.
///
/// The synthetic target vends each cache implementation through `newCache()`, so the
/// iterator sees the same cache protocol surface it would in production. When MLX
/// metallib is unavailable, the runtime assertions are skipped, but the test still
/// provides compile coverage for plain, TurboQuant, and TriAttention cache wiring.
struct DFlashIteratorKVCacheCombinationsTests {

    private let blockSize = 16
    private let promptLength = 3
    private let hiddenSize = 4
    private let targetLayerIds = [0, 1]

    enum TargetCacheKind: String, CaseIterable {
        case plain = "plain"
        case turboQuant3 = "turbo-quant-3bit"
        case turboQuant4 = "turbo-quant-4bit"
        case triAttention = "tri-attention"
    }

    private struct SyntheticDraftConfig: DFlashDraftConfiguration {
        let blockSize: Int
        let maskTokenId: Int
        let targetLayerIds: [Int]
    }

    private final class CacheBackedSyntheticTargetModel: Module, LanguageModel,
        KVCacheDimensionProvider, DFlashTargetModel
    {
        let kvHeads: [Int] = [1]

        private let hiddenSize: Int
        private let vocabSize: Int
        private let prefillToken: Int32
        private let verifyPosterior: [Int32]
        private let targetCacheFactory: @Sendable () -> any KVCache

        init(
            hiddenSize: Int,
            vocabSize: Int,
            prefillToken: Int32,
            verifyPosterior: [Int32],
            targetCacheFactory: @escaping @Sendable () -> any KVCache
        ) {
            self.hiddenSize = hiddenSize
            self.vocabSize = vocabSize
            self.prefillToken = prefillToken
            self.verifyPosterior = verifyPosterior
            self.targetCacheFactory = targetCacheFactory
        }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .tokens(input.text)
        }

        func sanitize(weights: [String : MLXArray]) -> [String : MLXArray] { weights }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            logits(for: MLXArray.zeros([inputs.dim(0), inputs.dim(1)], dtype: .int32))
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            [targetCacheFactory()]
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

    private static func makeTriAttentionConfiguration() -> TriAttentionConfiguration {
        let layer = TriAttentionLayerCalibration(
            qCenterReal: MLXArray.zeros([1, 1], dtype: .float32),
            qCenterImag: MLXArray.zeros([1, 1], dtype: .float32),
            qMeanNorm: MLXArray.ones([1, 1], dtype: .float32)
        )
        let calibration = TriAttentionCalibrationData(layers: [layer], qHeads: 1, kvHeads: 1)
        let rope = TriAttentionRoPEConfig(
            headDim: 1,
            rotatedDims: 0,
            traditional: false,
            omega: MLXArray.zeros([0], dtype: .float32)
        )
        return TriAttentionConfiguration(
            calibration: calibration,
            rope: rope,
            budget: 64,
            divideLength: 64,
            protectRecent: 1,
            protectInitial: 1
        )
    }

    private static func makeTargetCache(kind: TargetCacheKind) -> any KVCache {
        switch kind {
        case .plain:
            KVCacheSimple()
        case .turboQuant3:
            TurboQuantKVCache(bits: 3, seed: 0)
        case .turboQuant4:
            TurboQuantKVCache(bits: 4, seed: 0)
        case .triAttention:
            TriAttentionCache(
                base: KVCacheSimple(),
                configuration: makeTriAttentionConfiguration(),
                layerIndex: 0
            )
        }
    }

    private func makeIterator(cacheKind: TargetCacheKind) throws -> DFlashIterator {
        var proposals = Array(repeating: Int32(9), count: blockSize - 1)
        proposals[0] = 7
        proposals[1] = 7

        var posterior = Array(repeating: Int32(5), count: blockSize)
        posterior[0] = 7
        posterior[1] = 7
        posterior[2] = 5

        let target = CacheBackedSyntheticTargetModel(
            hiddenSize: hiddenSize,
            vocabSize: 64,
            prefillToken: 3,
            verifyPosterior: posterior,
            targetCacheFactory: { Self.makeTargetCache(kind: cacheKind) }
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
            draftConfig: draftConfig
        )
    }

    @Test(
        "DFlashIterator can run against plain, TurboQuant, and TriAttention target caches",
        .disabled(if: !dflashMetallibAvailable, "MLX metallib unavailable"),
        arguments: TargetCacheKind.allCases
    )
    func testIteratorWorksAcrossTargetCacheBackends(cacheKind: TargetCacheKind) throws {
        var iterator = try makeIterator(cacheKind: cacheKind)

        #expect(iterator.next() == 3)
        #expect(iterator.next() == 7)

        #expect(iterator.totalProposed == blockSize - 1)
        #expect(iterator.totalAccepted == 2)
        #expect(iterator.pendingTokens.count == 3)
        #expect(iterator.pendingIndex == 1)
        #expect(iterator.committedLength == promptLength + 3)

        let lastTargetHidden = try #require(iterator.lastTargetHidden)
        #expect(lastTargetHidden.shape == [1, 3, targetLayerIds.count * hiddenSize])

        let targetCache = try #require(iterator.targetCache.first)
        #expect(targetCache.offset == iterator.committedLength)

        switch cacheKind {
        case .plain:
            #expect(targetCache is KVCacheSimple)
        case .turboQuant3:
            let turbo = try #require(targetCache as? TurboQuantKVCache)
            #expect(turbo.turboQuantBits == 3)
            #expect(turbo.offset == iterator.committedLength)
        case .turboQuant4:
            let turbo = try #require(targetCache as? TurboQuantKVCache)
            #expect(turbo.turboQuantBits == 4)
            #expect(turbo.offset == iterator.committedLength)
        case .triAttention:
            let triAttention = try #require(targetCache as? TriAttentionCache)
            #expect(triAttention.offset == iterator.committedLength)
            #expect(triAttention.base is KVCacheSimple)
            #expect(triAttention.base.offset == iterator.committedLength)
        }
    }
}
