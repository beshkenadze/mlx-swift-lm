// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
@testable import MLXLMCommon
import Testing

struct DFlashIteratorPrefillTests {

    private func makeTargetModel() throws -> Qwen3Model {
        let config = try JSONDecoder().decode(
            Qwen3Configuration.self,
            from: Data(
                """
                {
                  "hidden_size": 32,
                  "num_hidden_layers": 5,
                  "intermediate_size": 64,
                  "num_attention_heads": 4,
                  "rms_norm_eps": 0.000001,
                  "vocab_size": 256,
                  "num_key_value_heads": 2,
                  "head_dim": 8,
                  "tie_word_embeddings": true
                }
                """.utf8))
        let model = Qwen3Model(config)
        MLX.eval(model)
        return model
    }

    private func makeDraftConfig() -> DFlashDraftConfig {
        DFlashDraftConfig(
            modelType: "qwen3",
            blockSize: 16,
            hiddenSize: 32,
            intermediateSize: 64,
            numHiddenLayers: 5,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 8,
            numTargetLayers: 5,
            maxPositionEmbeddings: 4_096,
            ropeTheta: 1_000_000,
            rmsNormEps: 0.000001,
            tieWordEmbeddings: true,
            vocabSize: 256,
            dflashConfig: .init(maskTokenId: 255, targetLayerIds: [0, 1, 2, 3, 4])
        )
    }

    @Test(
        "DFlashIterator prefill captures hidden state and buffers first token",
        .disabled(if: !(dflashHFTestsEnabled && dflashMetallibAvailable), "HF/metallib gate disabled")
    )
    func testPrefillSeedsPendingTokenAndTargetHidden() throws {
        let draftConfig = makeDraftConfig()
        let target = try makeTargetModel()
        let drafter = DFlashDraftModel(draftConfig)
        MLX.eval(drafter)

        let promptTokens = MLXArray([1, 2, 3]).reshaped(1, 3)
        var iterator = try DFlashIterator(
            promptTokens: promptTokens,
            target: target,
            drafter: drafter,
            draftConfig: draftConfig,
            stopTokenIds: [],
            maxTokens: 4
        )

        let firstToken = iterator.next()

        #expect(firstToken != nil)
        #expect(iterator.pendingIndex == 1)
        #expect(iterator.committedLength == 3)

        let lastTargetHidden = try #require(iterator.lastTargetHidden)
        #expect(lastTargetHidden.shape == [1, 3, draftConfig.dflashConfig.targetLayerIds.count * 32])
    }
}
