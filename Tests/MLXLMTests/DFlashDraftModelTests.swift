// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import Testing

struct DFlashDraftModelTests {

    @Test(
        "DFlash draft model returns [batch, blockSize, hiddenSize]",
        .disabled(if: !dflashMetallibAvailable, "MLX metallib unavailable")
    )
    func testDFlashDraftForwardShape() throws {
        let config = makeTestDFlashConfig()
        let model = DFlashDraftModel(config)
        let batchSize = 1
        let blockSize = config.blockSize
        let contextLength = 24
        let targetWidth = config.dflashConfig.targetLayerIds.count * config.hiddenSize
        let caches: [KVCache?] = Array(repeating: nil, count: config.numHiddenLayers)

        let noiseEmbedding = MLXRandom.normal([batchSize, blockSize, config.hiddenSize], dtype: .float32)
        let targetHidden = MLXRandom.normal([batchSize, contextLength, targetWidth], dtype: .float32)

        let output = model(
            noiseEmbedding: noiseEmbedding,
            targetHidden: targetHidden,
            caches: caches
        )
        MLX.eval(output)

        #expect(output.shape == [batchSize, blockSize, config.hiddenSize])
    }
}
