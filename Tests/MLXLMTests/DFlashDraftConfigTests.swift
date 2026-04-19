// Copyright © 2026 Apple Inc.

import Foundation
import MLXLLM
import Testing

struct DFlashDraftConfigTests {

    @Test("DFlash draft config decodes the canonical Qwen3-4B fixture")
    func testDFlashDraftConfigDecodesQwen34B() throws {
        let url = try #require(
            Bundle.module.url(forResource: "dflash-qwen3-4b-config", withExtension: "json"))
        let data = try Data(contentsOf: url)

        let config = try JSONDecoder().decode(DFlashDraftConfig.self, from: data)

        #expect(config.modelType == "qwen3")
        #expect(config.blockSize == 16)
        #expect(config.hiddenSize == 2_560)
        #expect(config.intermediateSize == 9_728)
        #expect(config.numHiddenLayers == 5)
        #expect(config.numAttentionHeads == 32)
        #expect(config.numKeyValueHeads == 8)
        #expect(config.headDim == 128)
        #expect(config.numTargetLayers == 36)
        #expect(config.maxPositionEmbeddings == 40_960)
        #expect(config.ropeTheta == 1_000_000)
        #expect(config.rmsNormEps == 0.000001)
        #expect(config.attentionBias == false)
        #expect(config.tieWordEmbeddings)
        #expect(config.vocabSize == 151_936)
        #expect(config.dflashConfig.maskTokenId == 151_669)
        #expect(config.dflashConfig.targetLayerIds == [1, 9, 17, 25, 33])
    }
}
