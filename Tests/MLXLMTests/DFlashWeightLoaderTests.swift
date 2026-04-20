// Copyright © 2026 Apple Inc.

import Foundation
import MLXLLM
import Testing

private let dflashHFTestsEnabled = ProcessInfo.processInfo.environment["DFLASH_TEST_HF"] != nil

struct DFlashWeightLoaderTests {

    @Test(
        "DFlash weight loader downloads and loads z-lab/Qwen3-4B-DFlash-b16",
        .enabled(if: dflashHFTestsEnabled && dflashMetallibAvailable)
    )
    func testLoadQwen34BDFlashDraftFromHuggingFace() async throws {
        let (model, config) = try await DFlashWeightLoader.load()

        #expect(config.modelType == "qwen3")
        #expect(config.blockSize == 16)
        #expect(config.numHiddenLayers == 5)
        #expect(config.dflashConfig.maskTokenId == 151_669)
        #expect(config.dflashConfig.targetLayerIds == [1, 9, 17, 25, 33])
        #expect(model.config.modelType == config.modelType)
        #expect(model.config.hiddenSize == config.hiddenSize)
        #expect(model.config.dflashConfig.targetLayerIds == config.dflashConfig.targetLayerIds)
    }
}
