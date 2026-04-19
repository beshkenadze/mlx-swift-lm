// Copyright © 2026 Apple Inc.

import Foundation
import Metal
import MLXLLM

let dflashMetallibAvailable: Bool = {
    if ProcessInfo.processInfo.environment["MLX_METALLIB_OK"] == "0" {
        return false
    }

    guard MTLCreateSystemDefaultDevice() != nil else { return false }

    let fileManager = FileManager.default
    let candidateNames = ["mlx.metallib", "default.metallib"]
    var roots: [URL] = [Bundle.main.bundleURL]
    if let resourceURL = Bundle.main.resourceURL {
        roots.append(resourceURL)
    }
    for bundle in Bundle.allBundles {
        roots.append(bundle.bundleURL)
        if let resourceURL = bundle.resourceURL {
            roots.append(resourceURL)
        }
    }

    for root in roots {
        for name in candidateNames {
            if fileManager.fileExists(atPath: root.appendingPathComponent(name).path) {
                return true
            }
            if let entries = try? fileManager.contentsOfDirectory(atPath: root.path) {
                for entry in entries where entry.hasSuffix(".bundle") {
                    let nested = root.appendingPathComponent(entry).appendingPathComponent(name)
                    if fileManager.fileExists(atPath: nested.path) {
                        return true
                    }
                }
            }
        }
    }

    return false
}()

func makeTestDFlashConfig() -> DFlashDraftConfig {
    DFlashDraftConfig(
        modelType: "qwen3",
        blockSize: 16,
        hiddenSize: 32,
        intermediateSize: 64,
        numHiddenLayers: 5,
        numAttentionHeads: 4,
        numKeyValueHeads: 2,
        headDim: 8,
        numTargetLayers: 36,
        maxPositionEmbeddings: 40_960,
        ropeTheta: 1_000_000,
        rmsNormEps: 0.000001,
        tieWordEmbeddings: true,
        vocabSize: 128,
        dflashConfig: .init(maskTokenId: 151_669, targetLayerIds: [1, 9, 17, 25, 33])
    )
}
