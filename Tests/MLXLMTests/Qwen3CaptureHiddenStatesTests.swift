// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

#if canImport(Metal)
    import Metal
#endif

/// True when MLX's Metal backend + metallib file are reachable. Tests that run
/// forward passes are skipped otherwise — same pattern as `MLXLMServerTests`.
private let metallibAvailable: Bool = {
    if ProcessInfo.processInfo.environment["MLX_METALLIB_OK"] == "0" {
        return false
    }
    #if canImport(Metal)
        guard MTLCreateSystemDefaultDevice() != nil else { return false }
    #else
        return false
    #endif

    let fm = FileManager.default
    let candidateNames = ["mlx.metallib", "default.metallib"]
    var roots: [URL] = [Bundle.main.bundleURL]
    if let r = Bundle.main.resourceURL { roots.append(r) }
    for b in Bundle.allBundles {
        roots.append(b.bundleURL)
        if let r = b.resourceURL { roots.append(r) }
    }
    for root in roots {
        for name in candidateNames {
            if fm.fileExists(atPath: root.appendingPathComponent(name).path) {
                return true
            }
            if let contents = try? fm.contentsOfDirectory(atPath: root.path) {
                for entry in contents where entry.hasSuffix(".bundle") {
                    let nested = root.appendingPathComponent(entry)
                        .appendingPathComponent(name)
                    if fm.fileExists(atPath: nested.path) { return true }
                }
            }
        }
    }
    return false
}()

struct Qwen3CaptureHiddenStatesTests {

    private func makeModel(layers: Int = 4) throws -> Qwen3Model {
        // Tiny synthetic Qwen3 — no pretrained weights required, parameters are
        // initialized randomly which is sufficient for shape/count assertions.
        let config = try JSONDecoder().decode(
            Qwen3Configuration.self,
            from: Data(
                """
                {
                  "hidden_size": 32,
                  "num_hidden_layers": \(layers),
                  "intermediate_size": 48,
                  "num_attention_heads": 4,
                  "rms_norm_eps": 0.000001,
                  "vocab_size": 64,
                  "num_key_value_heads": 2,
                  "head_dim": 8
                }
                """.utf8))
        let model = Qwen3Model(config)
        MLX.eval(model)
        return model
    }

    @Test(
        "captureHiddenStates returns requested layer outputs with correct shape",
        .disabled(if: !metallibAvailable, "MLX metallib unavailable")
    )
    func testCaptureHiddenStatesShapes() throws {
        let model = try makeModel()
        let seqLen = 5
        let hiddenSize = 32
        let inputs = MLXArray(
            (0 ..< seqLen).map { Int32($0 % 16) }
        ).reshaped(1, seqLen)

        let captures = model.captureHiddenStates(
            inputs: inputs,
            layerIndices: [1, 3],
            cache: nil
        )
        MLX.eval(captures)

        #expect(captures.count == 2)
        #expect(captures[0].shape == [1, seqLen, hiddenSize])
        #expect(captures[1].shape == [1, seqLen, hiddenSize])
    }

    @Test(
        "captureHiddenStates order matches requested indices",
        .disabled(if: !metallibAvailable, "MLX metallib unavailable")
    )
    func testCaptureHiddenStatesOrder() throws {
        let model = try makeModel()
        let seqLen = 3
        let inputs = MLXArray(
            (0 ..< seqLen).map { Int32($0) }
        ).reshaped(1, seqLen)

        let forward = model.captureHiddenStates(
            inputs: inputs, layerIndices: [0, 1, 2, 3], cache: nil)
        let reverse = model.captureHiddenStates(
            inputs: inputs, layerIndices: [3, 2, 1, 0], cache: nil)
        MLX.eval(forward)
        MLX.eval(reverse)

        // reverse[i] should equal forward[3 - i] (same forward pass, different
        // ordering of captured outputs).
        for i in 0 ..< 4 {
            let diff = (reverse[i] - forward[3 - i]).abs().max().item(Float.self)
            #expect(diff == 0.0)
        }
    }

    @Test(
        "lmHeadWeight is nil when tieWordEmbeddings is true, non-nil otherwise",
        .disabled(if: !metallibAvailable, "MLX metallib unavailable")
    )
    func testLMHeadWeightAccessor() throws {
        let untied = try JSONDecoder().decode(
            Qwen3Configuration.self,
            from: Data(
                """
                {
                  "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                  "num_attention_heads": 4, "rms_norm_eps": 0.000001,
                  "vocab_size": 64, "num_key_value_heads": 2, "head_dim": 8,
                  "tie_word_embeddings": false
                }
                """.utf8))
        let tied = try JSONDecoder().decode(
            Qwen3Configuration.self,
            from: Data(
                """
                {
                  "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 48,
                  "num_attention_heads": 4, "rms_norm_eps": 0.000001,
                  "vocab_size": 64, "num_key_value_heads": 2, "head_dim": 8,
                  "tie_word_embeddings": true
                }
                """.utf8))

        let untiedModel = Qwen3Model(untied)
        let tiedModel = Qwen3Model(tied)
        MLX.eval(untiedModel, tiedModel)

        #expect(untiedModel.lmHeadWeight != nil)
        #expect(untiedModel.lmHeadWeight?.shape == [64, 32])
        #expect(tiedModel.lmHeadWeight == nil)
        #expect(tiedModel.tiedEmbeddingWeight.shape == [64, 32])
    }
}
